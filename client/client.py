"""
Real-time Audio Enhancement Client

This client captures audio from the default microphone, sends it to the DeepFilterNet
server via WebSocket for enhancement, and outputs the enhanced audio to a virtual
microphone device.

Configuration:
- SAMPLE_RATE: 48000 Hz (matches DeepFilterNet model requirements)
- BLOCK_SIZE: 4800 samples (100ms blocks for low latency)
- CHANNELS: 1 (mono audio)
- DTYPE: int16 (16-bit PCM audio)
- INPUT_DEVICE: None (uses system default microphone)
- OUTPUT_DEVICE: None (uses system default output, or set to virtual mic device index)

Virtual Audio Device Setup:
==========================

WINDOWS - VB-Cable:
  Install: choco install vb-cable -y
  Or download from: https://vb-audio.com/Cable/
  Device name: "CABLE Input" (output) / "CABLE Output" (input)
  Example: OUTPUT_DEVICE = 4  # Check with --list-devices

macOS - BlackHole:
  Install: brew install blackhole-2ch
  Or download from: https://github.com/ExistentialAudio/BlackHole
  Device name: "BlackHole 2ch"
  Example: OUTPUT_DEVICE = 2  # Check with --list-devices

LINUX - ALSA Loopback:
  Install: sudo modprobe snd-aloop
  Make permanent: echo "snd-aloop" | sudo tee /etc/modules-load.d/snd-aloop.conf
  Device name: "Loopback"
  Example: OUTPUT_DEVICE = 6  # Check with --list-devices

Finding Device Indices:
======================
Run: python client/client.py --list-devices
Or: python -m sounddevice

Look for:
- INPUT_DEVICE: Your physical microphone (e.g., "Microphone (Realtek...)")
- OUTPUT_DEVICE: Your virtual audio device (e.g., "CABLE Input", "BlackHole 2ch", "Loopback")

Then either:
1. Use command-line: python client/client.py --input-device 1 --output-device 4
2. Or edit the constants below with your device indices

See VIRTUAL_AUDIO_SETUP.md for detailed setup instructions.

Architecture:
- Audio callback runs in a separate thread (sounddevice requirement)
- WebSocket communication uses asyncio
- Thread-safe queue bridges the audio callback and async WebSocket
"""

import asyncio
import numpy as np
import sounddevice as sd
import websockets
from queue import Queue
import threading
import argparse
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import wave
from pathlib import Path
from fractions import Fraction
import struct
import json
import os
try:
    from scipy.signal import resample_poly  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _resample_audio(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1-D audio using scipy if available, else numpy interp.
    Returns int16 array.
    """
    if orig_sr == target_sr:
        return x.astype(np.int16)
    if _HAS_SCIPY:
        ratio = Fraction(target_sr, orig_sr).limit_denominator(1000)
        up, down = ratio.numerator, ratio.denominator
        y = resample_poly(x.astype(np.float32), up, down)
        return np.clip(y, -32768, 32767).astype(np.int16)
    # Fallback: linear interpolation
    n = x.shape[0]
    duration = n / float(orig_sr)
    new_n = int(round(duration * target_sr))
    if new_n <= 1 or n <= 1:
        return x.astype(np.int16)
    t_old = np.linspace(0.0, duration, num=n, endpoint=False, dtype=np.float64)
    t_new = np.linspace(0.0, duration, num=new_n, endpoint=False, dtype=np.float64)
    y = np.interp(t_new, t_old, x.astype(np.float32))
    return np.clip(y, -32768, 32767).astype(np.int16)


def _calculate_db_level(audio_block: np.ndarray) -> float:
    """Calculate audio level in dB from int16 samples."""
    audio_float = audio_block.astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(audio_float ** 2) + 1e-10)
    db = 20.0 * np.log10(rms + 1e-10)
    return float(db)


def _apply_gain(audio_block: np.ndarray, gain: float) -> np.ndarray:
    """Apply gain to int16 audio block with clipping."""
    audio_float = audio_block.astype(np.float32) * gain
    return np.clip(audio_float, -32768, 32767).astype(np.int16)


def _apply_noise_gate(audio_block: np.ndarray, threshold_db: float) -> np.ndarray:
    """Apply noise gate - silence audio below threshold."""
    level_db = _calculate_db_level(audio_block)
    if level_db < threshold_db:
        return np.zeros_like(audio_block)
    return audio_block


def _mix_audio(dry: np.ndarray, wet: np.ndarray, mix: float) -> np.ndarray:
    """Mix dry (original) and wet (processed) audio.
    mix=0.0 is 100% dry, mix=1.0 is 100% wet.
    """
    dry_float = dry.astype(np.float32)
    wet_float = wet.astype(np.float32)
    mixed = dry_float * (1.0 - mix) + wet_float * mix
    return np.clip(mixed, -32768, 32767).astype(np.int16)

# Audio configuration
SAMPLE_RATE = 48000  # Hz - DeepFilterNet expects 48kHz
# Reduced block size for lower algorithmic latency (20ms @ 48kHz = 960 samples)
BLOCK_SIZE = 960     # samples ~20ms
CHANNELS = 1         # mono audio
DTYPE = np.int16     # 16-bit PCM

# Device configuration (None = system default)
# To find your device indices, run: python client/client.py --list-devices
INPUT_DEVICE = None   # Physical microphone input device index (e.g., 1)
OUTPUT_DEVICE = None  # Virtual audio device output index (e.g., 4 for VB-Cable, 2 for BlackHole)
                      # Windows: CABLE Input (VB-Cable)
                      # macOS: BlackHole 2ch
                      # Linux: Loopback device

# WebSocket configuration
SERVER_URL = "ws://localhost:8000/enhance"

# Thread-safe queues for audio data
input_queue = Queue(maxsize=32)   # Captured audio blocks (larger to absorb jitter)
output_queue = Queue(maxsize=32)  # Enhanced audio blocks

# Control flags
running = True
stats_lock = threading.Lock()
stats = {
    "blocks_sent": 0,
    "blocks_received": 0,
    "errors": 0,
    "avg_latency_ms": 0.0,
    "last_latency_ms": 0.0
}

# Connection status for audio callback control
ws_connected = threading.Event()
_last_queue_warn = 0.0

# Audio processing controls (thread-safe with locks)
processing_lock = threading.Lock()
processing_params = {
    "input_gain": 1.0,      # Pre-processing gain multiplier (0.0 - 2.0)
    "output_gain": 1.0,     # Post-processing gain multiplier (0.0 - 2.0)
    "dry_wet_mix": 1.0,     # 0.0=dry (original), 1.0=wet (enhanced)
    "pass_through": False,  # True=bypass enhancement entirely
    "noise_gate_threshold": -60.0,  # dB threshold for noise gate (disabled if <= -100)
    "enable_recording": False,      # Enable audio recording
}

# Audio level monitoring
audio_levels = {
    "input_level": -100.0,   # dB
    "output_level": -100.0,  # dB
}

# Recording buffer
recording_buffer = []
recording_lock = threading.Lock()


def audio_callback(indata, outdata, frames, time_info, status):
    """
    Audio callback function called by sounddevice in a separate thread.
    
    Args:
        indata: Input audio data from microphone (numpy array)
        outdata: Output audio data to virtual mic (numpy array to fill)
        frames: Number of frames per block
        time_info: Timing information
        status: Status flags
    """
    global running, stats
    
    if status:
        print(f"Audio callback status: {status}")
    
    if not running:
        outdata.fill(0)
        return
    
    try:
        # Convert to 1D array and copy to avoid reference issues
        audio_block = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        
        # Get current processing parameters (thread-safe)
        with processing_lock:
            input_gain = processing_params["input_gain"]
            output_gain = processing_params["output_gain"]
            dry_wet_mix = processing_params["dry_wet_mix"]
            pass_through = processing_params["pass_through"]
            noise_gate_threshold = processing_params["noise_gate_threshold"]
            enable_recording = processing_params["enable_recording"]
        
        # Calculate and store input level
        input_level = _calculate_db_level(audio_block)
        audio_levels["input_level"] = input_level
        
        # Apply input gain
        if input_gain != 1.0:
            audio_block = _apply_gain(audio_block, input_gain)
        
        # Apply noise gate to input if enabled
        if noise_gate_threshold > -100.0:
            audio_block = _apply_noise_gate(audio_block, noise_gate_threshold)
        
        # Store original for dry/wet mixing
        original_block = audio_block.copy()
        
        # If pass-through mode, skip enhancement
        if pass_through:
            output_block = audio_block
        else:
            # If websocket is not connected, use original audio
            if not ws_connected.is_set():
                output_block = audio_block
            else:
                # Send to enhancement queue (non-blocking)
                if not input_queue.full():
                    input_queue.put(audio_block)
                else:
                    # Rate-limit warnings to ~1/sec
                    global _last_queue_warn
                    import time as _time
                    now = _time.time()
                    if now - _last_queue_warn > 1.0:
                        print("Warning: Input queue full, dropping audio block")
                        _last_queue_warn = now
                
                # Get enhanced audio from output queue
                if not output_queue.empty():
                    enhanced_block = output_queue.get_nowait()
                    
                    # Ensure the block size matches
                    if len(enhanced_block) == frames:
                        output_block = enhanced_block
                    else:
                        print(f"Warning: Block size mismatch ({len(enhanced_block)} vs {frames})")
                        output_block = audio_block  # Fallback to original
                else:
                    # No enhanced audio available yet, use original
                    output_block = audio_block
        
        # Apply dry/wet mix if not 100% wet
        if dry_wet_mix < 1.0:
            output_block = _mix_audio(original_block, output_block, dry_wet_mix)
        
        # Apply output gain
        if output_gain != 1.0:
            output_block = _apply_gain(output_block, output_gain)
        
        # Calculate and store output level
        output_level = _calculate_db_level(output_block)
        audio_levels["output_level"] = output_level
        
        # Send to output
        outdata[:, 0] = output_block
        
        # Record if enabled
        if enable_recording:
            with recording_lock:
                recording_buffer.append(output_block.copy())
            
    except Exception as e:
        print(f"Error in audio callback: {e}")
        outdata.fill(0)
        with stats_lock:
            stats["errors"] += 1


async def websocket_handler():
    """Manage WebSocket connection with reconnection and split send/recv tasks."""
    global running, stats
    backoff = 1.0
    seq = 0
    MAGIC = b"THv1"
    while running:
        print(f"Connecting to WebSocket server: {SERVER_URL}")
        try:
            async with websockets.connect(
                SERVER_URL,
                open_timeout=3,
                close_timeout=1,
                ping_interval=20,
                ping_timeout=20,
            ) as websocket:
                print("✓ Connected to enhancement server")
                ws_connected.set()
                backoff = 1.0  # reset on success

                async def sender():
                    loop = asyncio.get_event_loop()
                    while running and ws_connected.is_set():
                        try:
                            audio_block = await loop.run_in_executor(None, input_queue.get, True, 0.1)
                        except Exception:
                            continue
                        try:
                            nonlocal seq
                            audio_bytes = audio_block.astype(np.int16).tobytes()
                            t_send_ms = float(loop.time() * 1000.0)
                            header = MAGIC + struct.pack("<I", seq) + struct.pack("<d", t_send_ms)
                            await websocket.send(header + audio_bytes)
                            seq = (seq + 1) & 0xFFFFFFFF
                            with stats_lock:
                                stats["blocks_sent"] += 1
                        except Exception:
                            with stats_lock:
                                stats["errors"] += 1
                            await asyncio.sleep(0.01)

                async def receiver():
                    loop = asyncio.get_event_loop()
                    lat_samples = []
                    while running and ws_connected.is_set():
                        try:
                            enhanced_bytes = await websocket.recv()
                            if isinstance(enhanced_bytes, (bytes, bytearray)) and len(enhanced_bytes) >= 16 and enhanced_bytes[:4] == MAGIC:
                                r_seq = struct.unpack_from("<I", enhanced_bytes, 4)[0]
                                t_send_ms = struct.unpack_from("<d", enhanced_bytes, 8)[0]
                                pcm_bytes = enhanced_bytes[16:]
                                now_ms = float(loop.time() * 1000.0)
                                latency = max(0.0, now_ms - t_send_ms)
                                lat_samples.append(latency)
                                if len(lat_samples) > 200:
                                    lat_samples.pop(0)
                                with stats_lock:
                                    stats["last_latency_ms"] = latency
                                    stats["avg_latency_ms"] = float(sum(lat_samples) / len(lat_samples))
                            else:
                                pcm_bytes = enhanced_bytes
                            enhanced_block = np.frombuffer(pcm_bytes, dtype=np.int16)
                            if not output_queue.full():
                                output_queue.put(enhanced_block)
                                with stats_lock:
                                    stats["blocks_received"] += 1
                            else:
                                try:
                                    _ = output_queue.get_nowait()
                                    output_queue.put(enhanced_block)
                                except Exception:
                                    pass
                        except websockets.exceptions.ConnectionClosed:
                            print("WebSocket connection closed by server")
                            break
                        except Exception:
                            with stats_lock:
                                stats["errors"] += 1
                            await asyncio.sleep(0.01)

                send_task = asyncio.create_task(sender())
                recv_task = asyncio.create_task(receiver())
                await asyncio.wait({send_task, recv_task}, return_when=asyncio.FIRST_EXCEPTION)
        except Exception as e:
            print(f"Failed to connect to WebSocket server: {e}")
            print("If remote, ensure the server is reachable and use wss:// behind TLS.")
        finally:
            ws_connected.clear()
            # Drain queues to prevent stale buildup
            try:
                while not input_queue.empty():
                    input_queue.get_nowait()
            except Exception:
                pass
            # Exponential backoff up to 10s
            if running:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)


async def stats_reporter():
    """
    Periodically print statistics about the audio processing.
    """
    global running, stats
    
    while running:
        await asyncio.sleep(5.0)
        with stats_lock:
            print(f"\n--- Stats ---")
            print(f"Blocks sent: {stats['blocks_sent']}")
            print(f"Blocks received: {stats['blocks_received']}")
            print(f"Errors: {stats['errors']}")
            print(f"Avg latency: {stats.get('avg_latency_ms', 0.0):.1f} ms")
            print(f"Last latency: {stats.get('last_latency_ms', 0.0):.1f} ms")
            print(f"Input queue: {input_queue.qsize()} | Output queue: {output_queue.qsize()}")
            print(f"Input level: {audio_levels.get('input_level', -100):.1f} dB")
            print(f"Output level: {audio_levels.get('output_level', -100):.1f} dB")
            with processing_lock:
                print(f"Input gain: {processing_params['input_gain']:.2f}x | Output gain: {processing_params['output_gain']:.2f}x")
                print(f"Mix: {processing_params['dry_wet_mix']*100:.0f}% wet | Pass-through: {processing_params['pass_through']}")
            print("-------------\n")


def start_recording():
    """Start recording processed audio."""
    global recording_buffer
    with recording_lock:
        recording_buffer = []
    with processing_lock:
        processing_params["enable_recording"] = True
    print("Recording started...")


def stop_recording() -> np.ndarray:
    """Stop recording and return the recorded audio."""
    with processing_lock:
        processing_params["enable_recording"] = False
    with recording_lock:
        if recording_buffer:
            recorded = np.concatenate(recording_buffer)
            recording_buffer.clear()
            print(f"Recording stopped. Captured {len(recorded)} samples.")
            return recorded
        else:
            print("No audio recorded.")
            return np.array([], dtype=np.int16)


def save_recording(audio: np.ndarray, filename: str, sample_rate: int = SAMPLE_RATE):
    """Save recorded audio to WAV file."""
    try:
        with wave.open(filename, 'wb') as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(sample_rate)
            wav_out.writeframes(audio.tobytes())
        file_size = Path(filename).stat().st_size / 1024
        duration = len(audio) / sample_rate
        print(f"✓ Recording saved: {filename} ({file_size:.1f} KB, {duration:.1f}s)")
        return True
    except Exception as e:
        print(f"✗ Error saving recording: {e}")
        return False


def get_presets_file_path() -> Path:
    """Get the path to the user presets file."""
    # Store in user's home directory
    home = Path.home()
    presets_dir = home / ".tonehoner"
    presets_dir.mkdir(exist_ok=True)
    return presets_dir / "presets.json"


def load_custom_presets() -> dict:
    """Load custom presets from JSON file."""
    presets_file = get_presets_file_path()
    if presets_file.exists():
        try:
            with open(presets_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load presets: {e}")
            return {}
    return {}


def save_custom_presets(presets: dict) -> bool:
    """Save custom presets to JSON file."""
    presets_file = get_presets_file_path()
    try:
        with open(presets_file, 'w') as f:
            json.dump(presets, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving presets: {e}")
        return False


def get_default_presets() -> dict:
    """Get the built-in default presets."""
    return {
        "Default": {
            "input_gain": 1.0,
            "output_gain": 1.0,
            "mix": 1.0,
            "noise_gate": -60.0,
            "pass_through": False
        },
        "Boost": {
            "input_gain": 1.5,
            "output_gain": 1.2,
            "mix": 1.0,
            "noise_gate": -50.0,
            "pass_through": False
        },
        "Subtle": {
            "input_gain": 1.0,
            "output_gain": 1.0,
            "mix": 0.5,
            "noise_gate": -70.0,
            "pass_through": False
        },
        "Aggressive": {
            "input_gain": 1.8,
            "output_gain": 1.5,
            "mix": 1.0,
            "noise_gate": -40.0,
            "pass_through": False
        },
    }


def list_audio_devices():
    """
    List all available audio input and output devices.
    """
    print("\n=== Available Audio Devices ===")
    print(sd.query_devices())
    print("\nUse the device index for INPUT_DEVICE and OUTPUT_DEVICE")
    print("To list devices anytime: python -m sounddevice\n")


async def process_audio_file(input_file: str, output_file: str, server_url: str):
    """
    Process an audio file through the enhancement server and save the result.
    
    Args:
        input_file: Path to input WAV file
        output_file: Path to output WAV file
        server_url: WebSocket server URL
    """
    print(f"\n{'='*60}")
    print(f"Processing Audio File")
    print(f"{'='*60}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Server: {server_url}")
    print(f"{'='*60}\n")
    
    # Read input WAV file
    print("Reading input file...")
    try:
        with wave.open(input_file, 'rb') as wav_in:
            sample_rate = wav_in.getframerate()
            num_channels = wav_in.getnchannels()
            sample_width = wav_in.getsampwidth()
            num_frames = wav_in.getnframes()
            
            # Read all audio data
            audio_bytes = wav_in.readframes(num_frames)
            
            # Convert to numpy array
            if sample_width == 2:  # 16-bit
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            else:
                print(f"Error: Only 16-bit WAV files are supported (got {sample_width*8}-bit)")
                return False
            
            # Convert stereo to mono if needed
            if num_channels == 2:
                print("Converting stereo to mono...")
                audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)
                num_channels = 1
            elif num_channels > 2:
                print(f"Error: Only mono or stereo files are supported (got {num_channels} channels)")
                return False
            
            duration_sec = len(audio_data) / sample_rate
            print(f"✓ Loaded: {num_frames} samples, {sample_rate}Hz, {num_channels} channel(s), {duration_sec:.2f}s")
            
            # Resample to model SR (48k) for best quality
            orig_sr = sample_rate
            if orig_sr != SAMPLE_RATE:
                print(f"Resampling from {orig_sr} Hz to {SAMPLE_RATE} Hz for processing...")
                audio_data = _resample_audio(audio_data, orig_sr, SAMPLE_RATE)
                sample_rate = SAMPLE_RATE
                print(f"✓ Resampled length: {len(audio_data)} samples")

            # Store reference RMS at processing SR for gentle gain matching later
            input_rms_ref = float(np.sqrt(np.mean(audio_data.astype(np.float32)**2) + 1e-12))
    
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error reading input file: {e}")
        return False
    
    # Connect to server and process
    print(f"\nConnecting to server: {server_url}")
    enhanced_blocks = []
    
    try:
        async with websockets.connect(server_url) as websocket:
            print("✓ Connected to server")
            
            # For file processing, use larger blocks with overlap-add crossfade
            proc_block = SAMPLE_RATE  # 1 second blocks at 48k
            xfade = 1024             # samples overlap for crossfade
            if xfade >= proc_block:
                xfade = proc_block // 4

            num_blocks = (len(audio_data) + proc_block - 1) // proc_block
            print(f"\nProcessing {num_blocks} blocks (block={proc_block}, xfade={xfade})...")

            prev_tail = None
            win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(xfade) / (xfade - 1))  # Hann window

            out_accum = []
            for i in range(num_blocks):
                start_idx = i * proc_block
                end_idx = min(start_idx + proc_block, len(audio_data))

                # Extract and pad to full block
                audio_block = audio_data[start_idx:end_idx]
                if len(audio_block) < proc_block:
                    audio_block = np.pad(audio_block, (0, proc_block - len(audio_block)), mode='constant')

                # Send block
                await websocket.send(audio_block.astype(np.int16).tobytes())

                # Receive enhanced block
                enhanced_bytes = await websocket.recv()
                enhanced_block = np.frombuffer(enhanced_bytes, dtype=np.int16)

                # Crossfade with previous tail if available
                if prev_tail is None:
                    out_accum.append(enhanced_block[:-xfade])
                    prev_tail = enhanced_block[-xfade:]
                else:
                    head = enhanced_block[:xfade]
                    tail = enhanced_block[xfade:-xfade]
                    # Apply symmetric crossfade
                    mixed = (prev_tail * (1.0 - win) + head * win).astype(np.int16)
                    out_accum.append(mixed)
                    out_accum.append(tail)
                    prev_tail = enhanced_block[-xfade:]

                # Progress indicator
                if (i + 1) % 5 == 0 or (i + 1) == num_blocks:
                    progress = (i + 1) / num_blocks * 100
                    print(f"  Progress: {i + 1}/{num_blocks} blocks ({progress:.1f}%)")

            # Append last tail
            if prev_tail is not None:
                out_accum.append(prev_tail)

            # Concatenate all parts
            enhanced_blocks = out_accum
            print(f"✓ Processed all {num_blocks} blocks")
    
    except Exception as e:
        print(f"✗ Server error: {e}")
        return False
    
    # Concatenate all enhanced blocks
    print("\nAssembling output...")
    enhanced_audio = np.concatenate(enhanced_blocks)
    
    # Trim to processed-length before any resample back
    enhanced_audio = enhanced_audio[:len(audio_data)]

    # Optional: match output RMS to input RMS to avoid loudness drop (do at processing SR)
    def _rms(x):
        x = x.astype(np.float32)
        return np.sqrt(np.mean(x * x) + 1e-12)

    try:
        in_rms = input_rms_ref if 'input_rms_ref' in locals() else _rms(enhanced_audio)
        out_rms = _rms(enhanced_audio)
        if out_rms > 0 and in_rms > 0:
            gain = np.clip(in_rms / out_rms, 0.5, 2.0)
            enhanced_audio = np.clip(enhanced_audio.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
            print(f"Applied output gain: {gain:.2f}x to match input RMS")
    except Exception:
        pass

    # If we resampled to 48k, convert back to original SR
    try:
        if 'orig_sr' in locals() and orig_sr != sample_rate:
            print(f"Resampling enhanced audio back to {orig_sr} Hz...")
            enhanced_audio = _resample_audio(enhanced_audio, sample_rate, orig_sr)
            out_sr = orig_sr
        else:
            out_sr = sample_rate
    except Exception as e:
        print(f"Warning: Resample back failed ({e}), keeping {sample_rate} Hz")
        out_sr = sample_rate

    # Write output WAV file
    print(f"Writing output file: {output_file}")
    try:
        with wave.open(output_file, 'wb') as wav_out:
            wav_out.setnchannels(1)  # Mono
            wav_out.setsampwidth(2)  # 16-bit
            wav_out.setframerate(out_sr)
            wav_out.writeframes(enhanced_audio.tobytes())
        
        output_size = Path(output_file).stat().st_size / 1024 / 1024
        print(f"✓ Output saved: {output_file} ({output_size:.2f} MB)")
    
    except Exception as e:
        print(f"✗ Error writing output file: {e}")
        return False
    
    print(f"\n{'='*60}")
    print("✓✓✓ Processing Complete! ✓✓✓")
    print(f"{'='*60}\n")
    
    return True


async def main_async():
    """
    Main async function that starts the WebSocket handler and stats reporter.
    """
    global running
    
    # Start WebSocket handler and stats reporter
    websocket_task = asyncio.create_task(websocket_handler())
    stats_task = asyncio.create_task(stats_reporter())
    
    # Wait for WebSocket to finish
    await websocket_task
    stats_task.cancel()


def main():
    """
    Main function that sets up the audio stream and runs the async event loop.
    """
    global running, SERVER_URL, INPUT_DEVICE, OUTPUT_DEVICE
    
    parser = argparse.ArgumentParser(
        description="Real-time audio enhancement client for DeepFilterNet"
    )
    parser.add_argument(
        "--list-devices", 
        action="store_true",
        help="List all available audio devices and exit"
    )
    parser.add_argument(
        "--server",
        default=SERVER_URL,
        help=f"WebSocket server URL (default: {SERVER_URL})"
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=INPUT_DEVICE,
        help="Input device index (default: system default)"
    )
    parser.add_argument(
        "--output-device",
        type=int,
        default=OUTPUT_DEVICE,
        help="Output device index (default: system default)"
    )
    # Default to GUI when running as a frozen (packaged) executable
    default_gui = bool(getattr(sys, 'frozen', False))
    parser.add_argument(
        "--gui",
        action="store_true",
        default=default_gui,
        help="Launch GUI for device selection and control (default in packaged exe)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Process an audio file instead of real-time streaming (WAV format)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: adds '_enhanced' suffix to input filename)"
    )
    
    args = parser.parse_args()
    
    # List devices and exit if requested
    if args.list_devices:
        list_audio_devices()
        return

    # Launch GUI if requested
    if args.gui:
        return launch_gui()
    
    # Process file if requested
    if args.file:
        input_file = args.file
        
        # Generate output filename if not provided
        if args.output:
            output_file = args.output
        else:
            input_path = Path(input_file)
            output_file = str(input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}")
        
        # Process the file
        success = asyncio.run(process_audio_file(input_file, output_file, args.server))
        sys.exit(0 if success else 1)
    
    # Update global configuration
    SERVER_URL = args.server
    INPUT_DEVICE = args.input_device
    OUTPUT_DEVICE = args.output_device
    
    print("=" * 60)
    print("Real-time Audio Enhancement Client")
    print("=" * 60)
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Block size: {BLOCK_SIZE} samples ({BLOCK_SIZE/SAMPLE_RATE*1000:.1f} ms)")
    print(f"Channels: {CHANNELS} (mono)")
    print(f"Server: {SERVER_URL}")
    print(f"Input device: {INPUT_DEVICE if INPUT_DEVICE is not None else 'default'}")
    print(f"Output device: {OUTPUT_DEVICE if OUTPUT_DEVICE is not None else 'default'}")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        # Open audio stream with both input and output
        with sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            device=(INPUT_DEVICE, OUTPUT_DEVICE),
            channels=CHANNELS,
            dtype=DTYPE,
            callback=audio_callback
        ):
            print("✓ Audio stream started")
            print("✓ Capturing from microphone...")
            
            # Run the async event loop
            asyncio.run(main_async())
    
    except KeyboardInterrupt:
        print("\n\nStopping client...")
        running = False
    except Exception as e:
        print(f"\nError: {e}")
        running = False
    finally:
        print("\n--- Final Stats ---")
        print(f"Blocks sent: {stats['blocks_sent']}")
        print(f"Blocks received: {stats['blocks_received']}")
        print(f"Errors: {stats['errors']}")
        print("\nClient stopped.")


# =====================
# GUI IMPLEMENTATION
# =====================

class ClientController:
    """
    Controls the lifetime of the audio stream and websocket loop
    so the GUI can start/stop the client cleanly.
    """
    def __init__(self):
        self.stream = None
        self.async_thread = None

    def start(self, server_url: str, input_device, output_device):
        global SERVER_URL, INPUT_DEVICE, OUTPUT_DEVICE, running

        # Update globals
        SERVER_URL = server_url
        INPUT_DEVICE = input_device if input_device != "default" else None
        OUTPUT_DEVICE = output_device if output_device != "default" else None

        # Reset running state and stats
        running = True
        with stats_lock:
            stats["blocks_sent"] = 0
            stats["blocks_received"] = 0
            stats["errors"] = 0
        while not input_queue.empty():
            try:
                input_queue.get_nowait()
            except Exception:
                break
        while not output_queue.empty():
            try:
                output_queue.get_nowait()
            except Exception:
                break

        # Start audio stream (non-context so we can stop later)
        self.stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            device=(INPUT_DEVICE, OUTPUT_DEVICE),
            channels=CHANNELS,
            dtype=DTYPE,
            callback=audio_callback
        )
        self.stream.start()

        # Start websocket loop in background thread
        self.async_thread = threading.Thread(target=lambda: asyncio.run(main_async()), daemon=True)
        self.async_thread.start()

    def stop(self):
        global running
        running = False
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        if self.async_thread is not None:
            self.async_thread.join(timeout=2.0)


def _get_device_choices():
    """
    Returns two lists of (label, index) for input and output devices.
    Includes a 'default' option.
    """
    try:
        devices = sd.query_devices()
    except Exception as e:
        return [(f"default", "default")], [(f"default", "default")]

    inputs = [("default", "default")]
    outputs = [("default", "default")]
    for idx, dev in enumerate(devices):
        name = dev.get("name", f"Device {idx}")
        if dev.get("max_input_channels", 0) > 0:
            inputs.append((f"{idx}: {name}", idx))
        if dev.get("max_output_channels", 0) > 0:
            outputs.append((f"{idx}: {name}", idx))
    return inputs, outputs


def launch_gui():
    controller = ClientController()

    root = tk.Tk()
    root.title("ToneHoner Client - Audio Enhancement")
    
    # Get screen dimensions and calculate appropriate window size
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Calculate window size (aim for 70% of screen height, maintain aspect ratio)
    if screen_height >= 900:
        # Large screen (1080p+)
        window_width = 700
        window_height = 650
    elif screen_height >= 768:
        # Medium screen (HD)
        window_width = 650
        window_height = 580
    else:
        # Small screen
        window_width = 600
        window_height = 520
    
    # Center window on screen
    x_position = (screen_width - window_width) // 2
    y_position = (screen_height - window_height) // 2
    
    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
    
    # Lock window size (disable resizing)
    root.resizable(False, False)
    
    # Set minimum size as fallback
    root.minsize(600, 520)

    # Create notebook (tabbed interface)
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # ========================================
    # TAB 1: Real-time Streaming
    # ========================================
    stream_frame = ttk.Frame(notebook, padding=12)
    notebook.add(stream_frame, text="Real-time Streaming")

    # Server URL
    ttk.Label(stream_frame, text="Server URL").grid(row=0, column=0, sticky=tk.W)
    server_var = tk.StringVar(value=SERVER_URL)
    server_entry = ttk.Entry(stream_frame, textvariable=server_var, width=45)
    server_entry.grid(row=0, column=1, columnspan=2, sticky=tk.W)

    # Devices
    in_choices, out_choices = _get_device_choices()

    ttk.Label(stream_frame, text="Input Device").grid(row=1, column=0, sticky=tk.W, pady=(8,0))
    in_var = tk.StringVar(value=in_choices[0][0])
    in_combo = ttk.Combobox(stream_frame, textvariable=in_var, values=[c[0] for c in in_choices], state="readonly", width=42)
    in_combo.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=(8,0))

    ttk.Label(stream_frame, text="Output Device").grid(row=2, column=0, sticky=tk.W, pady=(8,0))
    out_var = tk.StringVar(value=out_choices[0][0])
    out_combo = ttk.Combobox(stream_frame, textvariable=out_var, values=[c[0] for c in out_choices], state="readonly", width=42)
    out_combo.grid(row=2, column=1, columnspan=2, sticky=tk.W, pady=(8,0))

    # Controls
    start_btn = ttk.Button(stream_frame, text="Start")
    stop_btn = ttk.Button(stream_frame, text="Stop", state=tk.DISABLED)
    start_btn.grid(row=3, column=1, sticky=tk.W, pady=(12,4))
    stop_btn.grid(row=3, column=2, sticky=tk.W, pady=(12,4))

    # Stats
    sep = ttk.Separator(stream_frame)
    sep.grid(row=4, column=0, columnspan=3, sticky="ew", pady=8)

    stats_title = ttk.Label(stream_frame, text="Stats", font=("Segoe UI", 10, "bold"))
    stats_title.grid(row=5, column=0, sticky=tk.W)

    blocks_sent_var = tk.StringVar(value="0")
    blocks_recv_var = tk.StringVar(value="0")
    errors_var = tk.StringVar(value="0")
    latency_var = tk.StringVar(value="N/A")
    connection_var = tk.StringVar(value="Disconnected")

    ttk.Label(stream_frame, text="Connection:").grid(row=6, column=0, sticky=tk.W)
    connection_label = ttk.Label(stream_frame, textvariable=connection_var, foreground="red")
    connection_label.grid(row=6, column=1, sticky=tk.W)
    
    ttk.Label(stream_frame, text="Blocks sent:").grid(row=7, column=0, sticky=tk.W)
    ttk.Label(stream_frame, textvariable=blocks_sent_var).grid(row=7, column=1, sticky=tk.W)
    
    ttk.Label(stream_frame, text="Blocks received:").grid(row=8, column=0, sticky=tk.W)
    ttk.Label(stream_frame, textvariable=blocks_recv_var).grid(row=8, column=1, sticky=tk.W)
    
    ttk.Label(stream_frame, text="Avg Latency:").grid(row=9, column=0, sticky=tk.W)
    ttk.Label(stream_frame, textvariable=latency_var).grid(row=9, column=1, sticky=tk.W)
    
    ttk.Label(stream_frame, text="Errors:").grid(row=10, column=0, sticky=tk.W)
    ttk.Label(stream_frame, textvariable=errors_var).grid(row=10, column=1, sticky=tk.W)

    # Helpers to map label text -> index
    in_map = {label: idx for label, idx in in_choices}
    out_map = {label: idx for label, idx in out_choices}

    def on_start():
        try:
            url = server_var.get().strip()
            in_sel = in_map.get(in_var.get(), "default")
            out_sel = out_map.get(out_var.get(), "default")
            controller.start(url, in_sel, out_sel)
            start_btn.config(state=tk.DISABLED)
            stop_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Start Failed", f"Could not start client: {e}")

    def on_stop():
        try:
            controller.stop()
            start_btn.config(state=tk.NORMAL)
            stop_btn.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showwarning("Stop Warning", f"Stop encountered an issue: {e}")

    start_btn.config(command=on_start)
    stop_btn.config(command=on_stop)

    def refresh_stats():
        with stats_lock:
            blocks_sent_var.set(str(stats.get("blocks_sent", 0)))
            blocks_recv_var.set(str(stats.get("blocks_received", 0)))
            errors_var.set(str(stats.get("errors", 0)))
            avg_lat = stats.get("avg_latency_ms", 0.0)
            if avg_lat > 0:
                latency_var.set(f"{avg_lat:.1f} ms")
            else:
                latency_var.set("N/A")
        
        # Update connection status
        if ws_connected.is_set():
            connection_var.set("Connected")
            connection_label.config(foreground="green")
        else:
            connection_var.set("Disconnected")
            connection_label.config(foreground="red")
        
        root.after(1000, refresh_stats)

    # ========================================
    # TAB 2: File Processing
    # ========================================
    file_frame = ttk.Frame(notebook, padding=12)
    notebook.add(file_frame, text="File Processing")

    # Server URL for file processing
    ttk.Label(file_frame, text="Server URL").grid(row=0, column=0, sticky=tk.W)
    file_server_var = tk.StringVar(value=SERVER_URL)
    file_server_entry = ttk.Entry(file_frame, textvariable=file_server_var, width=45)
    file_server_entry.grid(row=0, column=1, columnspan=2, sticky=tk.W+tk.E)

    # Input file selection
    ttk.Label(file_frame, text="Input File").grid(row=1, column=0, sticky=tk.W, pady=(8,0))
    input_file_var = tk.StringVar(value="")
    input_file_entry = ttk.Entry(file_frame, textvariable=input_file_var, width=35)
    input_file_entry.grid(row=1, column=1, sticky=tk.W+tk.E, pady=(8,0))
    
    def browse_input():
        filename = filedialog.askopenfilename(
            title="Select Input Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if filename:
            input_file_var.set(filename)
            # Auto-generate output filename
            if not output_file_var.get():
                input_path = Path(filename)
                output_file_var.set(str(input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}"))
    
    input_browse_btn = ttk.Button(file_frame, text="Browse...", command=browse_input)
    input_browse_btn.grid(row=1, column=2, sticky=tk.W, padx=(4,0), pady=(8,0))

    # Output file selection
    ttk.Label(file_frame, text="Output File").grid(row=2, column=0, sticky=tk.W, pady=(8,0))
    output_file_var = tk.StringVar(value="")
    output_file_entry = ttk.Entry(file_frame, textvariable=output_file_var, width=35)
    output_file_entry.grid(row=2, column=1, sticky=tk.W+tk.E, pady=(8,0))
    
    def browse_output():
        filename = filedialog.asksaveasfilename(
            title="Select Output Audio File",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if filename:
            output_file_var.set(filename)
    
    output_browse_btn = ttk.Button(file_frame, text="Browse...", command=browse_output)
    output_browse_btn.grid(row=2, column=2, sticky=tk.W, padx=(4,0), pady=(8,0))

    # Process button
    process_btn = ttk.Button(file_frame, text="Process File")
    process_btn.grid(row=3, column=1, sticky=tk.W, pady=(12,4))

    # Progress and status
    sep2 = ttk.Separator(file_frame)
    sep2.grid(row=4, column=0, columnspan=3, sticky="ew", pady=8)

    status_label = ttk.Label(file_frame, text="Status", font=("Segoe UI", 10, "bold"))
    status_label.grid(row=5, column=0, sticky=tk.W)

    progress_var = tk.StringVar(value="Ready")
    progress_label = ttk.Label(file_frame, textvariable=progress_var, wraplength=500)
    progress_label.grid(row=6, column=0, columnspan=3, sticky=tk.W)

    # Progress bar
    progress_bar = ttk.Progressbar(file_frame, mode='indeterminate', length=400)
    progress_bar.grid(row=7, column=0, columnspan=3, sticky=tk.W+tk.E, pady=(8,0))

    def process_file():
        input_file = input_file_var.get().strip()
        output_file = output_file_var.get().strip()
        server_url = file_server_var.get().strip()

        if not input_file:
            messagebox.showerror("Error", "Please select an input file")
            return
        
        if not output_file:
            messagebox.showerror("Error", "Please select an output file")
            return

        # Disable button and start progress
        process_btn.config(state=tk.DISABLED)
        progress_var.set(f"Processing: {Path(input_file).name}")
        progress_bar.start(10)

        # Run processing in background thread
        def process_thread():
            try:
                success = asyncio.run(process_audio_file(input_file, output_file, server_url))
                
                # Update UI in main thread
                root.after(0, lambda: progress_bar.stop())
                root.after(0, lambda: process_btn.config(state=tk.NORMAL))
                
                if success:
                    root.after(0, lambda: progress_var.set(f"✓ Complete! Saved to: {Path(output_file).name}"))
                    root.after(0, lambda: messagebox.showinfo("Success", f"File processed successfully!\n\nOutput: {output_file}"))
                else:
                    root.after(0, lambda: progress_var.set("✗ Processing failed. Check console for details."))
                    root.after(0, lambda: messagebox.showerror("Error", "Processing failed. Check console for details."))
            
            except Exception as e:
                root.after(0, lambda: progress_bar.stop())
                root.after(0, lambda: process_btn.config(state=tk.NORMAL))
                root.after(0, lambda: progress_var.set(f"✗ Error: {str(e)}"))
                root.after(0, lambda: messagebox.showerror("Error", f"Processing error: {str(e)}"))

        thread = threading.Thread(target=process_thread, daemon=True)
        thread.start()

    process_btn.config(command=process_file)

    # Configure grid weights for resizing
    file_frame.columnconfigure(1, weight=1)

    # ========================================
    # TAB 3: Advanced Settings
    # ========================================
    settings_frame = ttk.Frame(notebook, padding=12)
    notebook.add(settings_frame, text="Advanced Settings")

    # Audio Processing Controls
    ttk.Label(settings_frame, text="Audio Processing", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0,8))

    # Input Gain
    ttk.Label(settings_frame, text="Input Gain:").grid(row=1, column=0, sticky=tk.W)
    input_gain_var = tk.DoubleVar(value=1.0)
    input_gain_scale = ttk.Scale(settings_frame, from_=0.0, to=2.0, variable=input_gain_var, orient=tk.HORIZONTAL, length=250)
    input_gain_scale.grid(row=1, column=1, sticky=tk.W+tk.E)
    input_gain_label = ttk.Label(settings_frame, text="1.00x")
    input_gain_label.grid(row=1, column=2, sticky=tk.W, padx=(4,0))

    # Output Gain
    ttk.Label(settings_frame, text="Output Gain:").grid(row=2, column=0, sticky=tk.W, pady=(8,0))
    output_gain_var = tk.DoubleVar(value=1.0)
    output_gain_scale = ttk.Scale(settings_frame, from_=0.0, to=2.0, variable=output_gain_var, orient=tk.HORIZONTAL, length=250)
    output_gain_scale.grid(row=2, column=1, sticky=tk.W+tk.E, pady=(8,0))
    output_gain_label = ttk.Label(settings_frame, text="1.00x")
    output_gain_label.grid(row=2, column=2, sticky=tk.W, padx=(4,0), pady=(8,0))

    # Dry/Wet Mix
    ttk.Label(settings_frame, text="Dry/Wet Mix:").grid(row=3, column=0, sticky=tk.W, pady=(8,0))
    mix_var = tk.DoubleVar(value=1.0)
    mix_scale = ttk.Scale(settings_frame, from_=0.0, to=1.0, variable=mix_var, orient=tk.HORIZONTAL, length=250)
    mix_scale.grid(row=3, column=1, sticky=tk.W+tk.E, pady=(8,0))
    mix_label = ttk.Label(settings_frame, text="100% Wet")
    mix_label.grid(row=3, column=2, sticky=tk.W, padx=(4,0), pady=(8,0))

    # Noise Gate
    ttk.Label(settings_frame, text="Noise Gate:").grid(row=4, column=0, sticky=tk.W, pady=(8,0))
    noise_gate_var = tk.DoubleVar(value=-60.0)
    noise_gate_scale = ttk.Scale(settings_frame, from_=-100.0, to=-20.0, variable=noise_gate_var, orient=tk.HORIZONTAL, length=250)
    noise_gate_scale.grid(row=4, column=1, sticky=tk.W+tk.E, pady=(8,0))
    noise_gate_label = ttk.Label(settings_frame, text="-60 dB")
    noise_gate_label.grid(row=4, column=2, sticky=tk.W, padx=(4,0), pady=(8,0))

    # Pass-through mode
    pass_through_var = tk.BooleanVar(value=False)
    pass_through_check = ttk.Checkbutton(settings_frame, text="Pass-through mode (bypass enhancement)", variable=pass_through_var)
    pass_through_check.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(12,4))

    # Apply button
    def apply_settings():
        with processing_lock:
            processing_params["input_gain"] = input_gain_var.get()
            processing_params["output_gain"] = output_gain_var.get()
            processing_params["dry_wet_mix"] = mix_var.get()
            processing_params["noise_gate_threshold"] = noise_gate_var.get()
            processing_params["pass_through"] = pass_through_var.get()
        preset_status_var.set("✓ Audio processing settings updated")
        preset_status_label.config(foreground="green")
        root.after(3000, lambda: preset_status_var.set(""))

    apply_btn = ttk.Button(settings_frame, text="Apply Settings", command=apply_settings)
    apply_btn.grid(row=6, column=1, sticky=tk.W, pady=(8,4))

    # Presets
    sep3 = ttk.Separator(settings_frame)
    sep3.grid(row=7, column=0, columnspan=3, sticky="ew", pady=12)

    ttk.Label(settings_frame, text="Presets", font=("Segoe UI", 10, "bold")).grid(row=8, column=0, columnspan=3, sticky=tk.W, pady=(0,8))

    # Status label for preset operations
    preset_status_var = tk.StringVar(value="")
    preset_status_label = ttk.Label(settings_frame, textvariable=preset_status_var, foreground="blue", font=("Segoe UI", 9))
    preset_status_label.grid(row=8, column=0, columnspan=3, sticky=tk.E, pady=(0,8))

    # Load custom presets
    custom_presets = load_custom_presets()
    all_presets = {**get_default_presets(), **custom_presets}

    def load_preset(preset_name):
        preset = all_presets.get(preset_name)
        if preset:
            input_gain_var.set(preset["input_gain"])
            output_gain_var.set(preset["output_gain"])
            mix_var.set(preset["mix"])
            noise_gate_var.set(preset["noise_gate"])
            pass_through_var.set(preset["pass_through"])
            apply_settings()
            preset_status_var.set(f"✓ Loaded preset: {preset_name}")
            preset_status_label.config(foreground="green")
            root.after(3000, lambda: preset_status_var.set(""))
        else:
            preset_status_var.set(f"✗ Preset '{preset_name}' not found")
            preset_status_label.config(foreground="red")
            root.after(3000, lambda: preset_status_var.set(""))

    def save_current_as_preset():
        """Save current settings as a named preset."""
        # Create dialog for preset name
        dialog = tk.Toplevel(root)
        dialog.title("Save Preset")
        dialog.geometry("350x120")
        dialog.resizable(False, False)
        dialog.transient(root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = root.winfo_x() + (root.winfo_width() - dialog.winfo_width()) // 2
        y = root.winfo_y() + (root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
        ttk.Label(dialog, text="Preset Name:").pack(pady=(10,5))
        name_var = tk.StringVar()
        name_entry = ttk.Entry(dialog, textvariable=name_var, width=30)
        name_entry.pack(pady=5)
        name_entry.focus()
        
        def save_preset():
            preset_name = name_var.get().strip()
            if not preset_name:
                messagebox.showerror("Error", "Please enter a preset name", parent=dialog)
                return
            
            # Check if overwriting default preset
            if preset_name in get_default_presets():
                messagebox.showerror("Error", f"Cannot overwrite built-in preset '{preset_name}'\nPlease choose a different name.", parent=dialog)
                return
            
            # Check if overwriting existing custom preset
            if preset_name in custom_presets:
                if not messagebox.askyesno("Confirm Overwrite", f"Preset '{preset_name}' already exists.\nDo you want to overwrite it?", parent=dialog):
                    return
            
            # Save current settings
            new_preset = {
                "input_gain": input_gain_var.get(),
                "output_gain": output_gain_var.get(),
                "mix": mix_var.get(),
                "noise_gate": noise_gate_var.get(),
                "pass_through": pass_through_var.get()
            }
            
            custom_presets[preset_name] = new_preset
            all_presets[preset_name] = new_preset
            
            if save_custom_presets(custom_presets):
                dialog.destroy()
                update_preset_dropdown()
                preset_status_var.set(f"✓ Preset '{preset_name}' saved successfully")
                preset_status_label.config(foreground="green")
                root.after(3000, lambda: preset_status_var.set(""))
            else:
                dialog.destroy()
                preset_status_var.set("✗ Failed to save preset")
                preset_status_label.config(foreground="red")
                root.after(3000, lambda: preset_status_var.set(""))
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Save", command=save_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Bind Enter key to save
        name_entry.bind('<Return>', lambda e: save_preset())

    def delete_preset():
        """Delete a custom preset."""
        selected = preset_combo.get()
        if not selected:
            messagebox.showwarning("Warning", "Please select a preset to delete")
            return
        
        if selected in get_default_presets():
            messagebox.showerror("Error", "Cannot delete built-in presets")
            return
        
        if selected not in custom_presets:
            messagebox.showerror("Error", "Selected preset not found")
            return
        
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete preset '{selected}'?"):
            del custom_presets[selected]
            del all_presets[selected]
            if save_custom_presets(custom_presets):
                update_preset_dropdown()
                preset_status_var.set(f"✓ Preset '{selected}' deleted")
                preset_status_label.config(foreground="green")
                root.after(3000, lambda: preset_status_var.set(""))
            else:
                preset_status_var.set("✗ Failed to delete preset")
                preset_status_label.config(foreground="red")
                root.after(3000, lambda: preset_status_var.set(""))

    def update_preset_dropdown():
        """Update the preset dropdown with current presets."""
        preset_names = list(all_presets.keys())
        preset_combo['values'] = preset_names
        if preset_names:
            preset_combo.current(0)

    # Preset selection and management
    preset_mgmt_frame = ttk.Frame(settings_frame)
    preset_mgmt_frame.grid(row=9, column=0, columnspan=3, sticky=tk.W+tk.E)
    
    # Dropdown for preset selection
    preset_combo = ttk.Combobox(preset_mgmt_frame, state="readonly", width=25)
    preset_combo['values'] = list(all_presets.keys())
    if preset_combo['values']:
        preset_combo.current(0)
    preset_combo.pack(side=tk.LEFT, padx=(0,4))
    
    ttk.Button(preset_mgmt_frame, text="Load", command=lambda: load_preset(preset_combo.get()), width=8).pack(side=tk.LEFT, padx=(0,4))
    ttk.Button(preset_mgmt_frame, text="Save As...", command=save_current_as_preset, width=10).pack(side=tk.LEFT, padx=(0,4))
    ttk.Button(preset_mgmt_frame, text="Delete", command=delete_preset, width=8).pack(side=tk.LEFT, padx=(0,4))
    
    # Quick access to default presets
    default_preset_frame = ttk.Frame(settings_frame)
    default_preset_frame.grid(row=10, column=0, columnspan=3, sticky=tk.W, pady=(8,0))
    ttk.Label(default_preset_frame, text="Quick Load:", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(0,4))
    ttk.Button(default_preset_frame, text="Default", command=lambda: load_preset("Default"), width=8).pack(side=tk.LEFT, padx=(0,2))
    ttk.Button(default_preset_frame, text="Boost", command=lambda: load_preset("Boost"), width=8).pack(side=tk.LEFT, padx=(0,2))
    ttk.Button(default_preset_frame, text="Subtle", command=lambda: load_preset("Subtle"), width=8).pack(side=tk.LEFT, padx=(0,2))
    ttk.Button(default_preset_frame, text="Aggressive", command=lambda: load_preset("Aggressive"), width=10).pack(side=tk.LEFT, padx=(0,2))

    # Audio Levels Display
    sep4 = ttk.Separator(settings_frame)
    sep4.grid(row=11, column=0, columnspan=3, sticky="ew", pady=12)

    ttk.Label(settings_frame, text="Audio Levels", font=("Segoe UI", 10, "bold")).grid(row=12, column=0, columnspan=3, sticky=tk.W, pady=(0,8))

    input_level_var = tk.StringVar(value="-100 dB")
    output_level_var = tk.StringVar(value="-100 dB")

    ttk.Label(settings_frame, text="Input:").grid(row=13, column=0, sticky=tk.W)
    input_level_label = ttk.Label(settings_frame, textvariable=input_level_var, font=("Consolas", 10))
    input_level_label.grid(row=13, column=1, sticky=tk.W)

    ttk.Label(settings_frame, text="Output:").grid(row=14, column=0, sticky=tk.W)
    output_level_label = ttk.Label(settings_frame, textvariable=output_level_var, font=("Consolas", 10))
    output_level_label.grid(row=14, column=1, sticky=tk.W)

    # Recording Controls
    sep5 = ttk.Separator(settings_frame)
    sep5.grid(row=15, column=0, columnspan=3, sticky="ew", pady=12)

    ttk.Label(settings_frame, text="Recording", font=("Segoe UI", 10, "bold")).grid(row=16, column=0, columnspan=3, sticky=tk.W, pady=(0,8))

    recording_status_var = tk.StringVar(value="Not recording")
    ttk.Label(settings_frame, textvariable=recording_status_var).grid(row=17, column=0, columnspan=2, sticky=tk.W)

    is_recording = tk.BooleanVar(value=False)

    def toggle_recording():
        if is_recording.get():
            # Stop recording
            audio = stop_recording()
            is_recording.set(False)
            recording_status_var.set("Not recording")
            record_btn.config(text="Start Recording")
            
            if len(audio) > 0:
                # Ask where to save
                filename = filedialog.asksaveasfilename(
                    title="Save Recording",
                    defaultextension=".wav",
                    filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
                )
                if filename:
                    save_recording(audio, filename)
                    messagebox.showinfo("Success", f"Recording saved to:\n{filename}")
        else:
            # Start recording
            start_recording()
            is_recording.set(True)
            recording_status_var.set("Recording...")
            record_btn.config(text="Stop Recording")

    record_btn = ttk.Button(settings_frame, text="Start Recording", command=toggle_recording)
    record_btn.grid(row=18, column=0, columnspan=2, sticky=tk.W, pady=(4,0))

    # Update labels for sliders
    def update_slider_labels(*args):
        input_gain_label.config(text=f"{input_gain_var.get():.2f}x")
        output_gain_label.config(text=f"{output_gain_var.get():.2f}x")
        mix_val = mix_var.get()
        if mix_val == 0.0:
            mix_label.config(text="100% Dry")
        elif mix_val == 1.0:
            mix_label.config(text="100% Wet")
        else:
            mix_label.config(text=f"{mix_val*100:.0f}% Wet")
        ng_val = noise_gate_var.get()
        if ng_val <= -100.0:
            noise_gate_label.config(text="OFF")
        else:
            noise_gate_label.config(text=f"{ng_val:.0f} dB")

    input_gain_var.trace_add("write", update_slider_labels)
    output_gain_var.trace_add("write", update_slider_labels)
    mix_var.trace_add("write", update_slider_labels)
    noise_gate_var.trace_add("write", update_slider_labels)

    # Update audio levels periodically
    def update_levels():
        input_level_var.set(f"{audio_levels.get('input_level', -100):.1f} dB")
        output_level_var.set(f"{audio_levels.get('output_level', -100):.1f} dB")
        root.after(100, update_levels)  # Update every 100ms

    update_levels()

    # Configure grid weights
    settings_frame.columnconfigure(1, weight=1)

    # ========================================
    # Main window cleanup
    # ========================================
    def on_close():
        try:
            controller.stop()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh_stats()
    root.mainloop()


if __name__ == "__main__":
    main()
