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

# Audio configuration
SAMPLE_RATE = 48000  # Hz - DeepFilterNet expects 48kHz
BLOCK_SIZE = 4800    # samples - 100ms blocks (48000 * 0.1)
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
input_queue = Queue(maxsize=10)   # Captured audio blocks
output_queue = Queue(maxsize=10)  # Enhanced audio blocks

# Control flags
running = True
stats_lock = threading.Lock()
stats = {
    "blocks_sent": 0,
    "blocks_received": 0,
    "errors": 0
}


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
        # Copy input audio to queue (non-blocking)
        # Convert to 1D array and copy to avoid reference issues
        audio_block = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        
        if not input_queue.full():
            input_queue.put(audio_block)
        else:
            print("Warning: Input queue full, dropping audio block")
        
        # Get enhanced audio from output queue
        if not output_queue.empty():
            enhanced_block = output_queue.get_nowait()
            
            # Ensure the block size matches
            if len(enhanced_block) == frames:
                outdata[:, 0] = enhanced_block
            else:
                print(f"Warning: Block size mismatch ({len(enhanced_block)} vs {frames})")
                outdata.fill(0)
        else:
            # No enhanced audio available yet, output silence
            outdata.fill(0)
            
    except Exception as e:
        print(f"Error in audio callback: {e}")
        outdata.fill(0)
        with stats_lock:
            stats["errors"] += 1


async def websocket_handler():
    """
    Async WebSocket handler that sends captured audio to the server
    and receives enhanced audio back.
    
    Runs in the asyncio event loop.
    """
    global running, stats
    
    print(f"Connecting to WebSocket server: {SERVER_URL}")
    
    try:
        async with websockets.connect(SERVER_URL) as websocket:
            print("✓ Connected to enhancement server")
            
            while running:
                try:
                    # Get audio block from input queue (non-blocking with timeout)
                    try:
                        audio_block = await asyncio.get_event_loop().run_in_executor(
                            None, input_queue.get, True, 0.1
                        )
                    except:
                        # Queue timeout, continue loop
                        continue
                    
                    # Convert numpy array to bytes (int16)
                    audio_bytes = audio_block.astype(np.int16).tobytes()
                    
                    # Send to server
                    await websocket.send(audio_bytes)
                    with stats_lock:
                        stats["blocks_sent"] += 1
                    
                    # Receive enhanced audio from server
                    enhanced_bytes = await websocket.recv()
                    
                    # Convert bytes back to numpy array
                    enhanced_block = np.frombuffer(enhanced_bytes, dtype=np.int16)
                    
                    # Put enhanced audio in output queue
                    if not output_queue.full():
                        output_queue.put(enhanced_block)
                        with stats_lock:
                            stats["blocks_received"] += 1
                    else:
                        print("Warning: Output queue full, dropping enhanced block")
                    
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed by server")
                    running = False
                    break
                except Exception as e:
                    print(f"Error in WebSocket handler: {e}")
                    with stats_lock:
                        stats["errors"] += 1
                    await asyncio.sleep(0.1)
    
    except Exception as e:
        print(f"Failed to connect to WebSocket server: {e}")
        print("Make sure the server is running on localhost:8000")
        running = False


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
            print(f"Input queue size: {input_queue.qsize()}")
            print(f"Output queue size: {output_queue.qsize()}")
            print("-------------\n")


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
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch GUI for device selection and control"
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
    root.title("ToneHoner Client")
    root.geometry("600x450")

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

    ttk.Label(stream_frame, text="Blocks sent:").grid(row=6, column=0, sticky=tk.W)
    ttk.Label(stream_frame, textvariable=blocks_sent_var).grid(row=6, column=1, sticky=tk.W)
    ttk.Label(stream_frame, text="Blocks received:").grid(row=7, column=0, sticky=tk.W)
    ttk.Label(stream_frame, textvariable=blocks_recv_var).grid(row=7, column=1, sticky=tk.W)
    ttk.Label(stream_frame, text="Errors:").grid(row=8, column=0, sticky=tk.W)
    ttk.Label(stream_frame, textvariable=errors_var).grid(row=8, column=1, sticky=tk.W)

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
