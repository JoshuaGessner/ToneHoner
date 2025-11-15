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
from tkinter import ttk, messagebox

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
    global running
    
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
    
    args = parser.parse_args()
    
    # List devices and exit if requested
    if args.list_devices:
        list_audio_devices()
        return

    # Launch GUI if requested
    if args.gui:
        return launch_gui()
    
    # Update global configuration
    global SERVER_URL, INPUT_DEVICE, OUTPUT_DEVICE
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


if __name__ == "__main__":
    main()


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
    root.geometry("520x360")

    main_frame = ttk.Frame(root, padding=12)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Server URL
    ttk.Label(main_frame, text="Server URL").grid(row=0, column=0, sticky=tk.W)
    server_var = tk.StringVar(value=SERVER_URL)
    server_entry = ttk.Entry(main_frame, textvariable=server_var, width=45)
    server_entry.grid(row=0, column=1, columnspan=2, sticky=tk.W)

    # Devices
    in_choices, out_choices = _get_device_choices()

    ttk.Label(main_frame, text="Input Device").grid(row=1, column=0, sticky=tk.W, pady=(8,0))
    in_var = tk.StringVar(value=in_choices[0][0])
    in_combo = ttk.Combobox(main_frame, textvariable=in_var, values=[c[0] for c in in_choices], state="readonly", width=42)
    in_combo.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=(8,0))

    ttk.Label(main_frame, text="Output Device").grid(row=2, column=0, sticky=tk.W, pady=(8,0))
    out_var = tk.StringVar(value=out_choices[0][0])
    out_combo = ttk.Combobox(main_frame, textvariable=out_var, values=[c[0] for c in out_choices], state="readonly", width=42)
    out_combo.grid(row=2, column=1, columnspan=2, sticky=tk.W, pady=(8,0))

    # Controls
    start_btn = ttk.Button(main_frame, text="Start")
    stop_btn = ttk.Button(main_frame, text="Stop", state=tk.DISABLED)
    start_btn.grid(row=3, column=1, sticky=tk.W, pady=(12,4))
    stop_btn.grid(row=3, column=2, sticky=tk.W, pady=(12,4))

    # Stats
    sep = ttk.Separator(main_frame)
    sep.grid(row=4, column=0, columnspan=3, sticky="ew", pady=8)

    stats_title = ttk.Label(main_frame, text="Stats", font=("Segoe UI", 10, "bold"))
    stats_title.grid(row=5, column=0, sticky=tk.W)

    blocks_sent_var = tk.StringVar(value="0")
    blocks_recv_var = tk.StringVar(value="0")
    errors_var = tk.StringVar(value="0")

    ttk.Label(main_frame, text="Blocks sent:").grid(row=6, column=0, sticky=tk.W)
    ttk.Label(main_frame, textvariable=blocks_sent_var).grid(row=6, column=1, sticky=tk.W)
    ttk.Label(main_frame, text="Blocks received:").grid(row=7, column=0, sticky=tk.W)
    ttk.Label(main_frame, textvariable=blocks_recv_var).grid(row=7, column=1, sticky=tk.W)
    ttk.Label(main_frame, text="Errors:").grid(row=8, column=0, sticky=tk.W)
    ttk.Label(main_frame, textvariable=errors_var).grid(row=8, column=1, sticky=tk.W)

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

    def on_close():
        try:
            controller.stop()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh_stats()
    root.mainloop()
