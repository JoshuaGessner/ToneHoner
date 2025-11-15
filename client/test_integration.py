"""
Integration Test for DeepFilterNet Audio Enhancement System

This test script validates the complete audio enhancement pipeline by:
1. Starting the FastAPI server locally
2. Sending a pre-recorded WAV file through the WebSocket endpoint
3. Verifying the enhanced audio output
4. Checking that frames are processed correctly

Requirements:
- Server must have the TorchScript model available at ./models/model_ts.pt
- Test audio file: test_audio.wav (mono, 48kHz, 16-bit PCM)
"""

import asyncio
import subprocess
import time
import wave
import numpy as np
import websockets
import sys
import os
from pathlib import Path
import signal

# Test configuration
SERVER_PORT = 8000
SERVER_URL = f"ws://localhost:{SERVER_PORT}/enhance"
TEST_AUDIO_FILE = "test_audio.wav"
OUTPUT_AUDIO_FILE = "test_output_enhanced.wav"
SERVER_STARTUP_TIMEOUT = 30  # seconds
BLOCK_SIZE = 4800  # 100ms at 48kHz

# Server process handle
server_process = None


def create_test_audio(filename, duration=2.0, sample_rate=48000):
    """
    Create a test audio file with a simple tone.
    
    Args:
        filename: Output WAV filename
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    """
    print(f"Creating test audio file: {filename}")
    
    # Generate a 440 Hz sine wave (A4 note)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    frequency = 440.0
    audio_float = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Convert to int16
    audio_int16 = (audio_float * 32767).astype(np.int16)
    
    # Write WAV file
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    print(f"✓ Created {filename} ({duration}s, {sample_rate}Hz)")


def read_wav_file(filename):
    """
    Read a WAV file and return the audio data as a numpy array.
    
    Args:
        filename: WAV file to read
        
    Returns:
        Tuple of (audio_data, sample_rate, num_channels)
    """
    with wave.open(filename, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        num_frames = wav_file.getnframes()
        sample_width = wav_file.getsampwidth()
        
        # Read audio data
        audio_bytes = wav_file.readframes(num_frames)
        
        # Convert to numpy array
        if sample_width == 2:  # 16-bit
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Handle stereo -> mono conversion
        if num_channels == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)
            num_channels = 1
        
        return audio_data, sample_rate, num_channels


def write_wav_file(filename, audio_data, sample_rate):
    """
    Write audio data to a WAV file.
    
    Args:
        filename: Output WAV filename
        audio_data: Numpy array of int16 audio samples
        sample_rate: Sample rate in Hz
    """
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())


def start_server():
    """
    Start the FastAPI server as a subprocess.
    
    Returns:
        subprocess.Popen object
    """
    global server_process
    
    print("Starting FastAPI server...")
    
    # Find the main.py file (should be in parent directory)
    script_dir = Path(__file__).parent
    server_script = script_dir.parent / "main.py"
    
    if not server_script.exists():
        raise FileNotFoundError(f"Server script not found: {server_script}")
    
    # Start server process
    server_process = subprocess.Popen(
        [sys.executable, str(server_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    print(f"✓ Server process started (PID: {server_process.pid})")
    return server_process


def wait_for_server(timeout=SERVER_STARTUP_TIMEOUT):
    """
    Wait for the server to be ready by polling the /ping endpoint.
    
    Args:
        timeout: Maximum time to wait in seconds
        
    Returns:
        True if server is ready, False otherwise
    """
    import urllib.request
    import urllib.error
    
    print(f"Waiting for server to be ready (timeout: {timeout}s)...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Try to connect to the ping endpoint
            response = urllib.request.urlopen(
                f"http://localhost:{SERVER_PORT}/ping",
                timeout=1
            )
            if response.status == 200:
                print("✓ Server is ready!")
                return True
        except (urllib.error.URLError, ConnectionRefusedError):
            # Server not ready yet
            time.sleep(0.5)
            continue
    
    print("✗ Server failed to start within timeout")
    return False


def stop_server():
    """
    Stop the server process gracefully.
    """
    global server_process
    
    if server_process is None:
        return
    
    print("\nStopping server...")
    
    try:
        # Try graceful shutdown first
        server_process.terminate()
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # Force kill if graceful shutdown fails
        print("Server didn't stop gracefully, forcing...")
        server_process.kill()
        server_process.wait()
    
    print(f"✓ Server stopped (PID: {server_process.pid})")
    server_process = None


async def send_audio_through_websocket(audio_data, block_size=BLOCK_SIZE):
    """
    Send audio data through the WebSocket and receive enhanced audio.
    
    Args:
        audio_data: Numpy array of int16 audio samples
        block_size: Number of samples per block
        
    Returns:
        Numpy array of enhanced audio samples
    """
    print(f"\nConnecting to WebSocket: {SERVER_URL}")
    
    enhanced_blocks = []
    frames_sent = 0
    frames_received = 0
    
    try:
        async with websockets.connect(SERVER_URL) as websocket:
            print("✓ WebSocket connected")
            
            # Split audio into blocks and send
            num_blocks = (len(audio_data) + block_size - 1) // block_size
            
            for i in range(num_blocks):
                start_idx = i * block_size
                end_idx = min(start_idx + block_size, len(audio_data))
                
                # Get audio block
                audio_block = audio_data[start_idx:end_idx]
                
                # Pad if necessary (last block might be shorter)
                if len(audio_block) < block_size:
                    audio_block = np.pad(
                        audio_block,
                        (0, block_size - len(audio_block)),
                        mode='constant'
                    )
                
                # Send block
                audio_bytes = audio_block.astype(np.int16).tobytes()
                await websocket.send(audio_bytes)
                frames_sent += 1
                
                # Receive enhanced block
                enhanced_bytes = await websocket.recv()
                enhanced_block = np.frombuffer(enhanced_bytes, dtype=np.int16)
                enhanced_blocks.append(enhanced_block)
                frames_received += 1
                
                # Progress indicator
                if (i + 1) % 10 == 0 or (i + 1) == num_blocks:
                    print(f"  Progress: {i + 1}/{num_blocks} blocks processed")
            
            print(f"✓ Sent {frames_sent} frames, received {frames_received} frames")
    
    except Exception as e:
        print(f"✗ WebSocket error: {e}")
        raise
    
    # Concatenate all enhanced blocks
    enhanced_audio = np.concatenate(enhanced_blocks)
    
    # Trim to original length (remove padding)
    enhanced_audio = enhanced_audio[:len(audio_data)]
    
    return enhanced_audio


def verify_output(original_audio, enhanced_audio):
    """
    Verify that the enhanced audio is valid.
    
    Args:
        original_audio: Original audio data
        enhanced_audio: Enhanced audio data
        
    Returns:
        True if tests pass, False otherwise
    """
    print("\n=== Verification ===")
    
    all_passed = True
    
    # Test 1: Length match
    print(f"Test 1: Length match")
    print(f"  Original length: {len(original_audio)} samples")
    print(f"  Enhanced length: {len(enhanced_audio)} samples")
    
    if len(original_audio) == len(enhanced_audio):
        print("  ✓ PASS: Lengths match")
    else:
        print("  ✗ FAIL: Length mismatch")
        all_passed = False
    
    # Test 2: Non-zero output
    print(f"\nTest 2: Non-zero output")
    non_zero_count = np.count_nonzero(enhanced_audio)
    print(f"  Non-zero samples: {non_zero_count}/{len(enhanced_audio)}")
    
    if non_zero_count > 0:
        print("  ✓ PASS: Output contains non-zero samples")
    else:
        print("  ✗ FAIL: Output is all zeros")
        all_passed = False
    
    # Test 3: Value range
    print(f"\nTest 3: Value range")
    min_val = np.min(enhanced_audio)
    max_val = np.max(enhanced_audio)
    print(f"  Range: [{min_val}, {max_val}]")
    
    if min_val >= -32768 and max_val <= 32767:
        print("  ✓ PASS: Values within int16 range")
    else:
        print("  ✗ FAIL: Values outside int16 range")
        all_passed = False
    
    # Test 4: Not identical to input
    print(f"\nTest 4: Audio was processed")
    if not np.array_equal(original_audio, enhanced_audio):
        print("  ✓ PASS: Output differs from input (audio was processed)")
    else:
        print("  ⚠ WARNING: Output identical to input (possible passthrough)")
    
    # Test 5: Signal statistics
    print(f"\nTest 5: Signal statistics")
    orig_rms = np.sqrt(np.mean(original_audio.astype(np.float32) ** 2))
    enh_rms = np.sqrt(np.mean(enhanced_audio.astype(np.float32) ** 2))
    print(f"  Original RMS: {orig_rms:.2f}")
    print(f"  Enhanced RMS: {enh_rms:.2f}")
    print(f"  RMS ratio: {enh_rms/orig_rms:.3f}")
    
    return all_passed


async def run_integration_test():
    """
    Main integration test function.
    """
    print("=" * 60)
    print("DeepFilterNet Integration Test")
    print("=" * 60)
    
    test_passed = False
    
    try:
        # Step 1: Create test audio if it doesn't exist
        if not os.path.exists(TEST_AUDIO_FILE):
            create_test_audio(TEST_AUDIO_FILE, duration=2.0)
        else:
            print(f"Using existing test audio: {TEST_AUDIO_FILE}")
        
        # Step 2: Read test audio
        print(f"\nReading test audio file...")
        original_audio, sample_rate, num_channels = read_wav_file(TEST_AUDIO_FILE)
        print(f"✓ Loaded audio: {len(original_audio)} samples, {sample_rate}Hz, {num_channels} channel(s)")
        
        # Verify sample rate
        if sample_rate != 48000:
            print(f"⚠ WARNING: Sample rate is {sample_rate}Hz, expected 48000Hz")
        
        # Step 3: Start server
        start_server()
        
        # Step 4: Wait for server to be ready
        if not wait_for_server():
            raise RuntimeError("Server failed to start")
        
        # Step 5: Send audio through WebSocket
        print("\n=== Processing Audio ===")
        enhanced_audio = await send_audio_through_websocket(original_audio)
        
        # Step 6: Save enhanced audio
        print(f"\nSaving enhanced audio to: {OUTPUT_AUDIO_FILE}")
        write_wav_file(OUTPUT_AUDIO_FILE, enhanced_audio, sample_rate)
        print(f"✓ Enhanced audio saved")
        
        # Step 7: Verify output
        test_passed = verify_output(original_audio, enhanced_audio)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        test_passed = False
    
    finally:
        # Clean up: stop server
        stop_server()
    
    # Print final result
    print("\n" + "=" * 60)
    if test_passed:
        print("✓✓✓ INTEGRATION TEST PASSED ✓✓✓")
    else:
        print("✗✗✗ INTEGRATION TEST FAILED ✗✗✗")
    print("=" * 60)
    
    return test_passed


def main():
    """
    Main entry point for the test script.
    """
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\nInterrupted by user")
        stop_server()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run the async test
    try:
        test_passed = asyncio.run(run_integration_test())
        sys.exit(0 if test_passed else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
        stop_server()
        sys.exit(1)


if __name__ == "__main__":
    main()
