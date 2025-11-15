"""
Generate a test audio file with noise and speech-like signals to verify
DeepFilterNet noise reduction effectiveness.
"""

import numpy as np
import wave
import argparse


def generate_speech_like_tone(duration_sec, sample_rate=48000):
    """
    Generate a speech-like signal (multiple harmonics).
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    
    # Fundamental frequency around 200 Hz (male voice range)
    f0 = 200
    signal = 0.0
    
    # Add harmonics to simulate voice
    harmonics = [1, 2, 3, 4, 5, 6]
    amplitudes = [1.0, 0.5, 0.3, 0.2, 0.15, 0.1]
    
    for h, amp in zip(harmonics, amplitudes):
        signal += amp * np.sin(2 * np.pi * f0 * h * t)
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    # Add amplitude envelope (simulate speech patterns)
    envelope = np.zeros_like(t)
    segment_len = int(0.3 * sample_rate)  # 300ms segments
    
    for i in range(0, len(t), segment_len):
        end = min(i + segment_len, len(t))
        # Random amplitude for each segment
        amp = np.random.uniform(0.3, 1.0)
        envelope[i:end] = amp
    
    signal = signal * envelope
    
    return signal * 0.7  # Scale down to leave headroom


def generate_white_noise(duration_sec, sample_rate=48000, amplitude=0.3):
    """
    Generate white noise.
    """
    num_samples = int(sample_rate * duration_sec)
    noise = np.random.normal(0, amplitude, num_samples)
    return noise


def generate_pink_noise(duration_sec, sample_rate=48000, amplitude=0.2):
    """
    Generate pink noise (more realistic background noise).
    """
    num_samples = int(sample_rate * duration_sec)
    
    # Generate white noise
    white = np.random.randn(num_samples)
    
    # Apply pink noise filter (1/f spectrum)
    # Simple approximation using cascaded filters
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
    
    # Use simple filtering
    pink = white.copy()
    for i in range(3, len(pink)):
        pink[i] = (b[0] * white[i] + b[1] * white[i-1] + b[2] * white[i-2] + b[3] * white[i-3] -
                   a[1] * pink[i-1] - a[2] * pink[i-2] - a[3] * pink[i-3])
    
    # Normalize and scale
    pink = pink / np.max(np.abs(pink)) * amplitude
    
    return pink


def save_wav(filename, audio, sample_rate=48000):
    """
    Save audio to WAV file.
    """
    # Convert to int16
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    print(f"✓ Saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Generate noisy test audio for DeepFilterNet")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    parser.add_argument("--noise-level", type=float, default=0.3, help="Noise amplitude (0-1)")
    parser.add_argument("--output-clean", type=str, default="test_clean.wav", help="Clean audio output")
    parser.add_argument("--output-noisy", type=str, default="test_noisy.wav", help="Noisy audio output")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Sample rate in Hz")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Generating Test Audio")
    print("=" * 60)
    print(f"Duration: {args.duration}s")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Noise level: {args.noise_level}")
    print()
    
    # Generate clean speech-like signal
    print("Generating speech-like signal...")
    clean_signal = generate_speech_like_tone(args.duration, args.sample_rate)
    
    # Generate pink noise (more realistic than white noise)
    print("Generating pink noise...")
    noise = generate_pink_noise(args.duration, args.sample_rate, args.noise_level)
    
    # Mix clean signal with noise
    noisy_signal = clean_signal + noise
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(noisy_signal))
    if max_val > 0.95:
        scale = 0.95 / max_val
        clean_signal *= scale
        noisy_signal *= scale
        print(f"Scaled audio by {scale:.3f} to prevent clipping")
    
    # Calculate SNR
    clean_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr_db = 10 * np.log10(clean_power / (noise_power + 1e-10))
    print(f"Signal-to-Noise Ratio: {snr_db:.1f} dB")
    print()
    
    # Save files
    save_wav(args.output_clean, clean_signal, args.sample_rate)
    save_wav(args.output_noisy, noisy_signal, args.sample_rate)
    
    print()
    print("=" * 60)
    print("✓ Test audio generated successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"1. Start server: python main.py")
    print(f"2. Process noisy file:")
    print(f"   python client/client.py --file {args.output_noisy} --output test_enhanced.wav")
    print(f"3. Compare: {args.output_clean}, {args.output_noisy}, test_enhanced.wav")
    print()


if __name__ == "__main__":
    main()
