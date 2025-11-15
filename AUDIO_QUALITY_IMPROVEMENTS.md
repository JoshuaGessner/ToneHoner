# Audio Quality Improvements

This document describes the improvements made to the ToneHoner audio processing pipeline to enhance noise reduction effectiveness.

## Changes Made

### 1. Fixed CUDA-to-NumPy Conversion Errors ✓

**Problem**: Server crashed with "can't convert cuda:0 device type tensor to numpy" when processing audio on GPU.

**Solution**:
- Added `.detach().cpu()` in `dfn_server.py` before returning enhanced tensor
- Ensured all tensors are properly moved to CPU before NumPy conversion in `main.py`
- Added input audio CPU check before model inference

**Files Modified**:
- `dfn_server.py`: Line 269 - Added `.detach().cpu()` to enhanced tensor
- `main.py`: Line 113 - Added `.detach().cpu().numpy()` conversion

### 2. Improved File Processing Quality ✓

**Problem**: Output files didn't sound significantly different from input.

**Solution**:
- **Resampling**: Automatically resample input audio to 48kHz (model's optimal rate)
- **Overlap-Add Processing**: Use 1-second blocks with 1024-sample Hann window crossfade
- **RMS Gain Matching**: Preserve input loudness by matching RMS levels (clamped ±2x)
- **Fallback Resampling**: Support environments without SciPy via numpy interpolation

**Files Modified**:
- `client/client.py`: Lines 180-350 - Complete rewrite of file processing logic

### 3. Enhanced Audio Normalization ✓

**Problem**: Audio values outside [-1, 1] range could reduce model effectiveness.

**Solution**:
- Added automatic clipping to [-1, 1] range with warning messages
- Ensure proper float32 normalization before model input
- Added validation checks for audio format

**Files Modified**:
- `dfn_server.py`: Lines 248-256 - Added clipping and normalization checks

### 4. Added Test Audio Generator ✓

**New Feature**: Created a script to generate clean and noisy test audio for validation.

**Features**:
- Generates speech-like harmonic signals
- Adds realistic pink noise (not just white noise)
- Configurable noise levels and duration
- Calculates and reports SNR

**New File**:
- `test_noise_reduction.py` - Complete test audio generation tool

## Testing the Improvements

### Step 1: Generate Test Audio

```powershell
# Generate a 3-second test file with moderate noise
C:/Users/jwges/dev/ToneHoner/venv/Scripts/python.exe test_noise_reduction.py --duration 3 --noise-level 0.4
```

This creates:
- `test_clean.wav` - Clean speech-like signal
- `test_noisy.wav` - Signal with pink noise added

### Step 2: Start the Server

```powershell
C:/Users/jwges/dev/ToneHoner/venv/Scripts/python.exe main.py
```

### Step 3: Process the Noisy File

```powershell
C:/Users/jwges/dev/ToneHoner/venv/Scripts/python.exe client/client.py --file test_noisy.wav --output test_enhanced.wav
```

### Step 4: Compare Results

Listen to all three files:
1. `test_clean.wav` - Reference clean signal
2. `test_noisy.wav` - Noisy input
3. `test_enhanced.wav` - DeepFilterNet output

**Expected Results**:
- Background noise should be significantly reduced
- Speech-like signal should remain clear
- Volume should match the input file

## Technical Details

### Audio Processing Pipeline

```
Input WAV (any SR)
    ↓
Resample to 48kHz (if needed)
    ↓
Split into 1s blocks with 1024-sample overlap
    ↓
For each block:
  - Convert int16 → float32 [-1, 1]
  - Send to server via WebSocket
  - Server: DeepFilterNet enhancement
  - Receive enhanced block
    ↓
Overlap-add crossfade (Hann window)
    ↓
RMS gain matching (preserve loudness)
    ↓
Resample back to original SR
    ↓
Save as WAV (original SR)
```

### Model Configuration

- **Model**: DeepFilterNet3 (pre-trained)
- **Sample Rate**: 48kHz
- **Post-filter**: Enabled (for stronger noise reduction)
- **Device**: CUDA (GPU) if available, else CPU

### Performance Notes

- **Real-time Factor (RTF)**: Typically 0.1-0.3x on GPU (10x faster than real-time)
- **Latency**: ~100ms for real-time streaming (4800 samples @ 48kHz)
- **File Processing**: Larger blocks (1s) for better quality, not real-time

## Troubleshooting

### No Noticeable Improvement?

1. **Check input audio quality**: If input has very little noise, output won't differ much
2. **Verify sample rate**: Use `--list-devices` to confirm 48kHz support
3. **Check SNR**: Generate test with `--noise-level 0.5` or higher
4. **Listen carefully**: Use good headphones, noise reduction is often subtle

### Still Getting CUDA Errors?

1. **Restart the server**: The model needs to be reloaded
2. **Check GPU availability**: `nvidia-smi` to verify CUDA is accessible
3. **Try CPU mode**: Uninstall CUDA PyTorch and use CPU version

### Output Too Quiet?

1. **RMS matching**: The gain matching tries to preserve loudness
2. **Check input levels**: Very quiet input = quiet output
3. **Manual boost**: Use audio editor to amplify if needed

## What's Next?

Potential future improvements:

1. **Real-time overlap-add**: Apply crossfading to streaming mode
2. **Configurable attenuation**: Add slider for noise reduction strength
3. **Multi-channel support**: Process stereo files
4. **LUFS normalization**: Perceptual loudness matching instead of RMS
5. **Spectral analysis**: Pre/post comparison graphs

## Files Changed Summary

| File | Changes | Lines Modified |
|------|---------|----------------|
| `dfn_server.py` | Fixed CUDA conversion, added normalization | ~15 lines |
| `main.py` | Added .cpu() to tensor conversion | ~3 lines |
| `client/client.py` | Complete file processing rewrite | ~150 lines |
| `test_noise_reduction.py` | New test audio generator | 160 lines (new) |

## Performance Comparison

### Before Improvements:
- ❌ CUDA errors on GPU systems
- ❌ Minimal noise reduction
- ❌ No resampling support
- ❌ Block artifacts in output

### After Improvements:
- ✅ Stable GPU processing
- ✅ Noticeable noise reduction
- ✅ Automatic resampling to optimal SR
- ✅ Smooth crossfaded output
- ✅ Preserved audio loudness
