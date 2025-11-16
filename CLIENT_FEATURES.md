# ToneHoner Client - Enhanced Features

## Overview
The ToneHoner client now includes comprehensive audio processing controls, custom preset management, and intelligent window sizing.

## New Features

### 1. Custom Preset Management

**Save Custom Presets:**
- Adjust audio processing settings (gain, mix, noise gate)
- Click "Save As..." button
- Enter a unique preset name
- Preset is saved to `~/.tonehoner/presets.json`

**Load Presets:**
- Select a preset from the dropdown
- Click "Load" to apply settings
- Quick access buttons for built-in presets (Default, Boost, Subtle, Aggressive)

**Delete Presets:**
- Select a custom preset from dropdown
- Click "Delete" to remove it
- Built-in presets cannot be deleted

**Preset Storage:**
- Custom presets are saved in: `%USERPROFILE%\.tonehoner\presets.json` (Windows)
- Presets persist between sessions
- Can be backed up by copying the JSON file

### 2. Audio Processing Controls

Located in the "Advanced Settings" tab:

**Input Gain** (0.0 - 2.0x)
- Pre-processing amplification
- Useful for quiet microphones
- Applied before enhancement

**Output Gain** (0.0 - 2.0x)
- Post-processing volume adjustment
- Applied after enhancement
- Final volume control

**Dry/Wet Mix** (0% - 100%)
- 0% = 100% original (dry) audio
- 100% = 100% enhanced (wet) audio
- Allows subtle enhancement by blending

**Pass-through Mode**
- Checkbox to bypass enhancement completely
- Audio still passes through with gain/effects
- Useful for A/B testing

**Noise Gate** (-100 to -20 dB)
- Silence audio below threshold
- Set to -100 dB or below to disable
- Helps eliminate background noise

### 3. Real-time Audio Monitoring

**Audio Levels:**
- Input level meter (dB)
- Output level meter (dB)
- Updates every 100ms
- Located in Advanced Settings tab

**Connection Status:**
- Visual indicator (green = connected, red = disconnected)
- Located in Real-time Streaming tab
- Shows WebSocket connection state

**Latency Monitoring:**
- Average round-trip latency display
- Measured in milliseconds
- Helps diagnose performance issues

### 4. Recording Capability

**Start/Stop Recording:**
- Button in Advanced Settings tab
- Records processed audio in real-time
- Status indicator shows recording state

**Save Recording:**
- Automatic save dialog when stopping
- Saves as 16-bit WAV file
- Includes duration and file size info

### 5. Intelligent Window Management

**Adaptive Resolution:**
- Automatically detects screen size
- Adjusts window size accordingly:
  - Large screens (1080p+): 700x650
  - Medium screens (HD): 650x580
  - Small screens: 600x520
- Centers window on screen

**Locked Window Size:**
- Window size is fixed (non-resizable)
- Prevents layout issues
- Minimum size: 600x520

## Built-in Presets

### Default
- Input Gain: 1.0x
- Output Gain: 1.0x
- Mix: 100% wet
- Noise Gate: -60 dB
- Standard 1:1 processing

### Boost
- Input Gain: 1.5x
- Output Gain: 1.2x
- Mix: 100% wet
- Noise Gate: -50 dB
- For quiet environments or soft speakers

### Subtle
- Input Gain: 1.0x
- Output Gain: 1.0x
- Mix: 50% wet
- Noise Gate: -70 dB
- Gentle enhancement, natural sound

### Aggressive
- Input Gain: 1.8x
- Output Gain: 1.5x
- Mix: 100% wet
- Noise Gate: -40 dB
- Maximum enhancement, tight noise gate

## Usage Tips

1. **Creating a Custom Preset:**
   - Go to Advanced Settings tab
   - Adjust all controls to desired values
   - Click "Apply Settings" to test
   - When satisfied, click "Save As..."
   - Give it a descriptive name (e.g., "Office Meeting", "Podcast")

2. **Monitoring Audio Levels:**
   - Watch the dB meters to ensure audio isn't clipping
   - Input should typically be between -20 to -6 dB
   - Adjust input gain if levels are too low/high

3. **Optimizing for Your Environment:**
   - Noisy environment: Use higher noise gate (-40 to -30 dB)
   - Quiet environment: Use lower noise gate (-70 to -60 dB)
   - Multiple speakers: Increase input gain, lower noise gate

4. **Testing Settings:**
   - Enable Pass-through mode to hear original audio
   - Disable Pass-through to hear enhanced audio
   - Compare to find optimal settings

5. **Backing Up Presets:**
   - Locate `%USERPROFILE%\.tonehoner\presets.json`
   - Copy file to backup location
   - Restore by copying back to original location

## Keyboard Shortcuts

Currently, the client uses mouse/click interaction. Future versions may include:
- Ctrl+S: Quick save recording
- Ctrl+P: Toggle pass-through mode
- Ctrl+R: Start/stop recording
- Space: Quick mute (when window has focus)

## Troubleshooting

**Presets not saving:**
- Check write permissions in `%USERPROFILE%\.tonehoner\`
- Ensure disk space is available
- Try running as administrator (Windows)

**Window too large/small:**
- Window auto-sizes based on screen resolution
- Minimum size is 600x520 pixels
- If screen is smaller, some content may be cut off

**Audio processing not applying:**
- Click "Apply Settings" after adjusting controls
- Check that WebSocket is connected (green indicator)
- Verify server is running

**Recording issues:**
- Ensure adequate disk space
- Check file permissions in save location
- Recording captures output after all processing

## Technical Details

**Preset File Format:**
```json
{
  "My Custom Preset": {
    "input_gain": 1.2,
    "output_gain": 1.1,
    "mix": 0.8,
    "noise_gate": -55.0,
    "pass_through": false
  }
}
```

**Audio Processing Order:**
1. Input from microphone
2. Input gain applied
3. Noise gate applied (if enabled)
4. Enhancement (if not pass-through)
5. Dry/wet mixing
6. Output gain applied
7. Output to virtual device
8. Recording (if enabled)

**Performance:**
- Audio callback runs in separate thread
- All controls use thread-safe locks
- Level metering updates at 10 Hz (100ms)
- Minimal CPU overhead for controls
