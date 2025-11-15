# Virtual Audio Device Setup Guide

This guide explains how to set up virtual audio devices on different operating systems for use with the DeepFilterNet audio enhancement client.

## Overview

A virtual audio device allows you to route the enhanced audio output as an input to other applications (e.g., Discord, Zoom, OBS). The client captures from your physical microphone, enhances the audio via the server, and outputs to the virtual device, which other apps can use as their microphone input.

---

## Windows

### Option 1: VB-Cable (Recommended)

**Manual Installation:**
1. Download VB-Cable from: https://vb-audio.com/Cable/
2. Extract the ZIP file
3. Right-click `VBCABLE_Setup_x64.exe` (or x86) and select "Run as Administrator"
4. Follow the installation wizard
5. Restart your computer

**Chocolatey Installation:**
```powershell
# Install Chocolatey first if not already installed:
# Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install VB-Cable
choco install vb-cable -y

# Restart required
Restart-Computer
```

**Verification:**
After installation, you should see "CABLE Input" and "CABLE Output" in your Windows sound settings.

### Option 2: VoiceMeeter (More Features)

VoiceMeeter provides more advanced audio routing capabilities:

```powershell
# Via Chocolatey
choco install voicemeeter -y
```

Or download from: https://vb-audio.com/Voicemeeter/

---

## macOS

### BlackHole (Recommended)

**Homebrew Installation:**
```bash
# Install Homebrew if not already installed:
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install BlackHole (2-channel version)
brew install blackhole-2ch

# For 16-channel version (multi-channel audio)
# brew install blackhole-16ch
```

**Manual Installation:**
1. Download from: https://github.com/ExistentialAudio/BlackHole
2. Open the `.pkg` file
3. Follow the installation wizard
4. Restart your computer (recommended)

**Verification:**
Open "Audio MIDI Setup" (in Applications/Utilities) and you should see "BlackHole 2ch" as an audio device.

**Creating a Multi-Output Device (Optional):**
If you want to hear the audio while routing it:
1. Open "Audio MIDI Setup"
2. Click the "+" button at bottom-left
3. Select "Create Multi-Output Device"
4. Check both your headphones/speakers and BlackHole
5. Set this as your output device in the client

---

## Linux

### ALSA Loopback Module

**Load the Module (Temporary):**
```bash
# Load the snd-aloop module
sudo modprobe snd-aloop

# Verify it's loaded
aplay -l | grep Loopback
```

**Make it Permanent:**
```bash
# Add to /etc/modules-load.d/
echo "snd-aloop" | sudo tee /etc/modules-load.d/snd-aloop.conf

# Or add to /etc/modules
echo "snd-aloop" | sudo tee -a /etc/modules

# Reload modules
sudo depmod -a
```

**PulseAudio Configuration:**
```bash
# Load PulseAudio loopback module
pactl load-module module-loopback latency_msec=1

# To make it persistent, add to /etc/pulse/default.pa:
echo "load-module module-loopback latency_msec=1" | sudo tee -a /etc/pulse/default.pa
```

**PipeWire Configuration (Modern Linux):**
```bash
# PipeWire usually automatically detects ALSA loopback devices
# Check available devices
pw-cli list-objects | grep node.name
```

### Alternative: Creating a Virtual Sink

```bash
# PulseAudio virtual sink
pactl load-module module-null-sink sink_name=virtual_mic sink_properties=device.description="Virtual_Microphone"

# Make the sink available as a source
pactl load-module module-remap-source source_name=virtual_mic_source master=virtual_mic.monitor source_properties=device.description="Virtual_Microphone_Source"
```

---

## Finding Device Indices

### Using Python sounddevice

Run this command to list all audio devices:

```bash
python -m sounddevice
```

Or use the client's built-in command:

```bash
python client/client.py --list-devices
```

**Example Output:**
```
  0 Microsoft Sound Mapper - Input, MME (2 in, 0 out)
  1 Microphone (Realtek High Defini, MME (2 in, 0 out)
> 2 Microsoft Sound Mapper - Output, MME (0 in, 2 out)
  3 Speakers (Realtek High Definiti, MME (0 in, 2 out)
  4 CABLE Input (VB-Audio Virtual , MME (0 in, 2 out)  <-- Virtual device output
  5 CABLE Output (VB-Audio Virtual, MME (2 in, 0 out)  <-- Virtual device input
```

### Identifying Your Devices

1. **Physical Microphone (INPUT_DEVICE):** Look for your actual microphone device (e.g., "Microphone (Realtek...)")
2. **Virtual Audio Device (OUTPUT_DEVICE):** 
   - **Windows:** "CABLE Input" (VB-Cable) or "VoiceMeeter Input"
   - **macOS:** "BlackHole 2ch" or "BlackHole 16ch"
   - **Linux:** "Loopback" or "virtual_mic"

### Using the Indices in client.py

Once you've identified the device indices, you can:

**Option 1: Command-line arguments**
```bash
python client/client.py --input-device 1 --output-device 4
```

**Option 2: Edit the script constants**
Open `client/client.py` and modify:
```python
INPUT_DEVICE = 1   # Your physical microphone index
OUTPUT_DEVICE = 4  # Your virtual audio device index
```

---

## Testing the Setup

### 1. Test Virtual Device Installation

**Windows:**
- Open Sound Settings → Input devices
- You should see "CABLE Output" as an available microphone

**macOS:**
- Open System Preferences → Sound → Input
- You should see "BlackHole 2ch" as an available input

**Linux:**
```bash
arecord -l  # List capture devices
aplay -l    # List playback devices
```

### 2. Test the Client

```bash
# Start the server first
python main.py

# In another terminal, start the client
python client/client.py --input-device 1 --output-device 4

# Speak into your microphone and verify:
# - Console shows blocks being sent/received
# - Other apps can use the virtual device as input
```

### 3. Test in Other Applications

1. Open Discord, Zoom, OBS, etc.
2. Go to audio settings
3. Select the virtual device as your microphone:
   - **Windows:** "CABLE Output"
   - **macOS:** "BlackHole 2ch"
   - **Linux:** "Loopback" or "virtual_mic_source"
4. Speak into your physical microphone
5. The enhanced audio should appear in the application

---

## Troubleshooting

### No Audio Output
- Verify the server is running and accessible
- Check that device indices are correct using `--list-devices`
- Ensure the virtual device is properly installed

### Audio Dropouts
- Try increasing block size in client.py (e.g., 9600 for 200ms)
- Check network latency if server is remote
- Verify CPU isn't overloaded during inference

### Can't Hear Yourself
This is normal - the audio goes directly to the virtual device. To monitor:
- **Windows:** Use "Listen to this device" in sound properties
- **macOS:** Create a Multi-Output Device in Audio MIDI Setup
- **Linux:** Use PulseAudio Volume Control to route the loopback

### Virtual Device Not Showing Up
- **Windows:** Restart after VB-Cable installation
- **macOS:** Restart after BlackHole installation
- **Linux:** Check `dmesg | grep snd` for module loading errors

---

## Advanced Configuration

### Stereo Processing
If you need stereo (2 channels), modify in `client.py`:
```python
CHANNELS = 2  # Change from 1 to 2
```

### Custom Sample Rates
DeepFilterNet expects 48kHz, but you can resample:
```python
SAMPLE_RATE = 44100  # Your device's native rate
# Add resampling logic to convert to 48kHz before sending
```

### Lower Latency
Reduce block size for lower latency (at cost of more overhead):
```python
BLOCK_SIZE = 2400  # 50ms blocks instead of 100ms
```

---

## Platform-Specific Notes

### Windows
- VB-Cable is free but limited to one virtual device
- VoiceMeeter (free) or VoiceMeeter Banana/Potato (donation) provide more flexibility
- Restart is typically required after installation

### macOS
- BlackHole is open-source and free
- Supports both Intel and Apple Silicon
- No restart required but recommended
- Consider creating an Aggregate Device for monitoring

### Linux
- ALSA loopback is built into the kernel
- PulseAudio/PipeWire configuration may vary by distribution
- Most flexible but requires more configuration

---

## Uninstallation

### Windows (VB-Cable)
Control Panel → Programs → Uninstall → VB-CABLE Driver → Uninstall

### macOS (BlackHole)
```bash
brew uninstall blackhole-2ch
# Or use the uninstaller from the .pkg
```

### Linux (ALSA Loopback)
```bash
sudo rmmod snd-aloop
sudo rm /etc/modules-load.d/snd-aloop.conf
```
