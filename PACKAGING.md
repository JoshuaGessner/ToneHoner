# Packaging Instructions

This guide explains how to package the ToneHoner Server (GUI) and Client into standalone executables. Windows PowerShell examples are provided; macOS/Linux notes remain below.

## Prerequisites

Install PyInstaller in your Python environment:

```bash
pip install pyinstaller
```

---

## Windows

### Build Both Executables (recommended)

Use the provided PowerShell script which builds using the included `.spec` files and collects required package data (torch, torchaudio, df, fastapi, starlette, websockets, sounddevice, numpy, etc.).

```powershell
# From repo root, with your venv activated
python -m pip install -r .\requirements.txt
python -m pip install pyinstaller

./build_executables.ps1

# Outputs:
# dist/ToneHoner-Server.exe  (Server GUI; no console)
# dist/ToneHoner-Client.exe  (Client GUI; no console)
```

Notes:
- First run may take a while due to torch packaging.
- Startup of onefile torch apps can be slower; subsequent runs are faster.

### Build Client Single Executable (manual alternative)

```powershell
# Navigate to the client directory
cd client

# Create single-file executable
pyinstaller --onefile --noconsole --name "ToneHoner-Client" client/client.py

# The executable will be in: dist\DeepFilterNet-Client.exe
```

### Build Server GUI (manual alternative)

```powershell
pyinstaller --onefile --noconsole --name "ToneHoner-Server" server_gui.py
```

This is simpler but may miss some dynamic dependencies. Prefer the provided `tonehoner_server.spec` and `tonehoner_client.spec` which use `collect_all()` for deep dependencies.

```powershell
pyinstaller --onefile --noconsole --name "DeepFilterNet-Client" client.py
```

> Note: The client includes a Tkinter GUI (`--gui` flag not required for packaged GUI if you use `--noconsole`).

### Build with Icon (Optional)

```powershell
# If you have an icon file (icon.ico)
pyinstaller --onefile --name "DeepFilterNet-Client" --icon=icon.ico client.py
```

### Build with Console Hidden (Optional)

For a GUI-like experience without the console window:

```powershell
pyinstaller --onefile --noconsole --name "DeepFilterNet-Client" client.py
```

### Installation

**Option 1: User Installation**
```powershell
# Copy to user's local bin directory
Copy-Item dist\DeepFilterNet-Client.exe $env:LOCALAPPDATA\Microsoft\WindowsApps\
```

**Option 2: System-wide Installation**
```powershell
# Copy to system directory (requires admin)
Copy-Item dist\DeepFilterNet-Client.exe C:\Windows\System32\
```

**Option 3: Portable**
```powershell
# Simply distribute the .exe file
# Users can run it from any location
```

### Running the Executable

```powershell
# If installed to PATH
DeepFilterNet-Client.exe --list-devices

# Or run directly
.\dist\DeepFilterNet-Client.exe --input-device 1 --output-device 4
```

### Creating an Installer (Advanced)

Use NSIS (Nullsoft Scriptable Install System) or Inno Setup to create a proper Windows installer:

```powershell
# Install NSIS via Chocolatey
choco install nsis -y

# Create installer script (example: installer.nsi)
# Then compile: makensis installer.nsi
```

---

## macOS

### Build Single Executable

```bash
# Navigate to the client directory
cd client

# Create single-file executable (client)
pyinstaller --onefile --name "deepfilternet-client" client.py

# The executable will be in: dist/deepfilternet-client
```

### Build macOS Application Bundle (.app) for Server GUI

```bash
# Create .app bundle with windowed mode
pyinstaller --onefile --windowed --name "ToneHoner Server" server_gui.py

# The .app will be in: dist/DeepFilterNet Client.app
```

### Build with Icon (Optional)

```bash
# If you have an icon file (icon.icns)
pyinstaller --onefile --windowed --name "DeepFilterNet Client" --icon=icon.icns client.py
```

### Build for Universal Binary (Intel + Apple Silicon)

```bash
# Install PyInstaller with universal2 support
pip install pyinstaller

# Build universal binary
pyinstaller --onefile --target-arch universal2 --name "deepfilternet-client" client.py
```

### Installation

**Option 1: Install to /usr/local/bin**
```bash
# Copy executable (requires sudo)
sudo cp dist/deepfilternet-client /usr/local/bin/

# Make executable
sudo chmod +x /usr/local/bin/deepfilternet-client

# Verify installation
which deepfilternet-client
```

**Option 2: Install .app to Applications**
```bash
# Copy .app bundle to Applications folder
cp -r "dist/DeepFilterNet Client.app" /Applications/

# Or with sudo for system-wide
sudo cp -r "dist/DeepFilterNet Client.app" /Applications/
```

**Option 3: User-local Installation**
```bash
# Create user bin directory if it doesn't exist
mkdir -p ~/bin

# Copy executable
cp dist/deepfilternet-client ~/bin/

# Add to PATH in ~/.zshrc or ~/.bash_profile
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Running the Executable

```bash
# If installed to PATH
deepfilternet-client --list-devices

# Or run .app bundle
open "/Applications/DeepFilterNet Client.app"

# Or run directly
./dist/deepfilternet-client --input-device 1 --output-device 2
```

GUI usage:
```bash
./dist/deepfilternet-client --gui
```

### Code Signing (Optional but Recommended)

To avoid "unidentified developer" warnings:

```bash
# Sign the application (requires Apple Developer account)
codesign --force --deep --sign "Developer ID Application: Your Name" "dist/DeepFilterNet Client.app"

# Verify signature
codesign --verify --verbose "dist/DeepFilterNet Client.app"

# Notarize with Apple (requires Xcode command-line tools)
xcrun notarytool submit "DeepFilterNet Client.app" --keychain-profile "AC_PASSWORD"
```

### Creating a DMG (Optional)

```bash
# Install create-dmg
brew install create-dmg

# Create DMG installer
create-dmg \
  --volname "DeepFilterNet Client" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --app-drop-link 600 185 \
  "DeepFilterNet-Client.dmg" \
  "dist/DeepFilterNet Client.app"
```

---

## Linux

### Build Single Executable

```bash
# Navigate to the client directory
cd client

# Create single-file executable
pyinstaller --onefile --name "deepfilternet-client" client.py

# The executable will be in: dist/deepfilternet-client
```

### Build with Reduced Size

```bash
# Strip debug symbols and use UPX compression
pyinstaller --onefile --strip --name "deepfilternet-client" client.py

# Or with UPX (if installed)
pyinstaller --onefile --strip --upx-dir=/usr/bin --name "deepfilternet-client" client.py
```

### Installation

**Option 1: System-wide Installation**
```bash
# Copy to /usr/local/bin (requires sudo)
sudo cp dist/deepfilternet-client /usr/local/bin/

# Make executable
sudo chmod +x /usr/local/bin/deepfilternet-client

# Verify installation
which deepfilternet-client
```

**Option 2: User-local Installation**
```bash
# Create user bin directory if it doesn't exist
mkdir -p ~/.local/bin

# Copy executable
cp dist/deepfilternet-client ~/.local/bin/

# Make executable
chmod +x ~/.local/bin/deepfilternet-client

# Add to PATH in ~/.bashrc or ~/.zshrc if not already
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Option 3: Create .desktop Entry (GUI Launcher)**
```bash
# Create desktop entry
cat > ~/.local/share/applications/deepfilternet-client.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=DeepFilterNet Client
Comment=Real-time audio enhancement client
Exec=/usr/local/bin/deepfilternet-client
Icon=audio-headset
Terminal=true
Categories=Audio;AudioVideo;
EOF

# Update desktop database
update-desktop-database ~/.local/share/applications/
```

### Running the Executable

```bash
# If installed to PATH
deepfilternet-client --list-devices

# Or run directly
./dist/deepfilternet-client --input-device 1 --output-device 6
```

### Creating a Tarball Distribution

```bash
# Create tarball with executable and documentation
mkdir -p deepfilternet-client-package
cp dist/deepfilternet-client deepfilternet-client-package/
cp ../VIRTUAL_AUDIO_SETUP.md deepfilternet-client-package/
cp requirements.txt deepfilternet-client-package/

# Create README for the package
cat > deepfilternet-client-package/README.txt << EOF
DeepFilterNet Client - Standalone Executable

Installation:
1. Copy deepfilternet-client to ~/.local/bin/ or /usr/local/bin/
2. Make executable: chmod +x deepfilternet-client
3. Run: deepfilternet-client --list-devices

See VIRTUAL_AUDIO_SETUP.md for virtual audio device setup.
EOF

# Create tarball
tar -czf deepfilternet-client-linux-x64.tar.gz deepfilternet-client-package/

# Clean up
rm -rf deepfilternet-client-package/
```

### Creating a DEB Package (Debian/Ubuntu)

```bash
# Create package structure
mkdir -p deepfilternet-client_1.0.0/DEBIAN
mkdir -p deepfilternet-client_1.0.0/usr/local/bin

# Copy executable
cp dist/deepfilternet-client deepfilternet-client_1.0.0/usr/local/bin/
chmod +x deepfilternet-client_1.0.0/usr/local/bin/deepfilternet-client

# Create control file
cat > deepfilternet-client_1.0.0/DEBIAN/control << EOF
Package: deepfilternet-client
Version: 1.0.0
Section: sound
Priority: optional
Architecture: amd64
Maintainer: Your Name <your.email@example.com>
Description: Real-time audio enhancement client using DeepFilterNet
 A client application that captures audio from a microphone,
 enhances it using DeepFilterNet AI, and outputs to a virtual
 audio device for use in other applications.
EOF

# Build package
dpkg-deb --build deepfilternet-client_1.0.0

# Install package
sudo dpkg -i deepfilternet-client_1.0.0.deb
```

### Creating an AppImage (Universal Linux)

```bash
# Install appimagetool
wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x appimagetool-x86_64.AppImage

# Create AppDir structure
mkdir -p DeepFilterNet-Client.AppDir/usr/bin
cp dist/deepfilternet-client DeepFilterNet-Client.AppDir/usr/bin/

# Create AppRun script
cat > DeepFilterNet-Client.AppDir/AppRun << 'EOF'
#!/bin/bash
SELF=$(readlink -f "$0")
HERE=${SELF%/*}
exec "${HERE}/usr/bin/deepfilternet-client" "$@"
EOF
chmod +x DeepFilterNet-Client.AppDir/AppRun

# Create .desktop file
cat > DeepFilterNet-Client.AppDir/deepfilternet-client.desktop << EOF
[Desktop Entry]
Name=DeepFilterNet Client
Exec=deepfilternet-client
Icon=audio-headset
Type=Application
Categories=Audio;
EOF

# Create AppImage
./appimagetool-x86_64.AppImage DeepFilterNet-Client.AppDir DeepFilterNet-Client-x86_64.AppImage
```

---

## Cross-Platform Considerations

### Dependencies

PyInstaller automatically bundles most dependencies, but some may require special handling:

**sounddevice / PortAudio:**
- Windows: Usually works out of the box
- macOS: May need to include PortAudio library manually
- Linux: May need to include ALSA/PulseAudio libraries

**Tkinter (GUI):**
- Bundled with most Python distributions on Windows/macOS
- On Linux you may need `python3-tk` system package:
  ```bash
  sudo apt-get install -y python3-tk
  ```

**Fix for missing audio libraries:**
```bash
# Add hidden imports
pyinstaller --onefile \
  --hidden-import=sounddevice \
  --hidden-import=_portaudio \
  client.py
```

### Spec Files (included)

We ship two spec files tailored for this project:

- `tonehoner_server.spec`: bundles `server_gui.py` with FastAPI/Uvicorn/torch/DeepFilterNet assets.
- `tonehoner_client.spec`: bundles `client/client.py` with sounddevice/websockets/numpy.

Build them with:

```powershell
python -m PyInstaller -y .\tonehoner_server.spec
python -m PyInstaller -y .\tonehoner_client.spec
```

Create a custom `.spec` file for reproducible builds:

```bash
# Generate spec file
pyinstaller --onefile client.py --name deepfilternet-client

# Edit deepfilternet-client.spec as needed

# Build from spec file
pyinstaller deepfilternet-client.spec
```

Example `.spec` file modifications (generic):
```python
# deepfilternet-client.spec
a = Analysis(
    ['client.py'],
    pathex=[],
    binaries=[],
    datas=[('../VIRTUAL_AUDIO_SETUP.md', '.')],  # Include documentation
    hiddenimports=['sounddevice', '_portaudio'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)
```

---

## Testing the Packaged Application

After building, test the executable:

```bash
# Test listing devices
./dist/deepfilternet-client --list-devices

# Test with actual devices (adjust indices)
./dist/deepfilternet-client --input-device 1 --output-device 4 --server ws://localhost:8000/enhance

# Test help
./dist/deepfilternet-client --help
```

---

## Distribution

### GitHub Releases

1. Create a new release on GitHub
2. Upload the executables:
   - `DeepFilterNet-Client.exe` (Windows)
   - `deepfilternet-client` (macOS/Linux)
   - `DeepFilterNet-Client.dmg` (macOS installer)
   - `deepfilternet-client-linux-x64.tar.gz` (Linux tarball)
   - `DeepFilterNet-Client-x86_64.AppImage` (Linux AppImage)

### Checksums

Generate checksums for security:

```bash
# SHA256 checksums
sha256sum dist/deepfilternet-client > dist/SHA256SUMS

# On macOS
shasum -a 256 dist/deepfilternet-client > dist/SHA256SUMS

# On Windows (PowerShell)
Get-FileHash dist\DeepFilterNet-Client.exe -Algorithm SHA256 | Format-List
```

---

## Troubleshooting

### "Failed to execute script" Error

Add debugging to see the error:
```bash
pyinstaller --onefile --debug all client.py
```

### Missing Modules

Add hidden imports:
```bash
pyinstaller --onefile --hidden-import=MODULE_NAME client.py
```

### Large File Size

Exclude unnecessary packages:
```bash
pyinstaller --onefile --exclude-module matplotlib --exclude-module PIL client.py
```

### Antivirus False Positives

Some antivirus software flags PyInstaller executables. Solutions:
- Code sign the executable
- Submit to antivirus vendors for whitelisting
- Distribute source code as alternative

---

## File Size Comparison

Typical sizes after packaging:

- **Windows**: ~50-80 MB (with all dependencies)
- **macOS**: ~40-70 MB (universal binary larger)
- **Linux**: ~45-75 MB (varies by included libraries)

To reduce size:
- Use `--exclude-module` for unused packages
- Enable UPX compression: `--upx-dir=/path/to/upx`
- Strip symbols: `--strip` (Linux/macOS)
