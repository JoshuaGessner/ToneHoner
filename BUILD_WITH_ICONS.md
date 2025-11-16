# Building ToneHoner with Icons

This guide shows how to build the ToneHoner executables and installers with custom icons.

## Quick Build Commands

### 1. Build Executables with Icons

```powershell
# Build Client executable
pyinstaller tonehoner_client.spec

# Build Server executable
pyinstaller tonehoner_server.spec
```

The executables will be created in `dist/`:
- `dist/ToneHoner-Client/` (folder with client executable and dependencies)
- `dist/ToneHoner-Server/` (folder with server executable and dependencies)

### 2. Build Installers with Icons

First, ensure you have Inno Setup installed:
```powershell
# Install via Chocolatey (if not already installed)
choco install innosetup
```

Then build the installers:
```powershell
# Build Client installer
iscc tonehoner_client_setup.iss

# Build Server installer
iscc tonehoner_server_setup.iss
```

The installers will be created in `dist/installers/`:
- `dist/installers/ToneHoner-Client-Setup-1.0.0.exe`
- `dist/installers/ToneHoner-Server-Setup-1.0.0.exe`

### 3. Build Everything at Once

```powershell
# Build both executables
pyinstaller tonehoner_client.spec
pyinstaller tonehoner_server.spec

# Build both installers
iscc tonehoner_client_setup.iss
iscc tonehoner_server_setup.iss
```

## Icon Details

All executables and installers now include custom icons:

- **Client**: Blue-themed icon with waveform (microphone/input focus)
- **Server**: Green-themed icon with waveform (processing/enhancement focus)
- **Installers**: Use the respective application icons

## Regenerating Icons

If you need to modify the icon design:

1. Edit `generate_icons.py` to change colors, design, or labels
2. Run the generator:
   ```powershell
   python generate_icons.py
   ```
3. Rebuild executables and installers

## Icon Files

The following icon files are used:
- `client_icon.ico` - PyInstaller client spec, Inno Setup client installer
- `server_icon.ico` - PyInstaller server spec, Inno Setup server installer
- `installer_icon.ico` - Available for custom setup branding (currently unused)

## Troubleshooting

### Icon not showing in executable
- Ensure the .ico file exists in the project root
- Verify the path in the .spec file is correct (relative to project root)
- Rebuild with `pyinstaller --clean tonehoner_*.spec`

### Icon not showing in installer
- Ensure the .ico file exists in the project root
- Verify the path in the .iss file is correct (relative to script location)
- Check Inno Setup compilation log for errors

### Icon appears corrupted
- Regenerate icons: `python generate_icons.py`
- Ensure Pillow is installed: `pip install Pillow`
- ICO files must contain multiple sizes (16x16 to 256x256)

## Distribution Checklist

Before distributing:
- [ ] Build executables with icons
- [ ] Test executables launch with correct icons
- [ ] Build installers with icons
- [ ] Test installers display correct icons during setup
- [ ] Verify installed applications show correct icons in Start Menu
- [ ] Verify desktop shortcuts (if created) show correct icons

## Version Updates

When updating the version:
1. Update `#define MyAppVersion` in both .iss files
2. Rebuild executables and installers
3. New installer names will automatically include the version number
