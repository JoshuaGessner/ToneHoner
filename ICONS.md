# ToneHoner Icons

This directory contains icon files for the ToneHoner application suite.

## Icon Files

### Application Icons
- **client_icon.ico** / **client_icon.png** - Blue-themed icon for ToneHoner Client
  - Represents the client/microphone input side of the application
  - Blue gradient with waveform visualization
  
- **server_icon.ico** / **server_icon.png** - Green-themed icon for ToneHoner Server
  - Represents the server/processing side of the application
  - Green/teal gradient with waveform visualization

- **installer_icon.ico** / **installer_icon.png** - Purple/Orange-themed icon for installers
  - Used for setup/installer packages
  - Purple to orange gradient with waveform visualization

## Regenerating Icons

If you need to regenerate the icons (e.g., for design changes), run:

```powershell
python generate_icons.py
```

This will create new .ico and .png files for all three icon types.

## Icon Specifications

- **Format**: ICO (Windows Icon) with multiple sizes embedded
- **Sizes**: 256x256, 128x128, 64x64, 48x48, 32x32, 16x16
- **Design**: Gradient background with stylized audio waveform bars
- **Labels**: Letter indicators (C/S/T) on larger sizes for easy identification

## Usage

The icons are automatically referenced in:
- **PyInstaller spec files**: `tonehoner_client.spec`, `tonehoner_server.spec`
- **Inno Setup scripts**: `tonehoner_client_setup.iss`, `tonehoner_server_setup.iss`

When building executables or installers, the build process will use these icons automatically.

## Dependencies

Icon generation requires:
- Python 3.10+
- Pillow library: `pip install Pillow`

## Color Schemes

### Client (Blue)
- Gradient: Dark Blue (#1E3C72) → Bright Blue (#2A9DF4)
- Waveform: Light Cyan (#64C8FF)

### Server (Green)
- Gradient: Dark Teal (#22577A) → Green (#57BB8A)
- Waveform: Light Green (#90EE90)

### Installer (Purple/Orange)
- Gradient: Purple (#581845) → Orange (#C77D48)
- Waveform: Light Orange (#FFC371)
