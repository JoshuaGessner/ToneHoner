# ToneHoner Icons - Quick Reference

## ✅ What Was Created

### Icon Files
- ✓ `client_icon.ico` + `client_icon.png` - Blue theme for Client
- ✓ `server_icon.ico` + `server_icon.png` - Green theme for Server  
- ✓ `installer_icon.ico` + `installer_icon.png` - Purple/Orange theme

### Updated Files
- ✓ `tonehoner_client.spec` - Now references `client_icon.ico`
- ✓ `tonehoner_server.spec` - Now references `server_icon.ico`
- ✓ `tonehoner_client_setup.iss` - Now references `client_icon.ico`
- ✓ `tonehoner_server_setup.iss` - Now references `server_icon.ico`

### Documentation
- ✓ `ICONS.md` - Icon documentation and specifications
- ✓ `BUILD_WITH_ICONS.md` - Complete build guide
- ✓ `generate_icons.py` - Icon generation script
- ✓ `build_all.ps1` - Automated build script

## 🚀 Quick Start

### Option 1: Build Everything (Recommended)
```powershell
.\build_all.ps1
```

### Option 2: Build Manually

**Executables:**
```powershell
pyinstaller tonehoner_client.spec
pyinstaller tonehoner_server.spec
```

**Installers:**
```powershell
iscc tonehoner_client_setup.iss
iscc tonehoner_server_setup.iss
```

## 📦 Output Files

After building, you'll have:

```
dist/
├── ToneHoner-Client/
│   ├── ToneHoner-Client.exe      # Client executable with blue icon
│   └── [dependencies...]          # Required DLLs and libraries
├── ToneHoner-Server/
│   ├── ToneHoner-Server.exe      # Server executable with green icon
│   └── [dependencies...]          # Required DLLs and libraries
└── installers/
    ├── ToneHoner-Client-Setup-1.0.0.exe  # Client installer
    └── ToneHoner-Server-Setup-1.0.0.exe  # Server installer
```

## 🎨 Icon Color Schemes

| Application | Colors | Theme |
|------------|---------|--------|
| **Client** | Dark Blue → Bright Blue | Microphone/Input |
| **Server** | Dark Teal → Green | Processing/Enhancement |
| **Installer** | Purple → Orange | Setup/Installation |

## 🔄 Regenerate Icons

To modify or regenerate icons:

```powershell
# Edit generate_icons.py to change design
python generate_icons.py
```

## 🛠️ Build Options

### Clean Build
```powershell
.\build_all.ps1 -Clean
```

### Skip Executables (only build installers)
```powershell
.\build_all.ps1 -SkipExecutables
```

### Skip Installers (only build executables)
```powershell
.\build_all.ps1 -SkipInstallers
```

## 📋 Prerequisites

- Python 3.10+ with PyInstaller
- Pillow library (for icon generation)
- Inno Setup (for installers)

Install missing dependencies:
```powershell
pip install pyinstaller Pillow
choco install innosetup
```

## 🎯 Next Steps

1. **Test the build script:**
   ```powershell
   .\build_all.ps1
   ```

2. **Verify icons appear correctly:**
   - Check executables in `dist/ToneHoner-Client/` and `dist/ToneHoner-Server/`
   - Run installers from `dist/installers/`
   - Verify icons in Start Menu after installation

3. **Distribute:**
   - Share the installers with users (recommended)
   - Or zip the folders from `dist/` for portable use

## 📚 More Information

- Full build guide: `BUILD_WITH_ICONS.md`
- Icon details: `ICONS.md`
- Project README: `README.md`
