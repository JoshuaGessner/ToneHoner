# CUDA 13 Setup Instructions

## Current Status
Your system has **PyTorch CPU-only version** installed. To enable GPU acceleration with CUDA 13, follow these steps:

## Prerequisites

1. **NVIDIA GPU** with compute capability 5.0 or higher
2. **NVIDIA Driver** version 545.23.06 or newer (for CUDA 13.0)
3. **CUDA Toolkit 13.0** (optional - PyTorch includes CUDA libraries)

## Check Your GPU

```powershell
# Check if you have an NVIDIA GPU
nvidia-smi
```

If this command works and shows your GPU, you're ready to proceed.

## Install CUDA-Enabled PyTorch

### Option 1: Install PyTorch with CUDA 12.4 (Recommended for compatibility)

PyTorch 2.5.1 officially supports CUDA 12.4, which works with CUDA 13 drivers:

```powershell
# Uninstall CPU version
pip uninstall torch torchaudio -y

# Install CUDA 12.4 version
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### Option 2: Install PyTorch with CUDA 12.1

If you encounter issues, try the CUDA 12.1 build:

```powershell
# Uninstall CPU version
pip uninstall torch torchaudio -y

# Install CUDA 12.1 version
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

## Verify Installation

After installation, verify CUDA is working:

```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
CUDA available: True
CUDA version: 12.4
GPU: NVIDIA GeForce RTX XXXX
```

## Reinstall DeepFilterNet

After installing CUDA-enabled PyTorch, reinstall DeepFilterNet:

```powershell
pip uninstall deepfilternet deepfilterlib -y
pip install deepfilternet
```

## Test the Setup

Run the setup script again:

```powershell
python dfn_server.py
```

You should now see:
```
CUDA is available! Version: 12.4
GPU Device: NVIDIA GeForce RTX XXXX
```

## Troubleshooting

### "CUDA not available" after installation

1. **Check driver version:**
   ```powershell
   nvidia-smi
   ```
   Driver version should be 545.23.06 or newer.

2. **Restart your terminal** after installation

3. **Check PyTorch build:**
   ```powershell
   python -c "import torch; print(torch.__version__)"
   ```
   Should show something like `2.5.1+cu124` (not `2.5.1+cpu`)

### OutOfMemoryError

If you get CUDA out of memory errors:
- Reduce batch size (the code already uses small chunks)
- Close other GPU-intensive applications
- The model will automatically fall back to CPU if GPU memory is insufficient

### Mixed CUDA Versions

If you have multiple CUDA versions installed:
- PyTorch includes its own CUDA libraries, so you don't need the full CUDA Toolkit
- Make sure `nvidia-smi` shows a compatible driver version
- PyTorch's bundled CUDA (12.4) will work with CUDA 13 drivers

## Performance Comparison

- **CPU**: RTF (Real-Time Factor) ~1-10x (slower than real-time)
- **GPU**: RTF ~0.1-0.5x (2-10x faster than real-time)

For real-time audio processing, GPU is highly recommended.

## Docker with CUDA

If using Docker, make sure to:
1. Install **NVIDIA Container Toolkit**
2. Use `--gpus all` flag when running the container
3. The Dockerfile already uses CUDA-enabled base image

```bash
docker run --gpus all -p 8000:8000 tonehoner-server:latest
```
