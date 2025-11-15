# ToneHoner: Real-time DeepFilterNet Audio Enhancement

ToneHoner provides a FastAPI WebSocket server for streaming audio enhancement and a Python client that captures microphone audio, sends it to the server, and outputs enhanced audio to a virtual microphone device.

- Server: `main.py` (FastAPI + WebSocket `/enhance`) and `dfn_server.py` (DeepFilterNet TorchScript export + inference helpers)
- Client: `client/client.py` (sounddevice + websockets)
- Containerization: `Dockerfile`
- Kubernetes: `k8s/deployment.yaml`, `k8s/values.yaml`
- Extra docs: `PACKAGING.md`, `VIRTUAL_AUDIO_SETUP.md`


## Architecture

- FastAPI server (`main.py`)
  - `POST /ping`: health check returns `{ "status": "ok" }`
  - `WS /enhance`: receives raw PCM blocks (int16), converts to float32 [-1, 1], calls `enhance_block` from `dfn_server.py`, converts back to int16 and returns
  - Loads TorchScript model on startup via `load_torchscript_model("./models/model_ts.pt")`
- DeepFilterNet export and inference (`dfn_server.py`)
  - Installs DeepFilterNet from GitHub
  - Loads pre-trained DeepFilterNet-2
  - Exports to TorchScript `./models/model_ts.pt`
  - Inference helper `enhance_block(audio: torch.Tensor) -> torch.Tensor`
- Python client (`client/client.py`)
  - Captures mic with `sounddevice`, block-based (default 100 ms @ 48 kHz)
  - Async WebSocket to server `/enhance`
  - Writes enhanced blocks to a virtual audio output device (e.g., VB-Cable, BlackHole, ALSA Loopback)

Data flow:

Mic (int16) → Client (int16 → float32 → WS) → Server (float32 → enhance_block → float32) → Client (float32 → int16) → Virtual Mic


## Prerequisites

- Python: 3.10+ recommended (tested with 3.11)
- OS: Windows/macOS/Linux
- GPU (optional): NVIDIA CUDA 13.x for GPU acceleration; otherwise CPU fallback works
- Virtual audio device (for client output to apps):
  - Windows: VB-Cable (Chocolatey or manual)
  - macOS: BlackHole (Homebrew)
  - Linux: ALSA Loopback / PulseAudio / PipeWire
- Dependencies:
  - Server: see `requirements.txt` (FastAPI, uvicorn[standard], websockets, torch, numpy)
  - Client: see `client/requirements.txt` (sounddevice, numpy, websockets, scipy)

See `VIRTUAL_AUDIO_SETUP.md` for step-by-step virtual device setup.


## Setup: Model Export (one-time)

Exports DeepFilterNet-2 TorchScript to `./models/model_ts.pt`.

```powershell
# From repo root (Windows PowerShell)
python dfn_server.py
```

Expected output (abbrev.):
```
DeepFilterNet-2 TorchScript Export Script
Installing DeepFilterNet from GitHub...
CUDA is available! Version: 13.x  # or CPU fallback
DeepFilterNet-2 loaded successfully on cuda:0
Tracing model with TorchScript...
TorchScript model saved successfully to: ./models/model_ts.pt
Export completed successfully!
```

If CUDA is unavailable, the script falls back to CPU.


## Run: Server

Install server requirements and start.

```powershell
# Windows PowerShell (from repo root)
pip install -r requirements.txt
python main.py
```

Health check:
```powershell
# In a second terminal
curl http://localhost:8000/ping
# {"status":"ok"}
```

WebSocket endpoint: `ws://localhost:8000/enhance`


## Run: Client

Install client requirements and run. Choose devices via `--list-devices` then set indices.

```powershell
# From repo root
pip install -r client/requirements.txt

# GUI mode (recommended)
python client/client.py --gui

# File processing mode
python client/client.py --file input.wav --output output_enhanced.wav
```

The client now supports two modes:

1. **GUI Mode (Recommended)**: A tabbed interface with two tabs:
   - **Real-time Streaming**: Capture audio from microphone → enhance → output to virtual device
     - Select input/output devices
     - Start/stop streaming
     - View statistics (blocks sent/received, errors)
   - **File Processing**: Process audio files in batch mode
     - Browse and select input WAV file
     - Browse and select output location (auto-suggests filename)
     - Process file with progress bar and status updates
     - Both tabs can specify different server URLs

2. **CLI Mode**: Command-line operation
   - Real-time: `python client/client.py --server ws://localhost:8000/enhance`
   - File processing: `python client/client.py --file input.wav --output output.wav --server ws://localhost:8000/enhance`

```powershell
# List audio devices
python client/client.py --list-devices

# CLI: Real-time streaming with specific devices
python client/client.py --input 1 --output 3

# CLI: File processing
python client/client.py --file audio.wav --output enhanced.wav

# CLI mode
python client/client.py --list-devices
python client/client.py --input-device 1 --output-device 4 --server ws://localhost:8000/enhance
```

Expected client output (abbrev.):
```
Real-time Audio Enhancement Client
Sample rate: 48000 Hz, Block size: 4800 (100.0 ms)
✓ Audio stream started
✓ Connected to enhancement server
--- Stats ---
Blocks sent: 120, Blocks received: 120, Errors: 0
```

Note: The client outputs the enhanced stream to the selected virtual audio device (use it as the microphone in other apps).

GUI notes:
- Select input/output devices from dropdowns (defaults are available)
- Enter server URL (default: ws://localhost:8000/enhance)
- Start/Stop controls and live stats (blocks sent/received, errors)
- If devices don’t appear, ensure drivers are installed (see Virtual Audio Setup)


## Packaging (Client)

Quick start using PyInstaller (see `PACKAGING.md` for full coverage):

```powershell
# Windows
cd client
# CLI build
pyinstaller --onefile --name "DeepFilterNet-Client" client.py
# GUI build (no console window)
pyinstaller --onefile --noconsole --name "DeepFilterNet-Client" client.py
```

```bash
# macOS/Linux
cd client
# CLI build
pyinstaller --onefile --name deepfilternet-client client.py
# GUI build (no console window, macOS app bundle optional)
pyinstaller --onefile --windowed --name "DeepFilterNet Client" client.py
```

Install to PATH (macOS/Linux):
```bash
sudo cp dist/deepfilternet-client /usr/local/bin/
sudo chmod +x /usr/local/bin/deepfilternet-client
```

More options (icons, signing, installers, AppImage, DEB) in `PACKAGING.md`.


## Docker

Build and run the GPU-enabled container (uses `nvcr.io/nvidia/pytorch:25.01-py3` with CUDA 13 support).

```bash
# Build image (from repo root)
docker build -t tonehoner-server:latest .

# Run with GPU and mapped models directory
docker run --rm -it \
  --gpus all \
  -p 8000:8000 \
  -v %cd%/models:/app/models \
  tonehoner-server:latest
# On bash use: -v $(pwd)/models:/app/models
```

Test:
```bash
curl http://localhost:8000/ping
# {"status":"ok"}
```


## Kubernetes

Apply the provided manifests or use Helm values.

```bash
# NodePort service on 8000 (nodePort 30800 by default)
kubectl apply -f k8s/deployment.yaml

# Check service and pods
kubectl get svc deepfilternet-server
kubectl get pods -l app=deepfilternet-server
```

Access:
```bash
# Replace with actual node IP
curl http://<NODE_IP>:30800/ping
```

Helm (values in `k8s/values.yaml`):
```bash
helm install deepfilternet-server ./deepfilternet-server -f k8s/values.yaml \
  --set image.repository=your-registry.example.com/deepfilternet-server \
  --set image.tag=latest
```


## Troubleshooting

- Latency too high
  - Reduce block size in `client/client.py` (e.g., 2400 = 50 ms)
  - Ensure server and client are on the same LAN or host
  - On Docker/K8s, verify GPU pass-through and avoid CPU-only
- No audio or silence
  - Confirm virtual device is installed and selected as `OUTPUT_DEVICE`
  - Use `--list-devices` to find correct indices
  - Ensure sample rate is 48 kHz; resample if needed
  - In GUI mode, verify you clicked Start and selected valid devices
- GPU memory or CUDA errors
  - Ensure NVIDIA driver + CUDA 13.x match PyTorch version
  - Use `docker run --gpus all` or K8s GPU resources
  - Fall back to CPU: server will still run, but slower
- WebSocket disconnects / frame errors
  - Keep consistent block size between client and server (default 4800)
  - Add retry/backoff in client; check server logs
- Model missing
  - Run `python dfn_server.py` to export `./models/model_ts.pt`
  - Verify `models/` is mounted in Docker/K8s
 - GUI doesn’t launch (Linux)
   - Ensure Tkinter is installed (e.g., `sudo apt-get install python3-tk`)
   - For remote sessions, ensure X11 forwarding or use CLI mode


## Example Commands & Expected Outputs

Health check:
```powershell
curl http://localhost:8000/ping
# {"status":"ok"}
```

Server start (abbrev.):
```
INFO:     Uvicorn running on http://0.0.0.0:8000
Loading DeepFilterNet model...
Model loaded successfully - server ready!
```

Client (abbrev.):
```
✓ Audio stream started
✓ Connected to enhancement server
--- Stats ---
Blocks sent: 60
Blocks received: 60
Errors: 0
```

Integration test:
```powershell
python client/test_integration.py
# ✓✓✓ INTEGRATION TEST PASSED ✓✓✓
```


## Future Work

- Authentication and authorization (e.g., API tokens/JWT for WS)
- TLS termination and secure WebSockets (wss://) with cert management
- Dynamic block size negotiation per client for adaptive latency
- CPU/GPU adaptive mode (auto-switch with load/availability)
- Multi-channel/stereo support and resampling pipeline
- Monitoring endpoints (/metrics) and tracing
- Model hot-reload and versioning
- Rate limiting and per-tenant quotas


## Repository Layout

```
.
├── main.py                    # FastAPI app (ping + WS /enhance)
├── dfn_server.py              # DeepFilterNet export + inference helpers
├── Dockerfile                 # Server container image
├── requirements.txt           # Server deps
├── client/
│   ├── client.py              # Real-time client (mic → WS → virtual device)
│   ├── requirements.txt       # Client deps
│   └── test_integration.py    # End-to-end test with WAV
├── k8s/
│   ├── deployment.yaml        # Deployment, Service, PVC
│   ├── values.yaml            # Helm values
│   └── README.md              # K8s guide
├── PACKAGING.md               # Packaging client into standalone executables
└── VIRTUAL_AUDIO_SETUP.md     # Virtual device setup for Windows/macOS/Linux
```
