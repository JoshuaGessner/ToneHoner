"""
FastAPI WebSocket server for DeepFilterNet audio enhancement.
This is a minimal skeleton with health check and WebSocket placeholder.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import torch
import os
import asyncio
import struct
from collections import deque
from typing import Optional

# Import the enhancement function from dfn_server
from dfn_server import enhance_block, load_deepfilternet_for_inference, unload_model

# Initialize FastAPI application
app = FastAPI(title="DeepFilterNet Audio Enhancement API")

# CORS middleware - configure allowed origins for production
# For development, allow all origins. For production, specify your domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production: ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: Simple API key authentication
# Set environment variable: API_KEY=your_secret_key
API_KEY = os.getenv("API_KEY", None)  # None = no authentication required

# Lightweight metrics storage
_metrics = {
    "frames_processed": 0,
    "framed_frames": 0,
    "avg_proc_ms": 0.0,
    "last_proc_ms": 0.0,
    "avg_rtf": 0.0,
    "queue_backlog": 0,
    "sample_rate": 48000,
}
_proc_window = deque(maxlen=100)


@app.on_event("startup")
async def startup_event():
    """Load model and perform warm-up inferences to reduce first-frame latency."""
    print("Loading DeepFilterNet model...")
    try:
        load_deepfilternet_for_inference()
        # Warm-up: run several dummy frames (20ms @ 48kHz = 960 samples)
        dummy = torch.zeros(960, dtype=torch.float32)
        for _ in range(5):
            _ = enhance_block(dummy)
        print("Model warm-up complete; server ready for low-latency processing")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        print("Server will start but enhancement will not work.")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Unload the DeepFilterNet model at server shutdown.
    This frees up memory and GPU resources.
    """
    print("Shutting down server...")
    unload_model()
    print("Server shutdown complete")


@app.post("/ping")
async def health_check():
    """
    Health check endpoint.
    Returns a simple status response to verify the server is running.
    """
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    """Expose simple processing metrics for observability."""
    return JSONResponse(_metrics)


@app.post("/control/reset")
async def control_reset():
    """Unload and reload the model to apply new settings.

    If env vars like TONEHONER_FP16, TONEHONER_DEBUG_RTF, or TONEHONER_METRICS_INTERVAL
    were changed externally (e.g., by a controller app), this will reinitialize the model
    and perform a warm-up.
    """
    try:
        unload_model()
        load_deepfilternet_for_inference()
        # Warm-up
        dummy = torch.zeros(960, dtype=torch.float32)
        for _ in range(3):
            _ = enhance_block(dummy)
        return {"status": "reloaded"}
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@app.websocket("/enhance")
async def websocket_enhance(websocket: WebSocket):
    """Real-time enhancement endpoint with receive/process pipeline.

    Separates network I/O (producer) from enhancement (consumer) to overlap
    frame reception with model inference, reducing end-to-end latency.
    """
    if API_KEY:
        api_key = websocket.query_params.get("api_key")
        if api_key != API_KEY:
            await websocket.close(code=1008, reason="Invalid or missing API key")
            print("Client connection rejected: Invalid API key")
            return

    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"Client connected from {client_host}")

    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)
    running = True

    MAGIC = b"THv1"

    def _unpack_frame(buf: bytes):
        """Return (seq, t_send_ms, pcm_bytes) if framed, else (None, None, buf)."""
        try:
            if len(buf) >= 16 and buf[:4] == MAGIC:
                seq = struct.unpack_from("<I", buf, 4)[0]
                t_send_ms = struct.unpack_from("<d", buf, 8)[0]
                return seq, t_send_ms, buf[16:]
        except Exception:
            pass
        return None, None, buf

    def _pack_frame(seq, t_send_ms, pcm_bytes: bytes):
        try:
            if seq is not None and t_send_ms is not None:
                header = MAGIC + struct.pack("<I", seq) + struct.pack("<d", t_send_ms)
                return header + pcm_bytes
        except Exception:
            pass
        return pcm_bytes

    async def producer():
        try:
            while running:
                audio_bytes = await websocket.receive_bytes()
                # Drop oldest if full to favor freshest audio
                if queue.full():
                    try:
                        _ = queue.get_nowait()
                    except Exception:
                        pass
                await queue.put(audio_bytes)
        except asyncio.CancelledError:
            # Graceful cancellation during shutdown
            return
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"Producer error ({client_host}): {e}")

    async def consumer():
        frame_count = 0
        import time as _time
        last_log = _time.time()
        proc_times = []  # ms
        log_interval = float(os.getenv("TONEHONER_METRICS_INTERVAL", "5"))
        sr_assumed = 48000.0  # assume 48k input
        while running:
            try:
                audio_bytes = await queue.get()
                frame_count += 1
                t0 = _time.perf_counter()
                seq, t_send_ms, payload = _unpack_frame(audio_bytes)
                if seq is not None:
                    _metrics["framed_frames"] += 1
                audio_int16 = np.frombuffer(payload, dtype=np.int16)
                # Fast path conversion without intermediate copy
                audio_float32 = (audio_int16.astype(np.float32) * (1.0 / 32768.0))
                audio_tensor = torch.from_numpy(audio_float32)
                # Offload enhancement to thread to keep event loop responsive
                enhanced_tensor = await asyncio.to_thread(enhance_block, audio_tensor)
                enhanced_float32 = enhanced_tensor.detach().cpu().numpy().astype(np.float32)
                enhanced_float32 = np.clip(enhanced_float32, -1.0, 1.0)
                enhanced_int16 = (enhanced_float32 * 32767.0).astype(np.int16)
                out_bytes = enhanced_int16.tobytes()
                out_bytes = _pack_frame(seq, t_send_ms, out_bytes)
                await websocket.send_bytes(out_bytes)
                t1 = _time.perf_counter()
                proc_ms = (t1 - t0) * 1000.0
                proc_times.append(proc_ms)
                # Update metrics
                _metrics["frames_processed"] += 1
                _metrics["last_proc_ms"] = proc_ms
                _metrics["queue_backlog"] = queue.qsize()
                # Periodic lightweight metrics
                now = _time.time()
                if now - last_log >= log_interval:
                    try:
                        if proc_times:
                            avg = sum(proc_times) / len(proc_times)
                            dur_ms = len(audio_int16) / sr_assumed * 1000.0
                            rtf = (avg / dur_ms) if dur_ms > 0 else 0.0
                            print(f"[metrics] frames={frame_count} avg_proc={avg:.2f} ms frame={dur_ms:.1f} ms RTF={rtf:.3f}")
                            _metrics["avg_proc_ms"] = avg
                            _metrics["avg_rtf"] = rtf
                    except Exception:
                        pass
                    finally:
                        proc_times.clear()
                        last_log = now
            except asyncio.CancelledError:
                # Graceful cancellation during shutdown
                return
            except WebSocketDisconnect:
                break
            except Exception as e:
                # Minimal logging to avoid hot-path overhead
                print(f"Enhancement error ({client_host}) frame {frame_count}: {e}")
                try:
                    await websocket.send_bytes(audio_bytes)
                except Exception:
                    pass

    producer_task = asyncio.create_task(producer())
    consumer_task = asyncio.create_task(consumer())

    try:
        done, pending = await asyncio.wait(
            {producer_task, consumer_task}, return_when=asyncio.FIRST_EXCEPTION
        )
    except asyncio.CancelledError:
        # Server is shutting down; cancel child tasks gracefully
        pass
    finally:
        running = False
        for task in (producer_task, consumer_task):
            if not task.done():
                task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass
        print(f"WebSocket connection closed from {client_host}")


if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
