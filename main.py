"""
FastAPI WebSocket server for DeepFilterNet audio enhancement.
This is a minimal skeleton with health check and WebSocket placeholder.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import torch
import os

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


@app.on_event("startup")
async def startup_event():
    """
    Load the DeepFilterNet model at server startup.
    This ensures the model is ready before accepting WebSocket connections.
    """
    print("Loading DeepFilterNet model...")
    try:
        load_deepfilternet_for_inference()
        print("Model loaded successfully - server ready!")
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


@app.websocket("/enhance")
async def websocket_enhance(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio enhancement.
    
    Expected flow:
    1. Client connects to /enhance (optionally with ?api_key=xxx)
    2. Client sends raw PCM audio frames as bytes (int16 format)
    3. Server converts to float32, processes with DeepFilterNet
    4. Server converts back to int16 and sends enhanced audio frames
    
    Audio format assumptions:
    - Input: int16 PCM bytes (little-endian)
    - Sample rate: 48kHz (or as required by model)
    - Channels: mono
    """
    # Optional API key authentication
    if API_KEY:
        # Check for API key in query parameters
        api_key = websocket.query_params.get("api_key")
        if api_key != API_KEY:
            await websocket.close(code=1008, reason="Invalid or missing API key")
            print("Client connection rejected: Invalid API key")
            return
    
    # Accept the WebSocket connection
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"Client connected from {client_host}")
    
    try:
        frame_count = 0
        
        while True:
            # Receive raw PCM audio data from client (as bytes)
            audio_bytes = await websocket.receive_bytes()
            frame_count += 1
            
            try:
                # Step 1: Convert bytes to int16 numpy array
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                
                # Step 2: Convert int16 [-32768, 32767] to float32 [-1.0, 1.0]
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                
                # Step 3: Convert numpy array to PyTorch tensor
                audio_tensor = torch.from_numpy(audio_float32)
                
                # Step 4: Enhance the audio using DeepFilterNet
                enhanced_tensor = enhance_block(audio_tensor)
                
                # Step 5: Convert back to numpy float32
                enhanced_float32 = enhanced_tensor.numpy()
                
                # Step 6: Clip to [-1.0, 1.0] range to prevent overflow
                enhanced_float32 = np.clip(enhanced_float32, -1.0, 1.0)
                
                # Step 7: Convert float32 [-1.0, 1.0] back to int16 [-32768, 32767]
                enhanced_int16 = (enhanced_float32 * 32767.0).astype(np.int16)
                
                # Step 8: Convert to bytes
                enhanced_bytes = enhanced_int16.tobytes()
                
                # Step 9: Send enhanced audio back to client
                await websocket.send_bytes(enhanced_bytes)
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                # Send original audio back on error
                await websocket.send_bytes(audio_bytes)
            
    except WebSocketDisconnect:
        print(f"Client from {client_host} disconnected (processed {frame_count} frames)")
    except Exception as e:
        print(f"WebSocket error from {client_host}: {e}")
    finally:
        # Ensure WebSocket is closed gracefully
        try:
            await websocket.close()
        except:
            pass  # Already closed
        print(f"WebSocket connection closed from {client_host}")


if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
