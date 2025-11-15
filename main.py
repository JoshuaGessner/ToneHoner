"""
FastAPI WebSocket server for DeepFilterNet audio enhancement.
This is a minimal skeleton with health check and WebSocket placeholder.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import torch

# Import the enhancement function from dfn_server
from dfn_server import enhance_block, load_torchscript_model

# Initialize FastAPI application
app = FastAPI(title="DeepFilterNet Audio Enhancement API")


@app.on_event("startup")
async def startup_event():
    """
    Load the TorchScript model at server startup.
    This ensures the model is ready before accepting WebSocket connections.
    """
    print("Loading DeepFilterNet model...")
    try:
        load_torchscript_model("./models/model_ts.pt")
        print("Model loaded successfully - server ready!")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        print("Server will start but enhancement will not work.")


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
    1. Client connects to /enhance
    2. Client sends raw PCM audio frames as bytes (int16 format)
    3. Server converts to float32, processes with DeepFilterNet
    4. Server converts back to int16 and sends enhanced audio frames
    
    Audio format assumptions:
    - Input: int16 PCM bytes (little-endian)
    - Sample rate: 48kHz (or as required by model)
    - Channels: mono
    """
    # Accept the WebSocket connection
    await websocket.accept()
    print("Client connected to /enhance")
    
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
        print(f"Client disconnected from /enhance (processed {frame_count} frames)")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Ensure WebSocket is closed gracefully
        try:
            await websocket.close()
        except:
            pass  # Already closed
        print("WebSocket connection closed")


if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
