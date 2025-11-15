"""
DeepFilterNet-2 TorchScript Export Script
Installs DeepFilterNet, loads the pre-trained model, and exports to TorchScript format.
Also provides inference functionality for audio enhancement.
"""

import os
import sys
import subprocess
import time
import torch


def install_deepfilternet():
    """
    Install DeepFilterNet from GitHub repository.
    """
    print("Installing DeepFilterNet from GitHub...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/Rikorose/DeepFilterNet.git"
        ])
        print("DeepFilterNet installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install DeepFilterNet: {e}")
        sys.exit(1)


def check_device():
    """
    Check if CUDA 12 is available, otherwise fall back to CPU.
    Returns the device to use for model inference.
    """
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        device = torch.device("cuda")
        print(f"CUDA is available! Version: {cuda_version}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        return device
    else:
        device = torch.device("cpu")
        print("CUDA not available. Falling back to CPU.")
        return device


def load_deepfilternet_model(device):
    """
    Load the pre-trained DeepFilterNet-2 checkpoint.
    Returns the loaded model ready for inference.
    """
    print("Loading DeepFilterNet-2 pre-trained model...")
    
    try:
        from df.enhance import enhance, init_df
        from df.io import resample
        
        # Initialize DeepFilterNet-2 model with pre-trained weights
        model, df_state, _ = init_df(
            model_base_dir=None,  # Uses default pre-trained models
            post_filter=True,
            log_level="INFO"
        )
        
        # Move model to the specified device (GPU or CPU)
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        print(f"DeepFilterNet-2 loaded successfully on {device}")
        return model, df_state
        
    except ImportError as e:
        print(f"Failed to import DeepFilterNet modules: {e}")
        print("Make sure DeepFilterNet is installed correctly.")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)


def export_to_torchscript(model, device, output_dir="./models"):
    """
    Export the DeepFilterNet model to TorchScript format.
    Saves the model as 'model_ts.pt' in the specified directory.
    """
    print("Exporting model to TorchScript format...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_ts.pt")
    
    try:
        # Create a dummy input tensor for tracing
        # DeepFilterNet typically expects audio input with shape (batch, channels, time)
        # Using a sample input size for demonstration
        dummy_input = torch.randn(1, 1, 48000).to(device)  # 1 second at 48kHz
        
        # Trace the model with the dummy input
        print("Tracing model with TorchScript...")
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Save the traced model
        traced_model.save(output_path)
        print(f"TorchScript model saved successfully to: {output_path}")
        
        # Verify the saved model
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Model file size: {file_size_mb:.2f} MB")
        
        return output_path
        
    except Exception as e:
        print(f"Failed to export model to TorchScript: {e}")
        print("\nNote: DeepFilterNet may have complex operations that are difficult to trace.")
        print("Consider using torch.jit.script() or implementing a custom export wrapper.")
        sys.exit(1)


# Global variables for inference
_model = None
_device = None


def load_torchscript_model(model_path="./models/model_ts.pt"):
    """
    Load the TorchScript model at startup for inference.
    Sets global _model and _device variables.
    
    Args:
        model_path: Path to the TorchScript model file
        
    Returns:
        Tuple of (model, device)
    """
    global _model, _device
    
    print("Loading TorchScript model for inference...")
    
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Determine device (GPU if available, else CPU)
        if torch.cuda.is_available():
            _device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            _device = torch.device("cpu")
            print("Using CPU for inference")
        
        # Load the TorchScript model
        start_time = time.time()
        _model = torch.jit.load(model_path, map_location=_device)
        _model.eval()  # Set to evaluation mode
        load_time = time.time() - start_time
        
        print(f"Model loaded successfully in {load_time:.3f} seconds")
        print(f"Device: {_device}")
        
        return _model, _device
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the export script first to generate the model.")
        raise
    except Exception as e:
        print(f"Failed to load TorchScript model: {e}")
        raise


def enhance_block(audio: torch.Tensor) -> torch.Tensor:
    """
    Enhance a block of audio using the loaded TorchScript model.
    
    Args:
        audio: 1-D tensor of float32 PCM samples in range [-1, 1]
               Shape: (num_samples,)
    
    Returns:
        Enhanced audio tensor with same shape as input
        
    Raises:
        RuntimeError: If model is not loaded or inference fails
        ValueError: If input audio format is invalid
    """
    global _model, _device
    
    # Validate model is loaded
    if _model is None or _device is None:
        raise RuntimeError(
            "Model not loaded. Call load_torchscript_model() first."
        )
    
    # Validate input
    if not isinstance(audio, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(audio)}")
    
    if audio.dim() != 1:
        raise ValueError(
            f"Expected 1-D tensor, got {audio.dim()}-D tensor with shape {audio.shape}"
        )
    
    if audio.dtype != torch.float32:
        print(f"Warning: Converting audio from {audio.dtype} to float32")
        audio = audio.float()
    
    # Check value range
    audio_min, audio_max = audio.min().item(), audio.max().item()
    if audio_min < -1.0 or audio_max > 1.0:
        print(f"Warning: Audio values outside [-1, 1] range: [{audio_min:.3f}, {audio_max:.3f}]")
    
    try:
        # Start profiling
        start_time = time.time()
        num_samples = audio.shape[0]
        
        # Prepare input: add batch and channel dimensions
        # Shape: (num_samples,) -> (1, 1, num_samples)
        audio_input = audio.unsqueeze(0).unsqueeze(0).to(_device)
        
        # Run inference
        with torch.no_grad():
            enhanced = _model(audio_input)
        
        # Remove batch and channel dimensions
        # Shape: (1, 1, num_samples) -> (num_samples,)
        enhanced = enhanced.squeeze(0).squeeze(0).cpu()
        
        # Profiling output
        inference_time = time.time() - start_time
        duration_sec = num_samples / 48000.0  # Assuming 48kHz
        rtf = inference_time / duration_sec if duration_sec > 0 else 0
        
        print(f"Inference: {num_samples} samples ({duration_sec:.3f}s audio) "
              f"processed in {inference_time:.3f}s (RTF: {rtf:.3f}x)")
        
        return enhanced
        
    except Exception as e:
        print(f"Error during inference: {e}")
        print(f"Input shape: {audio.shape}, dtype: {audio.dtype}, device: {audio.device}")
        raise RuntimeError(f"Inference failed: {e}")


def main():
    """
    Main execution flow:
    1. Install DeepFilterNet from GitHub
    2. Check device availability (CUDA 12 or CPU)
    3. Load pre-trained DeepFilterNet-2 checkpoint
    4. Export model to TorchScript
    5. Save to ./models/model_ts.pt
    """
    print("=" * 60)
    print("DeepFilterNet-2 TorchScript Export Script")
    print("=" * 60)
    
    # Step 1: Install DeepFilterNet
    install_deepfilternet()
    
    # Step 2: Check device availability
    device = check_device()
    print(f"\nUsing device: {device}")
    
    # Step 3: Load pre-trained model
    model, df_state = load_deepfilternet_model(device)
    
    # Step 4 & 5: Export to TorchScript and save
    output_path = export_to_torchscript(model, device)
    
    print("\n" + "=" * 60)
    print("Export completed successfully!")
    print(f"Device used: {device}")
    print(f"Model saved at: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
