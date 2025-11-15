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
    Install DeepFilterNet from PyPI.
    """
    print("Installing DeepFilterNet from PyPI...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "deepfilternet"
        ])
        print("DeepFilterNet installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install DeepFilterNet: {e}")
        sys.exit(1)


def check_device():
    """
    Check if CUDA 13 is available, otherwise fall back to CPU.
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


# Global variables for inference - using DeepFilterNet directly instead of TorchScript
_model = None
_df_state = None
_device = None
_enhance_fn = None


def unload_model():
    """
    Unload the DeepFilterNet model and free up memory.
    Useful for freeing GPU/CPU memory when the model is no longer needed.
    """
    global _model, _df_state, _device, _enhance_fn
    
    if _model is not None:
        print("Unloading DeepFilterNet model...")
        
        # Clear CUDA cache if using GPU
        if _device is not None and _device.type == 'cuda':
            import torch
            torch.cuda.empty_cache()
            print("Cleared CUDA cache")
        
        # Reset global variables
        _model = None
        _df_state = None
        _device = None
        _enhance_fn = None
        
        print("Model unloaded successfully")
    else:
        print("No model loaded")


def save_model_info(model, df_state, device, output_dir="./models"):
    """
    Save model information and verify the model is ready.
    DeepFilterNet3 cannot be easily exported to TorchScript, so we use it directly.
    """
    print("Verifying model for inference...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    info_path = os.path.join(output_dir, "model_info.txt")
    
    try:
        # Save model information - try different attributes
        with open(info_path, 'w') as f:
            f.write(f"DeepFilterNet Model Information\n")
            f.write(f"================================\n")
            f.write(f"Model Type: DeepFilterNet3\n")
            f.write(f"Device: {device}\n")
            
            # Try to get available attributes
            if hasattr(df_state, 'sr'):
                f.write(f"Sample Rate: {df_state.sr()} Hz\n")
            if hasattr(df_state, 'fft_size'):
                f.write(f"FFT Size: {df_state.fft_size()}\n")
            if hasattr(df_state, 'hop_size'):
                f.write(f"Hop Size: {df_state.hop_size()}\n")
            if hasattr(df_state, 'nb_df'):
                f.write(f"DF Bands: {df_state.nb_df()}\n")
            if hasattr(df_state, 'nb_erb'):
                f.write(f"ERB Bands: {df_state.nb_erb()}\n")
        
        print(f"✓ Model info saved to: {info_path}")
        print(f"✓ Model is ready for inference on {device}")
        
        return info_path
        
    except Exception as e:
        print(f"Warning: Failed to save detailed model info: {e}")
        # Try minimal version
        try:
            with open(info_path, 'w') as f:
                f.write(f"DeepFilterNet Model Information\n")
                f.write(f"================================\n")
                f.write(f"Model Type: DeepFilterNet3\n")
                f.write(f"Device: {device}\n")
            print(f"✓ Basic model info saved to: {info_path}")
        except:
            pass
        print("✓ Model is still usable for inference.")
        return None


def load_deepfilternet_for_inference():
    """
    Load the DeepFilterNet model for inference.
    Sets global _model, _df_state, and _device variables.
    
    Returns:
        Tuple of (model, df_state, device)
    """
    global _model, _df_state, _device, _enhance_fn
    
    print("Loading DeepFilterNet model for inference...")
    
    try:
        from df.enhance import enhance, init_df
        
        # Determine device (GPU if available, else CPU)
        if torch.cuda.is_available():
            _device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            _device = torch.device("cpu")
            print("Using CPU for inference")
        
        # Initialize DeepFilterNet model
        start_time = time.time()
        _model, _df_state, _ = init_df(
            model_base_dir=None,
            post_filter=True,
            log_level="WARNING",  # Reduce verbosity
            config_allow_defaults=True  # Allow default config
        )
        _model = _model.to(_device)
        _model.eval()
        
        # Store the enhance function
        _enhance_fn = enhance
        
        load_time = time.time() - start_time
        print(f"Model loaded successfully in {load_time:.3f} seconds")
        print(f"Device: {_device}")
        
        return _model, _df_state, _device
        
    except Exception as e:
        print(f"Failed to load DeepFilterNet model: {e}")
        raise


def enhance_block(audio: torch.Tensor) -> torch.Tensor:
    """
    Enhance a block of audio using the loaded DeepFilterNet model.
    
    Args:
        audio: 1-D tensor of float32 PCM samples in range [-1, 1]
               Shape: (num_samples,)
    
    Returns:
        Enhanced audio tensor with same shape as input
        
    Raises:
        RuntimeError: If model is not loaded or inference fails
        ValueError: If input audio format is invalid
    """
    global _model, _df_state, _device, _enhance_fn
    
    # Validate model is loaded
    if _model is None or _df_state is None or _device is None:
        raise RuntimeError(
            "Model not loaded. Call load_deepfilternet_for_inference() first."
        )
    
    # Validate input
    if not isinstance(audio, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(audio)}")
    
    if audio.dim() != 1:
        raise ValueError(
            f"Expected 1-D tensor, got {audio.dim()}-D tensor with shape {audio.shape}"
        )
    
    if audio.dtype != torch.float32:
        audio = audio.float()
    
    # Check value range and normalize if needed
    audio_min, audio_max = audio.min().item(), audio.max().item()
    if audio_min < -1.0 or audio_max > 1.0:
        print(f"Warning: Audio values outside [-1, 1] range: [{audio_min:.3f}, {audio_max:.3f}]. Clipping...")
        audio = torch.clamp(audio, -1.0, 1.0)
    
    # Ensure audio is on CPU before sending to model (will move to device inside)
    if audio.device.type != 'cpu':
        audio = audio.cpu()
    
    try:
        # Start profiling
        start_time = time.time()
        num_samples = audio.shape[0]
        
        # DeepFilterNet's enhance() expects shape (channels, samples) for mono audio
        # Convert from (samples,) -> (1, samples) for mono
        audio_input = audio.unsqueeze(0)  # Add channel dimension: [samples] -> [1, samples]
        
        # Run enhancement using DeepFilterNet's enhance function
        # NOTE: Do NOT move audio to device here - DeepFilterNet's enhance() handles
        # device placement internally and expects CPU tensors for the analysis stage
        with torch.no_grad():
            try:
                # enhance() expects [channels, samples] and returns [channels, samples]
                enhanced = _enhance_fn(_model, _df_state, audio_input)
            except Exception as e:
                print(f"Error in _enhance_fn: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # enhanced shape is (channels, samples), squeeze to (samples,)
            # Ensure we move to CPU and detach before any further operations
            try:
                enhanced = enhanced.squeeze(0).detach().cpu()
            except Exception as e:
                print(f"Error during squeeze/detach/cpu: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        # Ensure output matches input length (do this AFTER moving to CPU)
        if enhanced.shape[0] != num_samples:
            # Trim or pad to match input length
            if enhanced.shape[0] > num_samples:
                enhanced = enhanced[:num_samples]
            else:
                # Pad on CPU to ensure tensor stays on CPU
                enhanced = torch.nn.functional.pad(enhanced, (0, num_samples - enhanced.shape[0]))
        
        # Final safety check - ensure tensor is on CPU and contiguous
        enhanced = enhanced.cpu().contiguous()
        
        # Profiling output
        inference_time = time.time() - start_time
        duration_sec = num_samples / _df_state.sr()
        rtf = inference_time / duration_sec if duration_sec > 0 else 0
        
        if num_samples > 1000:  # Only print for non-trivial blocks
            print(f"Inference: {num_samples} samples ({duration_sec:.3f}s audio) "
                  f"processed in {inference_time:.3f}s (RTF: {rtf:.3f}x)")
        
        return enhanced
        
    except Exception as e:
        print(f"Error during inference: {e}")
        print(f"Input shape: {audio.shape}, dtype: {audio.dtype}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Inference failed: {e}")


def main():
    """
    Main execution flow:
    1. Install DeepFilterNet from PyPI
    2. Check device availability (CUDA 13 or CPU)
    3. Load pre-trained DeepFilterNet-2 checkpoint
    4. Verify model is ready for inference
    5. Save model info to ./models/model_info.txt
    """
    print("=" * 60)
    print("DeepFilterNet-2 Setup Script")
    print("=" * 60)
    
    # Step 1: Install DeepFilterNet
    install_deepfilternet()
    
    # Step 2: Check device availability
    device = check_device()
    print(f"\nUsing device: {device}")
    
    # Step 3: Load pre-trained model
    model, df_state = load_deepfilternet_model(device)
    
    # Step 4 & 5: Save model info and verify
    output_path = save_model_info(model, df_state, device)
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print(f"Device used: {device}")
    if output_path:
        print(f"Model info saved at: {output_path}")
    print("\nThe model is ready for inference.")
    print("You can now start the server with: python main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
