# Use NVIDIA PyTorch runtime image with CUDA support (latest stable)
FROM nvcr.io/nvidia/pytorch:24.11-py3

# Set working directory in container
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code and model files
COPY dfn_server.py .
COPY main.py .
COPY models/ ./models/

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Run the FastAPI server with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
