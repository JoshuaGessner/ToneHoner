#!/bin/bash
# ToneHoner Server Startup Script

echo "============================================================"
echo "ToneHoner Audio Enhancement Server"
echo "============================================================"
echo ""

# Check if virtual environment exists
if [ ! -f "venv/bin/python" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run: python -m venv venv"
    echo "Then install dependencies: pip install -r requirements.txt"
    exit 1
fi

# Optional: Set API key for authentication
# Uncomment and set your own secret key:
# export API_KEY="your-secret-key-here"

# Optional: Set custom port
# export PORT=8000

# Get local IP
LOCAL_IP=$(hostname -I | awk '{print $1}')

# Run server
echo "Starting server..."
echo ""
echo "Server will be accessible at:"
echo "  - Local: http://localhost:8000"
echo "  - Network: http://$LOCAL_IP:8000"
echo ""
echo "WebSocket endpoint: /enhance"
echo "Health check: http://localhost:8000/ping"
echo ""
echo "Press Ctrl+C to stop the server"
echo "============================================================"
echo ""

venv/bin/python main.py
