@echo off
REM ToneHoner Server Startup Script

echo ============================================================
echo ToneHoner Audio Enhancement Server
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then install dependencies: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Optional: Set API key for authentication
REM Uncomment and set your own secret key:
REM set API_KEY=your-secret-key-here

REM Optional: Set custom port
REM set PORT=8000

REM Activate virtual environment and run server
echo Starting server...
echo.
echo Server will be accessible at:
echo   - Local: http://localhost:8000
echo   - Network: http://YOUR_IP:8000
echo.
echo WebSocket endpoint: /enhance
echo Health check: http://localhost:8000/ping
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

venv\Scripts\python.exe main.py

pause
