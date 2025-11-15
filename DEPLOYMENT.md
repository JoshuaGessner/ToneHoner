# Remote Deployment Guide

This guide explains how to deploy the ToneHoner server to make it accessible over the internet.

## Quick Start

### Using Ngrok (Easiest)

```powershell
# Terminal 1: Start the server
python main.py

# Terminal 2: Expose via ngrok
ngrok http 8000
```

Copy the ngrok URL (e.g., `https://abc123.ngrok.io`) and clients can connect to:
```
wss://abc123.ngrok.io/enhance
```

### Using Your Own Server

```powershell
# Start server (already binds to 0.0.0.0)
python main.py
```

Clients connect to:
```
ws://YOUR_PUBLIC_IP:8000/enhance
```

## Security Configuration

### Enable API Key Authentication

```powershell
# Set environment variable
$env:API_KEY = "your-secret-key-here"

# Start server
python main.py
```

Clients must connect with the API key:
```powershell
python client/client.py --server "ws://server:8000/enhance?api_key=your-secret-key-here"
```

Or in GUI mode, enter: `ws://server:8000/enhance?api_key=your-secret-key-here`

### For Production (HTTPS/WSS)

Use a reverse proxy like nginx or Caddy:

**Caddy (Automatic HTTPS):**
```bash
# Install Caddy
sudo apt install caddy

# Create Caddyfile
cat > Caddyfile << EOF
yourdomain.com {
    reverse_proxy /enhance localhost:8000
    reverse_proxy /ping localhost:8000
}
EOF

# Start Caddy
sudo caddy start
```

Clients connect via secure WebSocket:
```
wss://yourdomain.com/enhance
```

## Deployment Methods

### 1. Port Forwarding (Home/Office)

**Steps:**
1. Find your server's local IP: `ipconfig` (look for IPv4)
2. Log into your router's admin panel
3. Forward external port 8000 → internal IP:8000
4. Find your public IP: `curl ifconfig.me`
5. Test: `curl http://YOUR_PUBLIC_IP:8000/ping`

**Client connects to:**
```
ws://YOUR_PUBLIC_IP:8000/enhance
```

**Limitations:**
- Requires static IP or dynamic DNS
- Router configuration required
- Security concerns (exposed to internet)

### 2. Ngrok (Quick Development)

**Installation:**
```powershell
# Windows
choco install ngrok -y

# Or download from https://ngrok.com
```

**Usage:**
```powershell
# Start server
python main.py

# Expose via ngrok (new terminal)
ngrok http 8000
```

**Output:**
```
Forwarding   https://abc123.ngrok.io -> http://localhost:8000
```

**Client connects to:**
```
wss://abc123.ngrok.io/enhance
```

**Advantages:**
- No router configuration needed
- Automatic HTTPS/WSS
- Works behind firewalls

**Limitations:**
- URL changes each time (unless paid plan)
- Bandwidth limits on free tier
- Third-party dependency

### 3. Cloud VPS (AWS, Azure, DigitalOcean)

**Example: AWS EC2**

1. **Launch EC2 Instance:**
   - Ubuntu 22.04
   - Instance type: `g4dn.xlarge` (GPU) or `t3.medium` (CPU)
   - Security group: Allow TCP 8000 (or 80/443 for nginx)

2. **Install Dependencies:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers (if using GPU)
sudo apt install nvidia-driver-535 -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit (for GPU)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

3. **Deploy with Docker:**
```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/ToneHoner.git
cd ToneHoner

# Build image
docker build -t tonehoner-server:latest .

# Run container
docker run -d \
  --name tonehoner \
  --gpus all \
  -p 8000:8000 \
  --restart unless-stopped \
  -e API_KEY="your-secret-key" \
  tonehoner-server:latest
```

4. **Test:**
```bash
curl http://localhost:8000/ping
```

**Client connects to:**
```
ws://EC2_PUBLIC_IP:8000/enhance?api_key=your-secret-key
```

### 4. Docker + Nginx + SSL

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  tonehoner:
    build: .
    container_name: tonehoner-server
    environment:
      - API_KEY=${API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - tonehoner-net

  nginx:
    image: nginx:alpine
    container_name: tonehoner-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    networks:
      - tonehoner-net
    depends_on:
      - tonehoner

networks:
  tonehoner-net:
    driver: bridge
```

**nginx.conf:**
```nginx
events {
    worker_connections 1024;
}

http {
    upstream tonehoner {
        server tonehoner:8000;
    }

    map $http_upgrade $connection_upgrade {
        default upgrade;
        '' close;
    }

    server {
        listen 80;
        server_name yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        location /enhance {
            proxy_pass http://tonehoner;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection $connection_upgrade;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        location /ping {
            proxy_pass http://tonehoner;
        }
    }
}
```

**Deploy:**
```bash
docker-compose up -d
```

**Client connects to:**
```
wss://yourdomain.com/enhance?api_key=your-secret-key
```

### 5. Kubernetes (Production Scale)

Already configured in `k8s/deployment.yaml`.

**Deploy:**
```bash
# Update image in deployment.yaml
kubectl apply -f k8s/deployment.yaml

# Get service IP
kubectl get svc deepfilternet-server
```

## Firewall Configuration

### Windows Firewall
```powershell
# Allow incoming connections on port 8000
New-NetFirewallRule -DisplayName "ToneHoner Server" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

### Linux UFW
```bash
sudo ufw allow 8000/tcp
sudo ufw reload
```

### Cloud Providers
- **AWS**: Security Groups - Allow TCP 8000
- **Azure**: Network Security Groups - Allow TCP 8000
- **GCP**: Firewall Rules - Allow TCP 8000

## Client Configuration

### Command Line
```powershell
# Local server
python client/client.py --server ws://localhost:8000/enhance

# Remote server
python client/client.py --server ws://SERVER_IP:8000/enhance

# With API key
python client/client.py --server "ws://SERVER_IP:8000/enhance?api_key=SECRET"

# Secure WebSocket
python client/client.py --server wss://yourdomain.com/enhance
```

### GUI Mode
```powershell
python client/client.py --gui
```
Then enter the server URL in the text field:
- `ws://192.168.1.100:8000/enhance` (local network)
- `ws://YOUR_PUBLIC_IP:8000/enhance` (internet)
- `wss://yourdomain.com/enhance` (secure)

## Monitoring & Logs

### View Server Logs
```powershell
# Direct run
python main.py  # Logs print to console

# Docker
docker logs -f tonehoner

# Kubernetes
kubectl logs -f deployment/deepfilternet-server
```

### Health Check
```powershell
# PowerShell
Invoke-RestMethod http://SERVER_IP:8000/ping

# curl
curl http://SERVER_IP:8000/ping

# Expected: {"status":"ok"}
```

## Performance Considerations

### Latency Optimization
- Deploy server geographically close to clients
- Use direct connections instead of VPNs
- Reduce `BLOCK_SIZE` in client for lower latency (at cost of quality)

### Bandwidth Requirements
- 48kHz mono audio @ 16-bit = ~94 KB/s per direction
- For 10 concurrent clients: ~2 MB/s total

### Scaling
- Each client connection uses 1 thread
- GPU memory: ~2GB for model
- CPU fallback: slower but works
- For multiple clients, consider load balancing multiple instances

## Security Checklist

- [ ] Enable API key authentication (`API_KEY` env variable)
- [ ] Use HTTPS/WSS for production (nginx/Caddy)
- [ ] Implement rate limiting (see TODO in server code)
- [ ] Restrict CORS origins in production
- [ ] Use firewall to limit access
- [ ] Monitor logs for suspicious activity
- [ ] Keep dependencies updated
- [ ] Use strong, random API keys

## Troubleshooting

### "Connection refused"
- Check if server is running: `curl http://localhost:8000/ping`
- Verify firewall rules
- Check port forwarding configuration

### "WebSocket connection failed"
- Ensure using correct protocol: `ws://` (not `http://`)
- For SSL: use `wss://` (not `https://`)
- Check API key if enabled

### "High latency"
- Reduce `BLOCK_SIZE` in client
- Check network speed: `speedtest-cli`
- Use server with GPU for faster processing
- Deploy server closer to clients

### "Out of memory" (GPU)
- Reduce concurrent connections
- Use CPU mode if insufficient GPU memory
- Increase GPU resources in cloud

## Cost Estimates

### Ngrok
- Free tier: Adequate for testing
- Pro: $8/month for static domain

### AWS EC2
- t3.medium (CPU): ~$30/month
- g4dn.xlarge (GPU): ~$400/month
- Data transfer: ~$0.09/GB

### DigitalOcean
- Basic CPU: $6-12/month
- GPU: $90-180/month

### Self-hosted
- Hardware + electricity
- No bandwidth costs
- Requires maintenance
