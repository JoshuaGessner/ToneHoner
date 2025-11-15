# DeepFilterNet Server - Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying the DeepFilterNet audio enhancement server.

## Prerequisites

1. **Kubernetes cluster** with GPU support (NVIDIA GPU Operator or device plugin)
2. **kubectl** configured to access your cluster
3. **Docker image** of the DeepFilterNet server pushed to a registry
4. **Model files** available in persistent storage

## Quick Start

### Option 1: Direct kubectl Deployment

```bash
# Apply the deployment
kubectl apply -f k8s/deployment.yaml

# Check deployment status
kubectl get deployments deepfilternet-server
kubectl get pods -l app=deepfilternet-server

# Check service
kubectl get service deepfilternet-server

# Get NodePort URL
kubectl get nodes -o wide
# Access at: http://<NODE_IP>:30800
```

### Option 2: Helm Chart Deployment

```bash
# Create a Helm chart structure (if not already created)
helm create deepfilternet-server

# Copy values.yaml to the chart directory
cp k8s/values.yaml deepfilternet-server/values.yaml

# Install with Helm
helm install deepfilternet-server ./deepfilternet-server \
  --set image.repository=your-registry.example.com/deepfilternet-server \
  --set image.tag=latest

# Or install from values file
helm install deepfilternet-server ./deepfilternet-server \
  -f k8s/values.yaml

# Check status
helm status deepfilternet-server

# Upgrade
helm upgrade deepfilternet-server ./deepfilternet-server \
  -f k8s/values.yaml

# Uninstall
helm uninstall deepfilternet-server
```

## Configuration

### Building and Pushing Docker Image

```bash
# Build the Docker image
docker build -t your-registry.example.com/deepfilternet-server:v1.0.0 .

# Push to registry
docker push your-registry.example.com/deepfilternet-server:v1.0.0
```

### Setting up Model Storage

The deployment expects models to be available in a Persistent Volume. You have several options:

**Option 1: Pre-populate PVC**

```bash
# Create a temporary pod to copy models
kubectl run model-loader --image=busybox --restart=Never \
  --overrides='
{
  "spec": {
    "containers": [{
      "name": "model-loader",
      "image": "busybox",
      "command": ["sleep", "3600"],
      "volumeMounts": [{
        "name": "models",
        "mountPath": "/models"
      }]
    }],
    "volumes": [{
      "name": "models",
      "persistentVolumeClaim": {
        "claimName": "deepfilternet-models-pvc"
      }
    }]
  }
}'

# Copy models to the pod
kubectl cp models/model_ts.pt model-loader:/models/

# Verify
kubectl exec model-loader -- ls -la /models/

# Clean up
kubectl delete pod model-loader
```

**Option 2: Build Models into Docker Image**

Modify the Dockerfile to include model export:

```dockerfile
# Add to Dockerfile
RUN python dfn_server.py
```

Then remove the volume mount from deployment.yaml.

**Option 3: Use InitContainer**

Add an init container to download/prepare models at startup.

### GPU Node Setup

Ensure your cluster has GPU support:

```bash
# Check GPU nodes
kubectl get nodes -o json | jq '.items[].status.allocatable."nvidia.com/gpu"'

# Install NVIDIA GPU Operator (if not already installed)
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator-resources \
  --create-namespace

# Verify GPU operator
kubectl get pods -n gpu-operator-resources
```

### Adjusting Node Selectors

Update the `nodeSelector` in deployment.yaml to match your cluster's labels:

```yaml
nodeSelector:
  node.kubernetes.io/instance-type: g4dn.xlarge  # AWS
  # cloud.google.com/gke-accelerator: nvidia-tesla-t4  # GCP
  # accelerator: nvidia-gpu  # Generic
```

## Accessing the Service

### NodePort Service

```bash
# Get node IP and port
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}')
NODE_PORT=$(kubectl get svc deepfilternet-server -o jsonpath='{.spec.ports[0].nodePort}')

echo "Service available at: ws://${NODE_IP}:${NODE_PORT}/enhance"

# Test health check
curl http://${NODE_IP}:${NODE_PORT}/ping
```

### LoadBalancer Service (Cloud)

Change service type in deployment.yaml or values.yaml:

```yaml
service:
  type: LoadBalancer
  port: 8000
```

```bash
# Get LoadBalancer IP
kubectl get svc deepfilternet-server

# Access at the EXTERNAL-IP
```

### Ingress (Recommended for Production)

Enable ingress in values.yaml:

```yaml
ingress:
  enabled: true
  className: "nginx"
  annotations:
    nginx.ingress.kubernetes.io/websocket-services: "deepfilternet-server"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: deepfilternet.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: deepfilternet-tls
      hosts:
        - deepfilternet.yourdomain.com
```

## Monitoring

### Logs

```bash
# View logs
kubectl logs -f deployment/deepfilternet-server

# View logs for specific pod
kubectl logs -f <pod-name>

# View logs with timestamps
kubectl logs --timestamps deployment/deepfilternet-server
```

### Resource Usage

```bash
# Check resource usage
kubectl top pod -l app=deepfilternet-server

# Check GPU usage (if metrics available)
kubectl exec <pod-name> -- nvidia-smi
```

### Health Checks

```bash
# Check readiness
kubectl get pods -l app=deepfilternet-server

# Manual health check
kubectl exec <pod-name> -- curl -s http://localhost:8000/ping
```

## Scaling

### Manual Scaling

```bash
# Scale replicas (note: multiple GPUs required)
kubectl scale deployment deepfilternet-server --replicas=3

# Check status
kubectl get deployment deepfilternet-server
```

### Horizontal Pod Autoscaling (HPA)

Not recommended for GPU workloads, but available:

```yaml
autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 3
  targetCPUUtilizationPercentage: 80
```

## Troubleshooting

### Pod Not Starting

```bash
# Describe pod
kubectl describe pod -l app=deepfilternet-server

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp

# Common issues:
# - Image pull errors: Check image name and registry credentials
# - GPU not available: Verify GPU nodes and operator
# - Model not found: Check PVC and model files
```

### GPU Not Detected

```bash
# Check GPU resources
kubectl describe node <gpu-node-name> | grep nvidia.com/gpu

# Check pod GPU allocation
kubectl describe pod <pod-name> | grep nvidia.com/gpu

# Exec into pod and check
kubectl exec <pod-name> -- nvidia-smi
```

### Service Not Accessible

```bash
# Check service
kubectl get svc deepfilternet-server

# Check endpoints
kubectl get endpoints deepfilternet-server

# Test from within cluster
kubectl run test-pod --image=curlimages/curl -it --rm -- \
  curl -s http://deepfilternet-server:8000/ping
```

### Readiness Probe Failing

```bash
# Check probe configuration
kubectl describe pod <pod-name> | grep -A 10 Readiness

# Test manually
kubectl exec <pod-name> -- curl -v http://localhost:8000/ping

# Adjust timing in deployment.yaml if needed:
readinessProbe:
  initialDelaySeconds: 30  # Increase if model loading is slow
  periodSeconds: 10
```

## Production Considerations

### Security

1. **Use private registry** with image pull secrets:
   ```bash
   kubectl create secret docker-registry regcred \
     --docker-server=your-registry.example.com \
     --docker-username=user \
     --docker-password=pass
   ```

2. **Enable network policies** to restrict traffic
3. **Use RBAC** for service accounts
4. **Scan images** for vulnerabilities

### High Availability

1. **Multiple replicas** across availability zones
2. **Pod disruption budgets** to maintain availability during updates
3. **Resource limits** to prevent resource exhaustion
4. **Liveness probes** to restart unhealthy pods

### Cost Optimization

1. **Use spot instances** for GPU nodes (with appropriate tolerations)
2. **Node autoscaling** to scale GPU nodes based on demand
3. **Preemptible pods** for non-critical workloads

### Monitoring and Observability

1. **Prometheus metrics** for resource usage
2. **Grafana dashboards** for visualization
3. **Distributed tracing** for request flow
4. **Log aggregation** (ELK, Loki, etc.)

## Clean Up

```bash
# Delete deployment
kubectl delete -f k8s/deployment.yaml

# Or with Helm
helm uninstall deepfilternet-server

# Delete PVC (if no longer needed)
kubectl delete pvc deepfilternet-models-pvc
```

## Advanced Configuration

### Using ConfigMap for Runtime Configuration

```yaml
env:
- name: SERVER_PORT
  valueFrom:
    configMapKeyRef:
      name: deepfilternet-config
      key: server_port
```

### Using Secrets for Sensitive Data

```bash
# Create secret
kubectl create secret generic deepfilternet-secrets \
  --from-literal=api-key=your-secret-key

# Reference in deployment
env:
- name: API_KEY
  valueFrom:
    secretKeyRef:
      name: deepfilternet-secrets
      key: api-key
```

### Multi-region Deployment

Deploy to multiple clusters and use a global load balancer for routing.

## References

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)
- [Helm Documentation](https://helm.sh/docs/)
