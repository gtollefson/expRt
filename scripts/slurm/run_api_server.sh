#!/bin/bash
#SBATCH --job-name=r-assistant-api
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/api_%j.out
#SBATCH --error=logs/api_%j.err
#SBATCH --partition=gpu  # Use GPU partition if available
#SBATCH --gres=gpu:1     # Request 1 GPU (optional)

# R Package Assistant - API Server
echo "🚀 Starting R Package Assistant API Server on HPCC"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"

# Get node information
NODE_IP=$(hostname -I | awk '{print $1}')
PORT=8000

echo "Server will be available at: http://$NODE_IP:$PORT"
echo "Node hostname: $(hostname)"

# Load environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export API_KEY="hpcc-production-key-$(date +%s)"
export VECTOR_STORE_PATH="data/vectorstore"
export MODEL_CACHE_DIR="models"

# Create API configuration
cat > config/api_config.json << EOF
{
  "host": "0.0.0.0",
  "port": $PORT,
  "workers": 1,
  "log_level": "info",
  "vector_store_path": "$VECTOR_STORE_PATH",
  "model_cache_dir": "$MODEL_CACHE_DIR",
  "embedding_model": "all-MiniLM-L6-v2",
  "llm_model": "microsoft/Phi-3-mini-4k-instruct",
  "max_concurrent_requests": 10,
  "request_timeout": 300,
  "api_key": "$API_KEY"
}
EOF

# Create startup script
cat > scripts/start_server.py << 'EOF'
#!/usr/bin/env python3
"""Start the R Package Assistant API server."""

import os
import sys
import json
import uvicorn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.server import app

def main():
    """Start the API server with production configuration."""
    
    # Load configuration
    config_path = "config/api_config.json"
    if Path(config_path).exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "log_level": "info"
        }
    
    print(f"🚀 Starting API server with configuration:")
    for key, value in config.items():
        if key != "api_key":  # Don't print API key
            print(f"  {key}: {value}")
    
    # Set environment variables
    os.environ["API_KEY"] = config.get("api_key", "default-key")
    os.environ["VECTOR_STORE_PATH"] = config.get("vector_store_path", "data/vectorstore")
    
    # Start server
    uvicorn.run(
        "api.server:app",
        host=config["host"],
        port=config["port"],
        workers=config.get("workers", 1),
        log_level=config.get("log_level", "info"),
        timeout_keep_alive=config.get("request_timeout", 300)
    )

if __name__ == "__main__":
    main()
EOF

# Make script executable
chmod +x scripts/start_server.py

# Create directories
mkdir -p config logs

# Display connection information
echo ""
echo "========================================="
echo "🌐 R Package Assistant API Server"
echo "========================================="
echo "Node: $SLURMD_NODENAME"
echo "IP Address: $NODE_IP"
echo "Port: $PORT"
echo "API Key: $API_KEY"
echo ""
echo "Connect from outside HPCC:"
echo "1. Set up SSH tunnel:"
echo "   ssh -L $PORT:$NODE_IP:$PORT your_username@hpcc_login_node"
echo ""
echo "2. Access API at:"
echo "   http://localhost:$PORT"
echo "   http://localhost:$PORT/docs (API documentation)"
echo ""
echo "3. Health check:"
echo "   curl -H \"Authorization: Bearer $API_KEY\" http://localhost:$PORT/health"
echo ""
echo "========================================="

# Start the server
echo "Starting API server..."
python scripts/start_server.py

# Cleanup on exit
trap 'echo "Shutting down API server..."; kill $!' EXIT

# Keep script running
wait 