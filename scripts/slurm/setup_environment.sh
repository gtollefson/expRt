#!/bin/bash
#SBATCH --job-name=r-assistant-setup
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

# R Package Assistant - Environment Setup
echo "🚀 Setting up R Package Assistant environment on HPCC"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"

# Create necessary directories
mkdir -p logs data/raw data/processed data/vectorstore models

# Load modules (adjust for your HPCC system)
module purge
module load Python/3.9.6-GCCcore-11.2.0
module load GCC/11.2.0
module load CUDA/11.7.0  # Optional, for GPU support

# Create virtual environment
echo "Creating Python virtual environment..."
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU or GPU version)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
else
    echo "No GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install core dependencies
echo "Installing core dependencies..."
pip install sentence-transformers
pip install faiss-cpu  # Use faiss-gpu if CUDA is available
pip install transformers
pip install fastapi uvicorn
pip install numpy pandas scikit-learn
pip install tiktoken
pip install langchain llama-index

# Install development dependencies
pip install pytest black flake8 pre-commit

# Install additional utilities
pip install tqdm python-dotenv pyyaml requests httpx

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import sentence_transformers; print(f'SentenceTransformers: {sentence_transformers.__version__}')"
python -c "import faiss; print(f'FAISS: {faiss.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Save requirements for reproducibility
pip freeze > requirements_hpcc.txt

echo "✅ Environment setup complete!"
echo "Virtual environment: $(pwd)/venv"
echo "To activate: source venv/bin/activate" 