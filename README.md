# 🚀 Local R Package Assistant Powered by LLMs

A cutting-edge, personal LLM-based assistant for R package development, CRAN maintenance, and reproducibility-focused projects.

## 📊 Current Development Status & Immediate Actions

### **Sprint Status: Foundation Architecture Complete** 
*Ready for HPCC deployment and dependency installation*

> **🤖 AI Assistant Context Instructions**: 
> - **Always read the complete roadmap section below** before making any development decisions
> - **Follow git best practices** with feature branches, descriptive commits, and automated workflows
> - **Use this status section** to understand current progress and immediate priorities
> - **Target system specs**: 32GB RAM, 8 cores, Linux HPCC environment

#### ✅ **Completed Components**
- **Core Architecture**: Complete project structure with 5 main modules
- **Text Chunking Pipeline**: Advanced chunker with token counting, sliding windows, R-specific metadata extraction
- **Vector Store Framework**: FAISS-based hierarchical indexing with metadata filtering
- **Embedding Pipeline**: Context-aware embedding generation with R-specific prefixes
- **API Server**: FastAPI backend with authentication, health checks, and mock responses
- **Frontend Interface**: Complete Shiny dashboard with query interface and statistics
- **HPCC Integration**: SLURM scripts for environment setup, batch processing, and server deployment
- **Containerization**: Docker setup for local development and testing

#### 🎯 **Immediate Next Actions** (Priority Order)
1. **Git Repository Setup**: Initialize repo, create dev branches, configure automation
2. **Transfer to HPCC**: `scp -r` entire project to HPCC home directory
3. **Environment Setup**: Submit `sbatch scripts/slurm/setup_environment.sh` 
4. **Install Dependencies**: Run setup script to install PyTorch, sentence-transformers, FAISS, etc.
5. **Test Core Pipeline**: Execute `python scripts/test_pipeline.py` to verify chunking works
6. **Create Test Data**: Generate sample R package documentation for initial testing
7. **Build Vector Store**: Process test packages and create initial FAISS index
8. **Deploy API Server**: Submit `sbatch scripts/slurm/run_api_server.sh`
9. **Connect Frontend**: Configure Shiny app to communicate with HPCC API

#### 🔧 **Git Workflow & Development Rules**
- **Main Branch**: `main` - production-ready code only
- **Development Branch**: `develop` - integration branch for features
- **Feature Branches**: `feature/phase-X-component-name` - specific development tasks
- **Commit Convention**: `type(scope): description` (e.g., `feat(chunking): add R-specific metadata extraction`)
- **Auto-commit Rule**: Every significant change gets committed with descriptive messages
- **Push Frequency**: Push to remote after each logical feature completion
- **Branch Strategy**: Create new feature branch for each roadmap phase/component

#### 🔧 **Technical Implementation Notes**
- **Memory Requirements**: 16-32GB RAM for full deployment (32GB recommended)
- **Model Downloads**: ~2-7GB for embedding + LLM models
- **API Authentication**: Uses bearer token authentication (dev key: `dev-key-123`)
- **Port Configuration**: API server runs on port 8000, requires SSH tunneling for external access
- **Batch Processing**: SLURM array jobs configured for parallel R package processing

#### 🚨 **Known Dependencies to Install**
```bash
pip install torch sentence-transformers faiss-cpu transformers
pip install fastapi uvicorn tiktoken langchain llama-index
```

#### 📁 **Project Structure Ready**
```
r-package-assistant/
├── src/chunking/          # Text processing & R package parsing
├── src/embeddings/        # Vector generation & FAISS store  
├── src/api/              # FastAPI server backend
├── frontend/shiny/       # R Shiny dashboard
├── scripts/slurm/        # HPCC deployment scripts
├── data/                 # Raw data, processed chunks, vector stores
└── requirements.txt      # Python dependencies
```

## 🎯 Project Goals

- Build a version-aware R package documentation assistant
- Deploy locally with lightweight LLMs (Phi-3, Mistral 7B)
- Provide fast, accurate responses for R developers
- Support CRAN maintainers and reproducibility workflows
- Create a sustainable alternative to expensive GPT-powered copilots

## 📋 Development Roadmap

### ✅ Phase 1: Foundation Build (Weeks 1–2)
- [ ] Core token counting and file chunking script
- [ ] Dynamic chunk sizing with sliding window overlap
- [ ] Document ingestion pipeline with metadata capture
- [ ] FAISS-based vector store implementation
- [ ] Simple retrieval testing script

### ⚡ Phase 2: Backend LLM and Retrieval System (Weeks 3–4)
- [ ] Local model deployment (Phi-3/Mistral 7B on HPCC)
- [ ] CPU vs GPU inference benchmarking
- [ ] End-to-end retrieval pipeline integration
- [ ] Metadata-aware retrieval enhancement

### 🔍 Phase 3: Query Refinement & Quality Boosters (Weeks 5–6)
- [ ] Hierarchical indexing implementation
- [ ] Context-aware retrieval mechanism
- [ ] User query pre-processing (functions, packages, versions)
- [ ] Cross-encoder re-ranking system
- [ ] Optional LLM fine-tuning on R documentation

### 🌐 Phase 4: Frontend Integration (Weeks 7–8)
- [ ] Shiny app deployment on shinyapps.io
- [ ] Flask/FastAPI server on HPCC
- [ ] API gateway with authentication
- [ ] Rate limiting and security measures

### 🛠️ Phase 5: Finalization and Optimization (Weeks 9–10)
- [ ] Local caching implementation
- [ ] Performance logging and analysis
- [ ] Docker containerization
- [ ] Documentation and deployment guides

## 🏗️ Project Structure

```
r-package-assistant/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── src/
│   ├── chunking/
│   ├── embeddings/
│   ├── retrieval/
│   ├── llm/
│   └── api/
├── data/
│   ├── raw/
│   ├── processed/
│   └── vectorstore/
├── tests/
├── scripts/
│   ├── slurm/
│   └── deployment/
├── frontend/
│   └── shiny/
└── docs/
```

## 🚀 Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the test pipeline: `python scripts/test_pipeline.py`
4. Launch the Shiny app: `Rscript frontend/shiny/app.R`

## 📊 Target Metrics

- Retrieval time: < 500ms
- Accuracy: > 85% on R documentation queries
- Memory usage: < 16GB for full deployment
- Concurrent users: 10+ on single HPCC node

## 🎤 Conference Presentation

**"A Fully Local, Free, and Version-Aware LLM-Powered Assistant for R Package Development and Reproducibility"**

Position as lightweight, sustainable alternative to expensive GPT-powered coding copilots. 