#!/bin/bash
#SBATCH --job-name=r-assistant-process
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/process_%j.out
#SBATCH --error=logs/process_%j.err
#SBATCH --array=1-10%2  # Process 10 batches, max 2 concurrent

# R Package Assistant - Package Processing
echo "🔄 Processing R packages for vector store"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"

# Load environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false  # Avoid warnings

# Create processing script
cat > scripts/batch_process.py << 'EOF'
#!/usr/bin/env python3
"""Batch processing script for R packages."""

import os
import sys
import json
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunking.r_package_parser import RPackageParser
from embeddings.embedding_generator import ContextAwareEmbeddingGenerator
from embeddings.vector_store import HierarchicalVectorStore


def process_package_batch(package_dirs: List[str], batch_id: int):
    """Process a batch of R packages."""
    print(f"Processing batch {batch_id} with {len(package_dirs)} packages")
    
    # Initialize components
    parser = RPackageParser()
    embedding_gen = ContextAwareEmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",
        device=None  # Auto-detect
    )
    
    # Create vector store
    vector_store = HierarchicalVectorStore(
        embedding_dim=embedding_gen.embedding_dim,
        index_type="flat"
    )
    
    all_chunks = []
    
    # Process each package
    for i, package_dir in enumerate(package_dirs):
        print(f"Processing package {i+1}/{len(package_dirs)}: {package_dir}")
        
        try:
            # Parse package
            chunks = parser.process_package(package_dir)
            print(f"  Found {len(chunks)} chunks")
            
            # Convert chunks to dict format for embedding
            chunks_data = []
            for chunk in chunks:
                chunk_text = ""
                # Read actual text content
                try:
                    with open(chunk.source_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        chunk_text = content[chunk.start_char:chunk.end_char]
                except:
                    chunk_text = f"Chunk from {chunk.source_file}"
                
                chunk_dict = {
                    'text': chunk_text,
                    'chunk_id': chunk.chunk_id,
                    'source_file': chunk.source_file,
                    'package_name': chunk.package_name,
                    'file_type': getattr(chunk, 'file_type', None),
                    'section_header': chunk.section_header,
                    'token_count': chunk.token_count,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char
                }
                chunks_data.append(chunk_dict)
            
            # Generate embeddings
            if chunks_data:
                print(f"  Generating embeddings for {len(chunks_data)} chunks")
                embedded_chunks = embedding_gen.encode_chunks_with_context(chunks_data)
                all_chunks.extend(embedded_chunks)
            
        except Exception as e:
            print(f"  Error processing {package_dir}: {e}")
            continue
    
    # Add to vector store
    if all_chunks:
        print(f"Adding {len(all_chunks)} chunks to vector store")
        embeddings = [chunk['embedding'] for chunk in all_chunks]
        metadata = [{k: v for k, v in chunk.items() if k != 'embedding'} 
                   for chunk in all_chunks]
        
        import numpy as np
        vector_store.add_vectors(np.array(embeddings), metadata)
        
        # Save batch vector store
        output_dir = f"data/vectorstore_batch_{batch_id}"
        vector_store.save(output_dir)
        
        # Save processing log
        log_data = {
            'batch_id': batch_id,
            'packages_processed': len(package_dirs),
            'total_chunks': len(all_chunks),
            'embedding_model': embedding_gen.model_name,
            'timestamp': str(datetime.now())
        }
        
        with open(f"data/processing_log_batch_{batch_id}.json", "w") as f:
            json.dump(log_data, f, indent=2)
        
        print(f"✅ Batch {batch_id} complete: {len(all_chunks)} chunks processed")
        return len(all_chunks)
    
    return 0


def main():
    """Main processing function."""
    import datetime
    
    batch_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 1))
    
    # Define package directories to process
    # This would be expanded to include actual R package locations
    base_packages = [
        "data/test_package",  # Our test package
        # Add more package directories here
        # "/path/to/cran/packages/ggplot2",
        # "/path/to/cran/packages/dplyr",
        # etc.
    ]
    
    # For demo, just process the test package
    package_dirs = base_packages
    
    if not package_dirs:
        print("No packages to process")
        return
    
    # Process the batch
    total_chunks = process_package_batch(package_dirs, batch_id)
    print(f"Batch processing complete: {total_chunks} chunks")


if __name__ == "__main__":
    main()
EOF

# Make the script executable
chmod +x scripts/batch_process.py

# Run the batch processing
echo "Starting batch processing..."
python scripts/batch_process.py

# Cleanup
rm scripts/batch_process.py

echo "✅ Package processing complete for batch $SLURM_ARRAY_TASK_ID" 