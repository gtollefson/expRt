#!/usr/bin/env python3
"""Test pipeline for R package assistant development."""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunking.r_package_parser import RPackageParser
from chunking.text_chunker import TextChunker


def test_chunking():
    """Test the chunking functionality with a simple example."""
    print("=" * 50)
    print("Testing Text Chunking")
    print("=" * 50)
    
    # Create test R package structure
    parser = RPackageParser()
    test_package_dir = parser.create_test_package_structure()
    
    print(f"Created test package at: {test_package_dir}")
    
    # Test chunking
    chunker = TextChunker(chunk_size=256, overlap_size=50)
    
    # Process the test package
    chunks = parser.process_package(test_package_dir)
    
    print(f"\nProcessed {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Source: {chunk.source_file}")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Package: {chunk.package_name}")
        print(f"  File type: {getattr(chunk, 'file_type', 'N/A')}")
        if hasattr(chunk, 'roxygen_functions') and chunk.roxygen_functions:
            print(f"  Functions: {chunk.roxygen_functions}")
    
    return chunks


def test_embeddings():
    """Test embedding generation (mock without actual models)."""
    print("\n" + "=" * 50)
    print("Testing Embedding Generation (Mock)")
    print("=" * 50)
    
    # Mock embedding data
    chunks_data = [
        {
            'text': 'This is a test chunk about R functions',
            'chunk_id': 'test_1',
            'package_name': 'TestPackage',
            'file_type': '.R'
        },
        {
            'text': 'Another chunk about R documentation',
            'chunk_id': 'test_2',
            'package_name': 'TestPackage',
            'file_type': '.Rd'
        }
    ]
    
    # Mock the embedding process (since we might not have models installed yet)
    try:
        from embeddings.embedding_generator import EmbeddingGenerator
        
        # This will fail if sentence-transformers isn't installed
        # but we can at least test the class structure
        print("EmbeddingGenerator class loaded successfully")
        print("Note: Actual embedding generation requires sentence-transformers")
        
    except ImportError as e:
        print(f"Embedding generation dependencies not available: {e}")
        print("Install with: pip install sentence-transformers torch")
    
    return chunks_data


def test_vector_store():
    """Test vector store functionality (mock)."""
    print("\n" + "=" * 50)
    print("Testing Vector Store (Mock)")
    print("=" * 50)
    
    try:
        from embeddings.vector_store import VectorStore
        import numpy as np
        
        # Create a simple test vector store
        store = VectorStore(embedding_dim=384)  # Common dimension for all-MiniLM-L6-v2
        
        # Mock embeddings and metadata
        embeddings = np.random.rand(3, 384).astype(np.float32)
        metadata = [
            {'chunk_id': 'test_1', 'text': 'Test chunk 1', 'package_name': 'TestPackage'},
            {'chunk_id': 'test_2', 'text': 'Test chunk 2', 'package_name': 'TestPackage'},
            {'chunk_id': 'test_3', 'text': 'Test chunk 3', 'package_name': 'AnotherPackage'}
        ]
        
        store.add_vectors(embeddings, metadata)
        
        print(f"Added {len(embeddings)} vectors to store")
        print(f"Store stats: {store.get_stats()}")
        
        # Test search with mock query
        query_embedding = np.random.rand(384).astype(np.float32)
        results = store.search(query_embedding, k=2)
        
        print(f"\nMock search returned {len(results)} results:")
        for result in results:
            print(f"  - {result['chunk_id']}: score={result['score']:.3f}")
        
        # Test saving (create directory structure)
        test_store_dir = "data/test_vectorstore"
        Path(test_store_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\nVector store functionality verified")
        print("Note: Actual FAISS operations require faiss-cpu installation")
        
    except ImportError as e:
        print(f"Vector store dependencies not available: {e}")
        print("Install with: pip install faiss-cpu numpy")


def main():
    """Run all tests."""
    print("🚀 R Package Assistant - Test Pipeline")
    print("Testing core functionality without heavy dependencies")
    
    try:
        # Test 1: Chunking (should work with standard library)
        chunks = test_chunking()
        
        # Test 2: Embeddings (mock test)
        chunks_data = test_embeddings()
        
        # Test 3: Vector store (mock test)
        test_vector_store()
        
        print("\n" + "=" * 50)
        print("✅ Test Pipeline Complete!")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full pipeline with: python scripts/full_pipeline.py")
        print("3. Start API server: python src/api/server.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 