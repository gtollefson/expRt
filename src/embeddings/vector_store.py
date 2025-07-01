"""FAISS-based vector store with metadata indexing."""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import faiss


class VectorStore:
    """FAISS-based vector store with metadata support."""
    
    def __init__(self, embedding_dim: int, index_type: str = "flat"):
        """
        Initialize the vector store.
        
        Args:
            embedding_dim: Dimension of the embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metadata = []  # Store metadata for each vector
        self.id_to_index = {}  # Map chunk IDs to vector indices
        
        # Create FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (for normalized vectors)
        elif index_type == "ivf":
            # IVF index for larger datasets
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)
        elif index_type == "hnsw":
            # HNSW index for fast approximate search
            self.index = faiss.IndexHNSWFlat(embedding_dim, 32)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
    
    def add_vectors(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """
        Add vectors and their metadata to the store.
        
        Args:
            embeddings: numpy array of embeddings
            metadata_list: List of metadata dictionaries
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Number of embeddings must match metadata list length")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        
        # Add vectors to FAISS index
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        
        # Store metadata
        for i, metadata in enumerate(metadata_list):
            vector_idx = start_idx + i
            self.metadata.append(metadata)
            
            # Map chunk ID to vector index
            if 'chunk_id' in metadata:
                self.id_to_index[metadata['chunk_id']] = vector_idx
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of results with scores and metadata
        """
        query_embedding = query_embedding.astype(np.float32)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search in FAISS index
        search_k = min(k * 5, self.index.ntotal) if filter_metadata else k
        scores, indices = self.index.search(query_embedding, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # No more results
                break
                
            metadata = self.metadata[idx].copy()
            metadata['score'] = float(score)
            metadata['index'] = int(idx)
            
            # Apply metadata filters
            if filter_metadata:
                if not self._matches_filter(metadata, filter_metadata):
                    continue
            
            results.append(metadata)
            
            if len(results) >= k:
                break
        
        return results
    
    def _matches_filter(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the given filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            metadata_value = metadata[key]
            
            # Handle different filter types
            if isinstance(value, list):
                if metadata_value not in value:
                    return False
            elif isinstance(value, dict):
                # Range filters
                if 'min' in value and metadata_value < value['min']:
                    return False
                if 'max' in value and metadata_value > value['max']:
                    return False
            else:
                if metadata_value != value:
                    return False
        
        return True
    
    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata by chunk ID."""
        if chunk_id in self.id_to_index:
            idx = self.id_to_index[chunk_id]
            return self.metadata[idx].copy()
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if not self.metadata:
            return {'total_vectors': 0}
        
        # Count by package
        packages = {}
        file_types = {}
        
        for meta in self.metadata:
            pkg = meta.get('package_name', 'unknown')
            packages[pkg] = packages.get(pkg, 0) + 1
            
            file_type = meta.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        return {
            'total_vectors': len(self.metadata),
            'packages': packages,
            'file_types': file_types,
            'index_type': self.index_type,
            'embedding_dim': self.embedding_dim
        }
    
    def save(self, directory: str):
        """Save the vector store to disk."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(dir_path / "index.faiss"))
        
        # Save metadata
        with open(dir_path / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        # Save ID mapping
        with open(dir_path / "id_mapping.pkl", "wb") as f:
            pickle.dump(self.id_to_index, f)
        
        # Save configuration
        config = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'total_vectors': len(self.metadata)
        }
        with open(dir_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Vector store saved to {directory}")
    
    @classmethod
    def load(cls, directory: str) -> 'VectorStore':
        """Load a vector store from disk."""
        dir_path = Path(directory)
        
        # Load configuration
        with open(dir_path / "config.json", "r") as f:
            config = json.load(f)
        
        # Create instance
        store = cls(config['embedding_dim'], config['index_type'])
        
        # Load FAISS index
        store.index = faiss.read_index(str(dir_path / "index.faiss"))
        
        # Load metadata
        with open(dir_path / "metadata.json", "r") as f:
            store.metadata = json.load(f)
        
        # Load ID mapping
        with open(dir_path / "id_mapping.pkl", "rb") as f:
            store.id_to_index = pickle.load(f)
        
        print(f"Vector store loaded from {directory}")
        return store


class HierarchicalVectorStore(VectorStore):
    """Enhanced vector store with hierarchical indexing."""
    
    def __init__(self, embedding_dim: int, index_type: str = "flat"):
        super().__init__(embedding_dim, index_type)
        
        # Hierarchical indices
        self.package_indices = {}  # Package name -> vector indices
        self.file_type_indices = {}  # File type -> vector indices
        self.section_indices = {}  # Section header -> vector indices
    
    def add_vectors(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """Add vectors with hierarchical indexing."""
        super().add_vectors(embeddings, metadata_list)
        
        # Update hierarchical indices
        start_idx = self.index.ntotal - len(embeddings)
        
        for i, metadata in enumerate(metadata_list):
            vector_idx = start_idx + i
            
            # Index by package
            pkg = metadata.get('package_name')
            if pkg:
                if pkg not in self.package_indices:
                    self.package_indices[pkg] = []
                self.package_indices[pkg].append(vector_idx)
            
            # Index by file type
            file_type = metadata.get('file_type')
            if file_type:
                if file_type not in self.file_type_indices:
                    self.file_type_indices[file_type] = []
                self.file_type_indices[file_type].append(vector_idx)
            
            # Index by section
            section = metadata.get('section_header')
            if section:
                if section not in self.section_indices:
                    self.section_indices[section] = []
                self.section_indices[section].append(vector_idx)
    
    def search_hierarchical(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        package_name: Optional[str] = None,
        file_type: Optional[str] = None,
        section_header: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search with hierarchical filtering."""
        # Get candidate indices based on hierarchy
        candidate_indices = set(range(self.index.ntotal))
        
        if package_name and package_name in self.package_indices:
            candidate_indices &= set(self.package_indices[package_name])
        
        if file_type and file_type in self.file_type_indices:
            candidate_indices &= set(self.file_type_indices[file_type])
        
        if section_header and section_header in self.section_indices:
            candidate_indices &= set(self.section_indices[section_header])
        
        # If we have specific indices, search within them
        if len(candidate_indices) < self.index.ntotal:
            candidate_indices = list(candidate_indices)
            if not candidate_indices:
                return []
            
            # Get embeddings for candidate indices
            candidate_embeddings = np.array([
                self.index.reconstruct(idx) for idx in candidate_indices
            ])
            
            # Compute similarities
            query_embedding = query_embedding.astype(np.float32)
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()
            
            # Get top k
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for i in top_indices:
                idx = candidate_indices[i]
                metadata = self.metadata[idx].copy()
                metadata['score'] = float(similarities[i])
                metadata['index'] = int(idx)
                results.append(metadata)
            
            return results
        
        # Fall back to regular search
        return self.search(query_embedding, k) 