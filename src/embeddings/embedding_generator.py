"""Embedding generation using sentence transformers."""

import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import torch


class EmbeddingGenerator:
    """Generate embeddings for text chunks using sentence transformers."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: HuggingFace model name for sentence transformers
            device: Device to run on (cuda/cpu), auto-detected if None
            normalize_embeddings: Whether to normalize embeddings to unit length
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load the model
        print(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.encode_texts([text], show_progress=False)[0]
    
    def encode_chunks_with_metadata(self, chunks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for chunks and combine with metadata.
        
        Args:
            chunks_data: List of dictionaries with 'text' key and metadata
            
        Returns:
            List of dictionaries with embeddings and metadata
        """
        texts = [chunk['text'] for chunk in chunks_data]
        embeddings = self.encode_texts(texts)
        
        # Combine embeddings with metadata
        for i, chunk in enumerate(chunks_data):
            chunk['embedding'] = embeddings[i]
            chunk['embedding_model'] = self.model_name
        
        return chunks_data
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'embedding_dim': self.embedding_dim,
            'normalize_embeddings': self.normalize_embeddings
        }


class ContextAwareEmbeddingGenerator(EmbeddingGenerator):
    """Enhanced embedding generator with context-aware features for R documentation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # R-specific prefixes to enhance embeddings
        self.r_prefixes = {
            'function': "R function documentation: ",
            'class': "R class documentation: ",
            'package': "R package information: ",
            'vignette': "R package vignette: ",
            'help': "R help documentation: "
        }
    
    def enhance_text_with_context(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add context-aware prefixes to text before embedding."""
        enhanced_text = text
        
        # Add R-specific context
        if metadata.get('file_type') == '.R' and metadata.get('roxygen_functions'):
            enhanced_text = self.r_prefixes['function'] + enhanced_text
        elif metadata.get('file_type') == '.Rd':
            enhanced_text = self.r_prefixes['help'] + enhanced_text
        elif metadata.get('file_type') == '.Rmd':
            enhanced_text = self.r_prefixes['vignette'] + enhanced_text
        elif 'DESCRIPTION' in metadata.get('source_file', ''):
            enhanced_text = self.r_prefixes['package'] + enhanced_text
        
        # Add package context if available
        if metadata.get('package_name'):
            enhanced_text = f"Package {metadata['package_name']}: " + enhanced_text
        
        # Add section context if available
        if metadata.get('section_header'):
            enhanced_text = f"Section {metadata['section_header']}: " + enhanced_text
        
        return enhanced_text
    
    def encode_chunks_with_context(self, chunks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate context-enhanced embeddings for chunks."""
        # Enhance texts with context
        enhanced_texts = []
        for chunk in chunks_data:
            original_text = chunk['text']
            enhanced_text = self.enhance_text_with_context(original_text, chunk)
            enhanced_texts.append(enhanced_text)
            chunk['enhanced_text'] = enhanced_text
        
        # Generate embeddings for enhanced texts
        embeddings = self.encode_texts(enhanced_texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks_data):
            chunk['embedding'] = embeddings[i]
            chunk['embedding_model'] = self.model_name
        
        return chunks_data 