"""Text chunking with token counting and sliding window overlap."""

import re
import tiktoken
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class ChunkMetadata:
    """Metadata for document chunks."""
    chunk_id: str
    source_file: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    section_header: Optional[str] = None
    package_name: Optional[str] = None
    package_version: Optional[str] = None
    token_count: int = 0
    file_type: Optional[str] = None
    roxygen_functions: Optional[List[str]] = None
    rd_name: Optional[str] = None
    rd_title: Optional[str] = None


class TextChunker:
    """Advanced text chunker with dynamic sizing and metadata capture."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap_size: int = 50,
        encoding_name: str = "cl100k_base",
        min_chunk_size: int = 100
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target tokens per chunk
            overlap_size: Overlap tokens between chunks
            encoding_name: Tokenizer encoding to use
            min_chunk_size: Minimum tokens for a valid chunk
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.encoding = tiktoken.get_encoding(encoding_name)
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def extract_section_headers(self, text: str) -> List[Dict[str, Any]]:
        """Extract section headers and their positions."""
        headers = []
        # R documentation patterns
        patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headers
            r'^\\section\{([^}]+)\}',  # LaTeX sections
            r'^@(\w+)',  # Roxygen tags
            r'^\s*([A-Z][A-Za-z\s]+):\s*$',  # Description sections
        ]
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    headers.append({
                        'header': match.group(1),
                        'line_num': i,
                        'char_pos': sum(len(l) + 1 for l in lines[:i])
                    })
                    break
        return headers
    
    def extract_package_metadata(self, text: str, file_path: str) -> Dict[str, str]:
        """Extract package name and version from file content or path."""
        metadata = {}
        
        # Extract from file path
        path_parts = Path(file_path).parts
        if 'packages' in path_parts:
            pkg_idx = path_parts.index('packages')
            if pkg_idx + 1 < len(path_parts):
                metadata['package_name'] = path_parts[pkg_idx + 1]
        
        # Extract from content (DESCRIPTION files, roxygen comments)
        version_match = re.search(r'Version:\s*([0-9.-]+)', text)
        if version_match:
            metadata['package_version'] = version_match.group(1)
            
        package_match = re.search(r'Package:\s*(\w+)', text)
        if package_match:
            metadata['package_name'] = package_match.group(1)
            
        return metadata
    
    def chunk_text(self, text: str, file_path: str = "") -> List[ChunkMetadata]:
        """
        Chunk text with overlapping windows and metadata extraction.
        
        Args:
            text: Input text to chunk
            file_path: Source file path for metadata
            
        Returns:
            List of ChunkMetadata objects
        """
        if not text.strip():
            return []
            
        # Extract metadata
        package_metadata = self.extract_package_metadata(text, file_path)
        headers = self.extract_section_headers(text)
        
        # Tokenize the entire text
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)
        
        if total_tokens <= self.chunk_size:
            # Single chunk
            return [ChunkMetadata(
                chunk_id=f"{Path(file_path).stem}_0",
                source_file=file_path,
                start_char=0,
                end_char=len(text),
                start_line=0,
                end_line=len(text.split('\n')) - 1,
                section_header=headers[0]['header'] if headers else None,
                package_name=package_metadata.get('package_name'),
                package_version=package_metadata.get('package_version'),
                token_count=total_tokens
            )]
        
        chunks = []
        chunk_idx = 0
        start_token = 0
        
        while start_token < total_tokens:
            # Calculate chunk boundaries
            end_token = min(start_token + self.chunk_size, total_tokens)
            
            # Get the text for this chunk
            chunk_tokens = tokens[start_token:end_token]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Calculate character positions
            preceding_tokens = tokens[:start_token]
            preceding_text = self.encoding.decode(preceding_tokens) if preceding_tokens else ""
            start_char = len(preceding_text)
            end_char = start_char + len(chunk_text)
            
            # Calculate line positions
            text_lines = text[:start_char].split('\n')
            start_line = len(text_lines) - 1
            chunk_lines = chunk_text.split('\n')
            end_line = start_line + len(chunk_lines) - 1
            
            # Find relevant section header
            relevant_header = None
            for header in headers:
                if header['char_pos'] <= start_char:
                    relevant_header = header['header']
                else:
                    break
            
            # Create chunk metadata
            chunk_metadata = ChunkMetadata(
                chunk_id=f"{Path(file_path).stem}_{chunk_idx}",
                source_file=file_path,
                start_char=start_char,
                end_char=end_char,
                start_line=start_line,
                end_line=end_line,
                section_header=relevant_header,
                package_name=package_metadata.get('package_name'),
                package_version=package_metadata.get('package_version'),
                token_count=len(chunk_tokens)
            )
            
            chunks.append(chunk_metadata)
            
            # Move to next chunk with overlap
            if end_token >= total_tokens:
                break
            start_token = end_token - self.overlap_size
            chunk_idx += 1
            
        return chunks
    
    def chunk_file(self, file_path: str) -> List[ChunkMetadata]:
        """Chunk a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.chunk_text(text, file_path)
        except Exception as e:
            print(f"Error chunking file {file_path}: {e}")
            return [] 