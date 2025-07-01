"""R package documentation parser and processor."""

import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from .text_chunker import TextChunker, ChunkMetadata


class RPackageParser:
    """Parser for R package documentation and source files."""
    
    def __init__(self, chunker: Optional[TextChunker] = None):
        """Initialize with optional custom chunker."""
        self.chunker = chunker or TextChunker()
        
        # R file extensions to process
        self.r_extensions = {'.R', '.r', '.Rmd', '.Rnw', '.Rd'}
        
        # Priority order for file types
        self.file_priorities = {
            'DESCRIPTION': 10,
            'NAMESPACE': 9,
            '.Rd': 8,  # Help files
            '.Rmd': 7,  # Vignettes
            '.R': 6,   # Source code
            'README': 5,
            'NEWS': 4,
            'CHANGELOG': 4,
        }
    
    def extract_roxygen_docs(self, r_code: str) -> List[Dict[str, Any]]:
        """Extract roxygen2 documentation from R source files."""
        docs = []
        lines = r_code.split('\n')
        current_doc = []
        current_function = None
        
        for i, line in enumerate(lines):
            # Roxygen comment
            if line.strip().startswith('#\''):
                current_doc.append(line.strip()[2:].strip())
            
            # Function definition
            elif re.match(r'^\s*(\w+)\s*<-\s*function', line):
                func_match = re.match(r'^\s*(\w+)\s*<-\s*function', line)
                if func_match and current_doc:
                    current_function = func_match.group(1)
                    docs.append({
                        'function_name': current_function,
                        'documentation': '\n'.join(current_doc),
                        'line_number': i,
                        'type': 'function'
                    })
                current_doc = []
                current_function = None
            
            # Class definition (S4)
            elif 'setClass' in line and current_doc:
                class_match = re.search(r'setClass\(["\'](\w+)["\']', line)
                if class_match:
                    docs.append({
                        'class_name': class_match.group(1),
                        'documentation': '\n'.join(current_doc),
                        'line_number': i,
                        'type': 'class'
                    })
                current_doc = []
            
            # Clear doc if we hit non-roxygen, non-function line
            elif line.strip() and not line.strip().startswith('#'):
                if not re.match(r'^\s*(\w+)\s*<-\s*function', line):
                    current_doc = []
        
        return docs
    
    def parse_rd_file(self, rd_content: str) -> Dict[str, Any]:
        """Parse R documentation (.Rd) files."""
        doc_info = {
            'name': None,
            'title': None,
            'description': None,
            'usage': None,
            'arguments': [],
            'value': None,
            'examples': None,
            'seealso': None,
            'author': None
        }
        
        # Extract sections using regex
        sections = {
            'name': r'\\name\{([^}]+)\}',
            'title': r'\\title\{([^}]+)\}',
            'description': r'\\description\{(.*?)\}',
            'usage': r'\\usage\{(.*?)\}',
            'value': r'\\value\{(.*?)\}',
            'examples': r'\\examples\{(.*?)\}',
            'seealso': r'\\seealso\{(.*?)\}',
            'author': r'\\author\{(.*?)\}'
        }
        
        for key, pattern in sections.items():
            match = re.search(pattern, rd_content, re.DOTALL)
            if match:
                doc_info[key] = match.group(1).strip()
        
        # Extract arguments
        args_section = re.search(r'\\arguments\{(.*?)\}', rd_content, re.DOTALL)
        if args_section:
            args_text = args_section.group(1)
            arg_items = re.findall(r'\\item\{([^}]+)\}\{(.*?)\}', args_text, re.DOTALL)
            doc_info['arguments'] = [
                {'name': name.strip(), 'description': desc.strip()}
                for name, desc in arg_items
            ]
        
        return doc_info
    
    def get_file_priority(self, file_path: str) -> int:
        """Get priority score for file processing order."""
        path = Path(file_path)
        
        # Check filename patterns
        for pattern, priority in self.file_priorities.items():
            if pattern.startswith('.') and path.suffix == pattern:
                return priority
            elif pattern in path.name.upper():
                return priority
        
        return 1  # Default priority
    
    def discover_r_files(self, package_dir: str) -> List[str]:
        """Discover all R-related files in a package directory."""
        files = []
        package_path = Path(package_dir)
        
        if not package_path.exists():
            return files
        
        # Standard R package structure
        search_dirs = ['R', 'man', 'vignettes', 'inst/doc', '.']
        
        for search_dir in search_dirs:
            dir_path = package_path / search_dir
            if dir_path.exists():
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file():
                        # Check if it's an R-related file
                        if (file_path.suffix in self.r_extensions or
                            file_path.name.upper() in ['DESCRIPTION', 'NAMESPACE', 'README', 'NEWS', 'CHANGELOG']):
                            files.append(str(file_path))
        
        # Sort by priority
        files.sort(key=self.get_file_priority, reverse=True)
        return files
    
    def process_package(self, package_dir: str) -> List[ChunkMetadata]:
        """Process an entire R package and return chunks with metadata."""
        all_chunks = []
        r_files = self.discover_r_files(package_dir)
        
        for file_path in r_files:
            try:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        return all_chunks
    
    def process_file(self, file_path: str) -> List[ChunkMetadata]:
        """Process a single R-related file."""
        path = Path(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with latin-1 encoding as fallback
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Basic chunking
        chunks = self.chunker.chunk_text(content, file_path)
        
        # Add file-type specific metadata
        for chunk in chunks:
            chunk.file_type = path.suffix or 'other'
            
            # Add R-specific metadata
            if path.suffix == '.R':
                # Extract roxygen docs for this chunk
                chunk_text = content[chunk.start_char:chunk.end_char]
                roxygen_docs = self.extract_roxygen_docs(chunk_text)
                if roxygen_docs:
                    chunk.roxygen_functions = [doc['function_name'] for doc in roxygen_docs]
            
            elif path.suffix == '.Rd':
                # Parse R documentation
                rd_info = self.parse_rd_file(content)
                chunk.rd_name = rd_info.get('name')
                chunk.rd_title = rd_info.get('title')
        
        return chunks
    
    def create_test_package_structure(self, output_dir: str = "data/test_package"):
        """Create a minimal test R package for development."""
        base_path = Path(output_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # DESCRIPTION file
        description_content = """Package: TestPackage
Title: Test Package for LLM Assistant
Version: 0.1.0
Author: Test Author
Maintainer: Test Author <test@example.com>
Description: A minimal test package for developing the R package LLM assistant.
License: MIT
Encoding: UTF-8
Imports: ggplot2, dplyr
Suggests: testthat, knitr
"""
        
        # R function file
        r_function_content = """#' Add two numbers
#'
#' This function takes two numeric inputs and returns their sum.
#'
#' @param x A numeric value
#' @param y A numeric value
#' @return The sum of x and y
#' @examples
#' add_numbers(2, 3)
#' add_numbers(10, -5)
#' @export
add_numbers <- function(x, y) {
  if (!is.numeric(x) || !is.numeric(y)) {
    stop("Both inputs must be numeric")
  }
  return(x + y)
}

#' Create a simple plot
#'
#' Generate a basic scatter plot using ggplot2.
#'
#' @param data A data frame with x and y columns
#' @param title Plot title
#' @return A ggplot object
#' @import ggplot2
#' @export
simple_plot <- function(data, title = "Simple Plot") {
  ggplot(data, aes(x = x, y = y)) +
    geom_point() +
    labs(title = title) +
    theme_minimal()
}
"""
        
        # Create directories
        (base_path / "R").mkdir(exist_ok=True)
        (base_path / "man").mkdir(exist_ok=True)
        
        # Write files
        with open(base_path / "DESCRIPTION", "w") as f:
            f.write(description_content)
        
        with open(base_path / "R" / "functions.R", "w") as f:
            f.write(r_function_content)
        
        print(f"Created test package structure in {output_dir}")
        return str(base_path) 