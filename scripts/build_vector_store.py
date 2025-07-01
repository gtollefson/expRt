#!/usr/bin/env python3
"""Build vector store with real R package data and embeddings."""

import sys
import os
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chunking.r_package_parser import RPackageParser
from chunking.text_chunker import TextChunker
from embeddings.embedding_generator import ContextAwareEmbeddingGenerator
from embeddings.vector_store import HierarchicalVectorStore

def create_sample_r_packages():
    """Create sample R package structures for testing."""
    packages_dir = Path("data/sample_packages")
    packages_dir.mkdir(parents=True, exist_ok=True)
    
    # Package 1: ggplot2-like package
    ggplot_dir = packages_dir / "ggplot2"
    ggplot_dir.mkdir(exist_ok=True)
    
    # DESCRIPTION file
    (ggplot_dir / "DESCRIPTION").write_text("""Package: ggplot2
Title: Create Elegant Data Visualisations Using the Grammar of Graphics
Version: 3.4.4
Authors@R: person("Hadley", "Wickham", email = "hadley@rstudio.com", role = c("aut", "cre"))
Description: A system for 'declaratively' creating graphics, based on "The Grammar of Graphics". 
    You provide the data, tell 'ggplot2' how to map variables to aesthetics, what graphical 
    primitives to use, and it takes care of the details.
Depends: R (>= 3.3)
Imports: cli, glue, gtable (>= 0.1.1), isoband, lifecycle (>= 1.0.1), MASS, mgcv, rlang (>= 1.0.0), 
    scales (>= 1.2.0), stats, tibble, vctrs (>= 0.5.0), withr (>= 2.5.0)
License: MIT + file LICENSE
URL: https://ggplot2.tidyverse.org
""")
    
    # R functions
    r_dir = ggplot_dir / "R"
    r_dir.mkdir(exist_ok=True)
    
    (r_dir / "ggplot.R").write_text("""#' Create a new ggplot
#' 
#' ggplot() initializes a ggplot object. It can be used to declare the input data frame
#' for a graphic and to specify the set of plot aesthetics intended to be common
#' throughout all subsequent layers unless specifically overridden.
#' 
#' @param data Default dataset to use for plot. If not already a data.frame,
#'   will be converted to one by fortify(). If not specified, must be supplied
#'   in each layer added to the plot.
#' @param mapping Default list of aesthetic mappings to use for plot.
#'   If not specified, must be supplied in each layer added to the plot.
#' @param ... Other arguments passed on to methods. Not currently used.
#' @param environment DEPRECATED. Used prior to tidy evaluation.
#' @return A ggplot object.
#' @export
#' @examples
#' # Create a simple scatter plot
#' ggplot(mtcars, aes(x = mpg, y = wt)) + geom_point()
#' 
#' # Create plot with no data
#' p <- ggplot() + geom_point(aes(x = mpg, y = wt), data = mtcars)
#' print(p)
ggplot <- function(data = NULL, mapping = aes(), ..., environment = parent.frame()) {
  if (!missing(environment)) {
    lifecycle::deprecate_warn("3.0.0", "ggplot(environment = )")
  }
  
  UseMethod("ggplot")
}

#' @export
#' @rdname ggplot
ggplot.default <- function(data = NULL, mapping = aes(), ..., environment = parent.frame()) {
  if (!is.null(data) && !is.data.frame(data)) {
    data <- fortify(data)
  }
  
  structure(
    list(
      data = data,
      layers = list(),
      scales = scales_list(),
      mapping = mapping,
      theme = theme_get(),
      coordinates = coord_cartesian(),
      facet = facet_null(),
      plot_env = environment
    ),
    class = c("gg", "ggplot")
  )
}

#' Add components to a ggplot
#' 
#' @param object An object of class ggplot
#' @param plot logical. If TRUE, plot is returned; if FALSE, a list.
#' @export
#' @rdname ggplot
print.ggplot <- function(x, ...) {
  plot <- ggplot_build(x)
  grid.draw(ggplot_gtable(plot))
  invisible(x)
}
""")
    
    (r_dir / "geom-point.R").write_text("""#' Points
#' 
#' The point geom is used to create scatterplots. The scatterplot is most useful for 
#' displaying the relationship between two continuous variables.
#' 
#' @section Aesthetics:
#' geom_point() understands the following aesthetics (required aesthetics are in bold):
#' \\itemize{
#'   \\item \\strong{x}
#'   \\item \\strong{y}
#'   \\item alpha
#'   \\item colour
#'   \\item fill
#'   \\item group
#'   \\item shape
#'   \\item size
#'   \\item stroke
#' }
#' 
#' @param mapping Set of aesthetic mappings created by aes() or aes_().
#' @param data The data to be displayed in this layer.
#' @param stat The statistical transformation to use on the data for this layer, as a string.
#' @param position Position adjustment, either as a string, or the result of a call to a position adjustment function.
#' @param ... Other arguments passed on to layer().
#' @param na.rm If FALSE, the default, missing values are removed with a warning.
#' @param show.legend logical. Should this layer be included in the legends?
#' @param inherit.aes If FALSE, overrides the default aesthetics.
#' @export
#' @examples
#' p <- ggplot(mtcars, aes(wt, mpg))
#' p + geom_point()
#' 
#' # Add aesthetic mappings
#' p + geom_point(aes(colour = factor(cyl)))
#' p + geom_point(aes(shape = factor(cyl)))
#' p + geom_point(aes(size = qsec))
geom_point <- function(mapping = NULL, data = NULL,
                      stat = "identity", position = "identity",
                      ...,
                      na.rm = FALSE,
                      show.legend = NA,
                      inherit.aes = TRUE) {
  layer(
    data = data,
    mapping = mapping,
    stat = stat,
    geom = GeomPoint,
    position = position,
    show.legend = show.legend,
    inherit.aes = inherit.aes,
    params = list(
      na.rm = na.rm,
      ...
    )
  )
}
""")
    
    # Package 2: Data manipulation package
    dplyr_dir = packages_dir / "dplyr"
    dplyr_dir.mkdir(exist_ok=True)
    
    (dplyr_dir / "DESCRIPTION").write_text("""Package: dplyr
Title: A Grammar of Data Manipulation
Version: 1.1.4
Authors@R: person("Hadley", "Wickham", email = "hadley@rstudio.com", role = c("aut", "cre"))
Description: A fast, consistent tool for working with data frame like objects,
    both in memory and out of memory.
Depends: R (>= 3.5.0)
Imports: cli (>= 3.4.0), generics, glue (>= 1.3.2), lifecycle (>= 1.0.3),
    magrittr (>= 1.5), pillar (>= 1.9.0), R6, rlang (>= 1.1.0), tibble (>= 3.2.0),
    tidyselect (>= 1.2.0), utils, vctrs (>= 0.6.4)
License: MIT + file LICENSE
URL: https://dplyr.tidyverse.org
""")
    
    dplyr_r_dir = dplyr_dir / "R"
    dplyr_r_dir.mkdir(exist_ok=True)
    
    (dplyr_r_dir / "filter.R").write_text("""#' Keep rows that match a condition
#' 
#' The filter() function is used to subset a data frame, retaining all rows that
#' satisfy your conditions. To be retained, the row must produce a value of TRUE
#' for all conditions.
#' 
#' @param .data A data frame, data frame extension (e.g. a tibble), or a lazy data frame
#' @param ... Logical predicates defined in terms of the variables in .data
#' @param .preserve Relevant when the .data input is grouped.
#' @return An object of the same type as .data. The output has the following properties:
#' @export
#' @examples
#' # Filter by single condition
#' filter(starwars, species == "Human")
#' filter(starwars, mass > 1000)
#' 
#' # Filter by multiple conditions
#' filter(starwars, hair_color == "none" & eye_color == "black")
#' filter(starwars, hair_color == "none", eye_color == "black")
filter <- function(.data, ..., .preserve = FALSE) {
  UseMethod("filter")
}

#' @export
filter.data.frame <- function(.data, ..., .preserve = FALSE) {
  loc <- eval_select_impl(.data, ..., error_call = current_env())
  
  if (length(loc) == 0) {
    return(.data[0L, , drop = FALSE])
  }
  
  .data[loc, , drop = FALSE]
}

#' Keep distinct/unique rows
#' 
#' Select only unique/distinct rows from a data frame.
#' 
#' @inheritParams filter
#' @param ... Optional variables to use when determining uniqueness.
#' @param .keep_all If TRUE, keep all variables in .data.
#' @export
#' @examples
#' df <- tibble(
#'   x = sample(10, 100, rep = TRUE),
#'   y = sample(10, 100, rep = TRUE)
#' )
#' nrow(df)
#' nrow(distinct(df))
#' nrow(distinct(df, x, y))
distinct <- function(.data, ..., .keep_all = FALSE) {
  UseMethod("distinct")
}
""")
    
    return packages_dir

def find_package_directories() -> List[Path]:
    """Find all available R package directories."""
    package_dirs = []
    
    # Check for real downloaded packages
    real_packages_dir = Path("data/r_packages/packages")
    if real_packages_dir.exists():
        print(f"Found real R packages directory: {real_packages_dir}")
        for pkg_dir in real_packages_dir.iterdir():
            if pkg_dir.is_dir():
                package_dirs.append(pkg_dir)
    
    # Check for sample packages as fallback
    sample_packages_dir = Path("data/sample_packages")
    if sample_packages_dir.exists():
        print(f"Found sample packages directory: {sample_packages_dir}")
        for pkg_dir in sample_packages_dir.iterdir():
            if pkg_dir.is_dir():
                package_dirs.append(pkg_dir)
    
    # If no packages found, create samples
    if not package_dirs:
        print("No packages found, creating sample packages...")
        sample_dir = create_sample_r_packages()
        for pkg_dir in sample_dir.iterdir():
            if pkg_dir.is_dir():
                package_dirs.append(pkg_dir)
    
    return package_dirs

def main():
    """Main function to build the vector store."""
    print("Building R Package Assistant Vector Store")
    print("=" * 50)
    
    # Find all package directories
    package_dirs = find_package_directories()
    print(f"Found {len(package_dirs)} R packages to process")
    
    for pkg_dir in package_dirs[:5]:  # Show first 5
        print(f"  - {pkg_dir.name}")
    if len(package_dirs) > 5:
        print(f"  ... and {len(package_dirs) - 5} more")
    
    # Initialize components
    print("\nInitializing text chunker...")
    chunker = TextChunker(
        chunk_size=1000,
        overlap_size=200,
        min_chunk_size=100
    )
    
    print("Initializing R package parser...")
    parser = RPackageParser()
    
    print("Initializing embedding generator...")
    try:
        embedding_generator = ContextAwareEmbeddingGenerator(
            model_name="all-MiniLM-L6-v2",
            normalize_embeddings=True
        )
        print(f"Embedding model loaded successfully (dim: {embedding_generator.embedding_dim})")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        print("This might be the first time downloading the model...")
        raise
    
    print("Initializing vector store...")
    vector_store = HierarchicalVectorStore(
        embedding_dim=embedding_generator.embedding_dim,
        index_type="flat"
    )
    
    # Process packages
    all_chunks = []
    processed_packages = 0
    
    for package_dir in package_dirs:
        try:
            print(f"\nProcessing package: {package_dir.name}")
            
            # Parse package and get chunks directly
            chunk_metadata_list = parser.process_package(str(package_dir))
            print(f"  Created {len(chunk_metadata_list)} chunks")
            
            # Convert ChunkMetadata objects to dictionaries with text content
            package_chunks = []
            for chunk_meta in chunk_metadata_list:
                try:
                    # Read the source file and extract the chunk text
                    with open(chunk_meta.source_file, 'r', encoding='utf-8', errors='ignore') as f:
                        full_text = f.read()
                        chunk_text = full_text[chunk_meta.start_char:chunk_meta.end_char]
                    
                    # Create dictionary format expected by embedding generator
                    chunk_dict = {
                        'text': chunk_text,
                        'chunk_id': chunk_meta.chunk_id,
                        'source_file': chunk_meta.source_file,
                        'package_name': chunk_meta.package_name,
                        'file_type': chunk_meta.file_type,
                        'start_char': chunk_meta.start_char,
                        'end_char': chunk_meta.end_char,
                        'start_line': chunk_meta.start_line,
                        'end_line': chunk_meta.end_line,
                        'token_count': chunk_meta.token_count,
                        'section_header': chunk_meta.section_header,
                        'roxygen_functions': chunk_meta.roxygen_functions,
                        'rd_name': chunk_meta.rd_name,
                        'rd_title': chunk_meta.rd_title
                    }
                    package_chunks.append(chunk_dict)
                    
                except Exception as e:
                    print(f"    Warning: Could not read text for chunk {chunk_meta.chunk_id}: {e}")
                    continue
            
            print(f"  Successfully converted {len(package_chunks)} chunks with text")
            all_chunks.extend(package_chunks)
            processed_packages += 1
            
        except Exception as e:
            print(f"  Error processing {package_dir.name}: {e}")
            continue
    
    print(f"\nSuccessfully processed {processed_packages} packages")
    print(f"Total chunks created: {len(all_chunks)}")
    
    if not all_chunks:
        print("No chunks created! Check package processing.")
        return
    
    # Generate embeddings
    print("Generating embeddings with context enhancement...")
    try:
        chunks_with_embeddings = embedding_generator.encode_chunks_with_context(all_chunks)
        print(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise
    
    # Extract embeddings and metadata for vector store
    embeddings = []
    metadata_list = []
    
    for chunk in chunks_with_embeddings:
        embeddings.append(chunk['embedding'])
        
        # Prepare metadata for vector store
        metadata = {
            'chunk_id': chunk['chunk_id'],
            'text': chunk['text'],
            'enhanced_text': chunk.get('enhanced_text', chunk['text']),
            'package_name': chunk.get('package_name'),
            'source_file': chunk.get('source_file'),
            'file_type': chunk.get('file_type'),
            'token_count': chunk.get('token_count'),
            'section_header': chunk.get('section_header'),
            'roxygen_functions': chunk.get('roxygen_functions', []),
            'embedding_model': chunk.get('embedding_model')
        }
        metadata_list.append(metadata)
    
    # Add to vector store
    print("Building FAISS vector index...")
    try:
        import numpy as np
        embeddings_array = np.array(embeddings)
        vector_store.add_vectors(embeddings_array, metadata_list)
        print(f"Added {len(embeddings)} vectors to FAISS index")
    except Exception as e:
        print(f"Error building FAISS index: {e}")
        raise
    
    # Save vector store
    store_path = "data/vectorstore"
    print(f"Saving vector store to: {store_path}")
    try:
        vector_store.save(store_path)
        print("Vector store saved successfully")
    except Exception as e:
        print(f"Error saving vector store: {e}")
        raise
    
    # Display statistics
    stats = vector_store.get_stats()
    print("\nVector Store Statistics:")
    print(f"  Total vectors: {stats['total_vectors']}")
    print(f"  Embedding dimension: {stats['embedding_dim']}")
    print(f"  Index type: {stats['index_type']}")
    print(f"  Packages processed: {len(stats['packages'])}")
    
    print("  Package breakdown:")
    for pkg, count in sorted(stats['packages'].items()):
        print(f"    {pkg}: {count} chunks")
    
    print("  File type breakdown:")
    for file_type, count in sorted(stats['file_types'].items()):
        print(f"    {file_type}: {count} chunks")
    
    # Test search functionality
    print("\nTesting search functionality...")
    test_queries = [
        "How to create a scatter plot?",
        "Filter data frame rows",
        "genomic ranges operations",
        "string manipulation functions",
        "data manipulation with dplyr",
        "BSgenome human genome",
        "IRanges interval operations"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            query_embedding = embedding_generator.encode_single(query)
            results = vector_store.search(query_embedding, k=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"  {i}. Package: {result.get('package_name', 'Unknown')}")
                    print(f"     File: {result.get('source_file', 'Unknown')}")
                    print(f"     Score: {result.get('score', 0):.3f}")
                    print(f"     Text: {result.get('text', '')[:100]}...")
            else:
                print("  No results found")
        except Exception as e:
            print(f"  Error during search: {e}")
    
    print(f"\nVector store successfully built and saved to {store_path}")
    print("Next steps:")
    print("1. Start API server: python src/api/server.py")
    print("2. Launch Shiny frontend: Rscript frontend/shiny/app.R")
    print("3. Test queries through the web interface")

if __name__ == "__main__":
    main() 