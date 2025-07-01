#!/usr/bin/env python3
"""Download R package documentation from CRAN."""

import os
import sys
import requests
import tarfile
import shutil
from pathlib import Path

def download_file(url: str, destination: Path) -> bool:
    """Download a file from URL to destination."""
    try:
        print(f"Downloading: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded to: {destination}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def extract_package(archive_path: Path, extract_dir: Path):
    """Extract R package archive."""
    try:
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        
        # Find the extracted package directory
        extracted_items = list(extract_dir.iterdir())
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            return extracted_items[0]
        return extract_dir
    except Exception as e:
        print(f"Error extracting {archive_path}: {e}")
        return None

def download_cran_package(package_name: str, download_dir: Path):
    """Download a package from CRAN."""
    try:
        packages_url = "https://cran.r-project.org/src/contrib/PACKAGES"
        response = requests.get(packages_url, timeout=30)
        response.raise_for_status()
        
        # Parse PACKAGES file to find version
        lines = response.text.split('\n')
        version = None
        
        for i, line in enumerate(lines):
            if line.startswith(f"Package: {package_name}"):
                for j in range(i+1, min(i+10, len(lines))):
                    if lines[j].startswith("Version: "):
                        version = lines[j].split("Version: ")[1].strip()
                        break
                break
        
        if version:
            archive_name = f"{package_name}_{version}.tar.gz"
            url = f"https://cran.r-project.org/src/contrib/{archive_name}"
            destination = download_dir / archive_name
            
            if download_file(url, destination):
                extract_dir = download_dir / f"{package_name}_extracted"
                extract_dir.mkdir(exist_ok=True)
                return extract_package(destination, extract_dir)
    
    except Exception as e:
        print(f"Error downloading CRAN package {package_name}: {e}")
    
    return None

def main():
    """Main function."""
    # Start with core data manipulation packages
    packages = [
        "argparse",
        "plyr", 
        "stringr",
        "purrr",
        "dplyr"
    ]
    
    print("R Package Documentation Downloader")
    print("=" * 50)
    print(f"Downloading {len(packages)} packages...")
    
    output_dir = Path("data/r_packages")
    download_dir = output_dir / "downloads"
    packages_dir = output_dir / "packages"
    
    download_dir.mkdir(parents=True, exist_ok=True)
    packages_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = {}
    
    for package_name in packages:
        print(f"\nProcessing package: {package_name}")
        
        final_package_dir = packages_dir / package_name
        if final_package_dir.exists():
            print(f"  Already exists: {final_package_dir}")
            downloaded[package_name] = final_package_dir
            continue
        
        print(f"  Downloading from CRAN...")
        extracted_path = download_cran_package(package_name, download_dir)
        
        if extracted_path and extracted_path.exists():
            if extracted_path != final_package_dir:
                if final_package_dir.exists():
                    shutil.rmtree(final_package_dir)
                shutil.move(str(extracted_path), str(final_package_dir))
            
            downloaded[package_name] = final_package_dir
            print(f"  Successfully downloaded: {package_name}")
        else:
            print(f"  Failed to download: {package_name}")
    
    print(f"\nDownload Summary:")
    print(f"Successfully downloaded: {len(downloaded)}/{len(packages)} packages")
    
    for package_name, path in downloaded.items():
        print(f"  {package_name}: {path}")
    
    print(f"\nPackages saved to: {packages_dir}")
    print("You can now run build_vector_store.py to process these packages!")

if __name__ == "__main__":
    main()