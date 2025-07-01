#!/usr/bin/env python3
"""Simple test download script."""

import requests
from pathlib import Path

def test_download():
    """Test function."""
    print("Test download script is working!")
    print("Current directory:", Path.cwd())
    
    # Create test directory
    test_dir = Path("data/test")
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {test_dir}")

def main():
    """Main function."""
    print("Running download test...")
    test_download()
    print("Test completed!")

if __name__ == "__main__":
    main() 