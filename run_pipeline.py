#!/usr/bin/env python3
"""
Thesis Pipeline Runner

Executes thesis_pipeline_v6.py (the active, stable pipeline).

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
import os

def main():
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run the main pipeline
    result = subprocess.run([sys.executable, "thesis_pipeline_v6.py"], check=False)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
