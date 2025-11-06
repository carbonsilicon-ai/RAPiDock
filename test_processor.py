#!/usr/bin/env python3
"""
Test script for RAPiDock Processor
"""

import sys
import os
import tempfile
import shutil
import pandas as pd
from workdir.processor import main


def test_main_function():
    """Test the main function with its new parameter-based interface"""
    print("\nTesting main function interface...")
    
    # Check if required files exist
    protein_file = "/workdir/data/7bbg/7BBG.pdb"
    reference_peptide_file = "/workdir/data/7bbg/7bbg_peptide.pdb"
    peptides = ["HKILHRLLQDS", "EKHKILHRLLQDS"]
    
    try:
        main(
            protein=protein_file, 
            reference_peptide=reference_peptide_file, 
            peptides=peptides,
            output="/workdir/results/test_run/"
        )
        print("✓ Main function test passed")
    except Exception as e:
        print(f"✗ Main function test failed: {e}")

if __name__ == "__main__":
    test_main_function()
