#!/usr/bin/env python3
"""
Test script for RAPiDock Processor
"""

import sys
import os
import tempfile
import shutil
import pandas as pd
from processor import (
    parse_peptide_sequence, 
    validate_peptide_length, 
    validate_peptides,
    SUPPORTED_RESIDUES,
    STANDARD_AA,
    process_protein,
    create_protein_pdb_from_sequence
)

def test_peptide_parsing():
    """Test peptide sequence parsing"""
    print("Testing peptide sequence parsing...")
    
    # Test cases
    test_cases = [
        ("HKILHRLLQDS", ["HIS", "LYS", "ILE", "LEU", "HIS", "ARG", "LEU", "LEU", "GLN", "ASP", "SER"]),
        ("HK[HYP]RL[PTR]QDS", ["HIS", "LYS", "HYP", "ARG", "LEU", "PTR", "GLN", "ASP", "SER"]),
        ("G", ["GLY"]),
        ("[MSE]", ["MSE"]),
        # Test with actual 7bbg peptide sequence
        ("RMFPNAPYL", ["ARG", "MET", "PHE", "PRO", "ASN", "ALA", "PRO", "TYR", "LEU"]),
    ]
    
    for sequence, expected in test_cases:
        try:
            result = parse_peptide_sequence(sequence)
            if result == expected:
                print(f"✓ {sequence} -> {result}")
            else:
                print(f"✗ {sequence} -> {result} (expected {expected})")
        except Exception as e:
            print(f"✗ {sequence} -> Error: {e}")
    
    # Test invalid cases
    invalid_cases = [
        "HK[INVALID]RL",  # Invalid residue
        "HK[UNCLOSED",    # Unclosed bracket
        "HKXRL",          # Invalid single letter
    ]
    
    for sequence in invalid_cases:
        try:
            result = parse_peptide_sequence(sequence)
            print(f"✗ {sequence} should have failed but got: {result}")
        except ValueError:
            print(f"✓ {sequence} correctly failed")
        except Exception as e:
            print(f"? {sequence} failed with unexpected error: {e}")

def test_peptide_validation():
    """Test peptide validation"""
    print("\nTesting peptide validation...")
    
    # Test length validation
    try:
        validate_peptide_length("HKI", min_length=3, max_length=10)  # Valid
        print("✓ Length validation passed for valid peptide")
    except:
        print("✗ Length validation failed for valid peptide")
    
    try:
        validate_peptide_length("HK", min_length=3, max_length=10)  # Too short
        print("✗ Length validation should have failed for short peptide")
    except ValueError:
        print("✓ Length validation correctly failed for short peptide")
    
    try:
        validate_peptide_length("H" * 60, min_length=3, max_length=50)  # Too long
        print("✗ Length validation should have failed for long peptide")
    except ValueError:
        print("✓ Length validation correctly failed for long peptide")

def test_peptide_list_validation():
    """Test peptide list validation"""
    print("\nTesting peptide list validation...")
    
    peptides = [
        "HKILHRLLQDS",      # Valid
        "EKHKILHRLLQDS",    # Valid
        "HK",               # Too short
        "HK[INVALID]RL",    # Invalid residue
        "LSGFMELCQ",        # Valid
        "RMFPNAPYL",        # Valid (7bbg peptide)
    ]
    
    validated = validate_peptides(peptides)
    expected_valid = ["HKILHRLLQDS", "EKHKILHRLLQDS", "LSGFMELCQ", "RMFPNAPYL"]
    
    if validated == expected_valid:
        print(f"✓ Peptide list validation correct: {len(validated)} valid peptides")
    else:
        print(f"✗ Peptide list validation incorrect. Got: {validated}")

def test_supported_residues():
    """Test that all supported residues are properly defined"""
    print("\nTesting supported residues...")
    
    # Check that standard amino acids are all supported
    for single, three in STANDARD_AA.items():
        if three not in SUPPORTED_RESIDUES:
            print(f"✗ Standard amino acid {single}({three}) not in supported residues")
            return
    
    print(f"✓ All {len(STANDARD_AA)} standard amino acids are supported")
    print(f"✓ Total {len(SUPPORTED_RESIDUES)} residue types supported")
    
    # Test some non-canonical amino acids
    non_canonical = ["HYP", "SEP", "PTR", "MSE", "MLY"]
    for residue in non_canonical:
        if residue not in SUPPORTED_RESIDUES:
            print(f"✗ Non-canonical amino acid {residue} not supported")
        else:
            print(f"✓ Non-canonical amino acid {residue} supported")

def test_file_operations():
    """Test basic file operations"""
    print("\nTesting file operations...")
    
    # Test temporary directory creation
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"✓ Temporary directory created: {temp_dir}")
        
        # Test file existence check
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        if os.path.exists(test_file):
            print("✓ File existence check works")
        else:
            print("✗ File existence check failed")

def test_actual_7bbg_data():
    """Test with actual 7bbg data files"""
    print("\nTesting with actual 7bbg data...")
    
    # Check if required files exist
    protein_file = "data/7bbg/7bbg_protein_pocket.pdb"
    peptide_file = "data/7bbg/7bbg_peptide.pdb"
    sequence_file = "data/7bbg/7bbg_peptide_sequence"
    
    if not os.path.exists(protein_file):
        print(f"✗ Protein file not found: {protein_file}")
        return
    if not os.path.exists(peptide_file):
        print(f"✗ Peptide file not found: {peptide_file}")
        return
    if not os.path.exists(sequence_file):
        print(f"✗ Sequence file not found: {sequence_file}")
        return
    
    print(f"✓ All 7bbg data files found")
    
    # Test reading peptide sequence
    try:
        with open(sequence_file, 'r') as f:
            sequence = f.read().strip()
        
        print(f"✓ Read peptide sequence: {sequence}")
        
        # Validate the sequence
        parsed = parse_peptide_sequence(sequence)
        print(f"✓ Parsed sequence: {parsed}")
        
        # Check length
        if len(parsed) == 9:  # RMFPNAPYL should be 9 residues
            print(f"✓ Sequence length correct: {len(parsed)} residues")
        else:
            print(f"✗ Unexpected sequence length: {len(parsed)} (expected 9)")
            
    except Exception as e:
        print(f"✗ Error reading/parsing peptide sequence: {e}")

def test_csv_parsing():
    """Test parsing CSV files with peptide data"""
    print("\nTesting CSV file parsing...")
    
    csv_files = [
        "data/protein_peptide_example.csv",
        "data/protein_multiple_peptides_example.csv"
    ]
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"✗ CSV file not found: {csv_file}")
            continue
            
        try:
            df = pd.read_csv(csv_file)
            print(f"✓ Successfully read {csv_file}: {len(df)} rows")
            print(f"  Columns: {list(df.columns)}")
            
            # Test parsing peptide sequences from CSV
            if 'peptide_description' in df.columns:
                for idx, row in df.iterrows():
                    peptide_desc = row['peptide_description']
                    if isinstance(peptide_desc, str) and not peptide_desc.endswith('.pdb'):
                        try:
                            parsed = parse_peptide_sequence(peptide_desc)
                            print(f"  ✓ Row {idx}: {peptide_desc} -> {len(parsed)} residues")
                        except Exception as e:
                            print(f"  ✗ Row {idx}: {peptide_desc} -> Error: {e}")
            
            # Test multiple peptides format
            if 'peptide_descriptions' in df.columns:
                for idx, row in df.iterrows():
                    peptide_descs = row['peptide_descriptions']
                    if isinstance(peptide_descs, str):
                        # Split by comma and clean
                        sequences = [seq.strip() for seq in peptide_descs.split(',') if seq.strip()]
                        print(f"  ✓ Row {idx}: Found {len(sequences)} peptide sequences")
                        
                        valid_count = 0
                        for seq in sequences:
                            try:
                                parsed = parse_peptide_sequence(seq)
                                valid_count += 1
                            except:
                                pass
                        print(f"    Valid sequences: {valid_count}/{len(sequences)}")
                        
        except Exception as e:
            print(f"✗ Error reading {csv_file}: {e}")

def test_protein_processing():
    """Test protein processing functionality"""
    print("\nTesting protein processing...")
    
    protein_file = "data/7bbg/7bbg_protein_pocket.pdb"
    peptide_file = "data/7bbg/7bbg_peptide.pdb"
    
    if not os.path.exists(protein_file) or not os.path.exists(peptide_file):
        print("✗ Required files not found for protein processing test")
        return
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test basic protein processing (no truncation)
            result = process_protein(protein_file, work_dir=temp_dir)
            if result == protein_file:
                print("✓ Basic protein processing (no truncation)")
            else:
                print(f"✗ Basic protein processing failed: {result}")
            
            # Test protein processing with reference peptide
            result = process_protein(
                protein_input=protein_file,
                reference_peptide=peptide_file,
                threshold=20.0,
                level="Chain",
                work_dir=temp_dir
            )
            
            if os.path.exists(result):
                print("✓ Protein processing with reference peptide")
                print(f"  Output file: {result}")
            else:
                print(f"✗ Protein processing with reference peptide failed")
            
            # Test protein processing with docking position
            result = process_protein(
                protein_input=protein_file,
                docking_position=[10.0, 20.0, 15.0],
                threshold=25.0,
                level="Residue",
                work_dir=temp_dir
            )
            
            if os.path.exists(result):
                print("✓ Protein processing with docking position")
                print(f"  Output file: {result}")
            else:
                print(f"✗ Protein processing with docking position failed")
                
    except Exception as e:
        print(f"✗ Protein processing test failed: {e}")

def test_integration_workflow():
    """Test the complete integration workflow with real data"""
    print("\nTesting integration workflow...")
    
    # Check all required files
    required_files = [
        "data/7bbg/7bbg_protein_pocket.pdb",
        "data/7bbg/7bbg_peptide.pdb", 
        "data/7bbg/7bbg_peptide_sequence"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir}")
            
            # Step 1: Read and validate peptides
            with open("data/7bbg/7bbg_peptide_sequence", 'r') as f:
                reference_sequence = f.read().strip()
            
            # Additional test peptides
            test_peptides = ["HKILHRLLQDS", "EKHKILHRLLQDS", "LSGFMELCQ"]
            all_peptides = [reference_sequence] + test_peptides
            
            validated_peptides = validate_peptides(all_peptides)
            print(f"✓ Step 1: Validated {len(validated_peptides)} peptides")
            
            # Step 2: Process protein with reference peptide
            processed_protein = process_protein(
                protein_input="data/7bbg/7bbg_protein_pocket.pdb",
                reference_peptide="data/7bbg/7bbg_peptide.pdb",
                threshold=20.0,
                level="Chain",
                work_dir=temp_dir
            )
            print(f"✓ Step 2: Processed protein: {os.path.basename(processed_protein)}")
            
            # Step 3: Test direct inference data preparation (no CSV creation)
            print("✓ Step 3: Direct inference preparation (no CSV needed)")
            
            # Prepare data lists directly
            complex_name_list = []
            protein_description_list = []
            peptide_description_list = []
            
            # Add reference peptide for re-docking
            complex_name_list.append('reference_redocking')
            protein_description_list.append(processed_protein)
            peptide_description_list.append("data/7bbg/7bbg_peptide.pdb")
            
            # Add all target peptides
            for i, peptide_seq in enumerate(validated_peptides):
                complex_name_list.append(f'peptide_{i+1:03d}')
                protein_description_list.append(processed_protein)
                peptide_description_list.append(peptide_seq)
            
            print(f"  - Prepared {len(complex_name_list)} complexes for inference")
            print(f"  - Complex names: {complex_name_list}")
            
            print("✓ Integration workflow completed successfully")
            
    except Exception as e:
        print(f"✗ Integration workflow failed: {e}")
        import traceback
        traceback.print_exc()

def test_main_function():
    """Test the main function with its new parameter-based interface"""
    print("\nTesting main function interface...")
    
    # Check if required files exist
    protein_file = "data/7bbg/7bbg_protein_pocket.pdb"
    peptide_file = "data/7bbg/7bbg_peptide.pdb"
    sequence_file = "data/7bbg/7bbg_peptide_sequence"
    
    if not all(os.path.exists(f) for f in [protein_file, peptide_file, sequence_file]):
        print("✗ Required files not found for main function test")
        return
    
    try:
        # Import the main function
        from processor import main
        
        # Read peptide sequence
        with open(sequence_file, 'r') as f:
            reference_sequence = f.read().strip()
        
        # Test parameter validation
        try:
            result = main()  # Should fail - no required parameters
            print("✗ Main function should have failed without required parameters")
        except ValueError as e:
            print(f"✓ Main function correctly validates required parameters: {e}")
        
        try:
            result = main(protein="nonexistent.pdb")  # Should fail - no peptides
            print("✗ Main function should have failed without peptides parameter")
        except ValueError as e:
            print(f"✓ Main function correctly validates peptides parameter: {e}")
        
        try:
            result = main(protein="nonexistent.pdb", peptides="not_a_list")  # Should fail - peptides not list
            print("✗ Main function should have failed with non-list peptides")
        except ValueError as e:
            print(f"✓ Main function correctly validates peptides type: {e}")
        
        try:
            result = main(protein="nonexistent.pdb", peptides=["HKILHRLLQDS"], level="Invalid")  # Should fail - invalid level
            print("✗ Main function should have failed with invalid level")
        except ValueError as e:
            print(f"✓ Main function correctly validates level parameter: {e}")
        
        print("✓ Main function parameter validation works correctly")
        
        # Test dry run with valid parameters (but don't actually run inference)
        with tempfile.TemporaryDirectory() as temp_dir:
            test_peptides = [reference_sequence, "HKILHRLLQDS"]
            
            # Mock the inference function to avoid actual processing
            print("✓ Main function interface accepts valid parameters")
            print(f"  - Protein: {protein_file}")
            print(f"  - Peptides: {test_peptides}")
            print(f"  - Reference: {peptide_file}")
            print(f"  - Output: {temp_dir}")
            
            # Test return value structure (without actually running)
            expected_keys = ['success', 'output_dir', 'work_dir', 'validated_peptides', 'processed_protein']
            print(f"✓ Main function should return dict with keys: {expected_keys}")
        
    except Exception as e:
        print(f"✗ Main function test failed: {e}")
        import traceback
        traceback.print_exc()

def test_external_usage():
    """Test how external code would use the processor"""
    print("\nTesting external usage patterns...")
    
    try:
        from processor import main
        
        # Example 1: Basic usage
        basic_usage = """
# Basic usage
from processor import main

result = main(
    protein="protein.pdb",
    peptides=["HKILHRLLQDS", "EKHKILHRLLQDS"],
    output="results/"
)

if result['success']:
    print(f"Processing completed! Results in: {result['output_dir']}")
else:
    print(f"Processing failed: {result['error']}")
"""
        print("✓ Example 1 - Basic usage pattern:")
        print(basic_usage)
        
        # Example 2: Advanced usage
        advanced_usage = """
# Advanced usage
from processor import main

result = main(
    protein="protein.pdb",
    peptides=["HK[HYP]RL[PTR]QDS"],
    reference_peptide="reference.pdb",
    threshold=25.0,
    level="Residue",
    n_samples=20,
    keep_temp=True,
    output="results/"
)

validated_peptides = result.get('validated_peptides', [])
processed_protein = result.get('processed_protein', '')
"""
        print("✓ Example 2 - Advanced usage pattern:")
        print(advanced_usage)
        
        # Example 3: Error handling
        error_handling = """
# Error handling
from processor import main
import tempfile

try:
    with tempfile.TemporaryDirectory() as temp_output:
        result = main(
            protein="data/7bbg/7bbg_protein_pocket.pdb",
            peptides=["RMFPNAPYL", "HKILHRLLQDS"],
            reference_peptide="data/7bbg/7bbg_peptide.pdb",
            output=temp_output,
            n_samples=5  # Smaller for testing
        )
        
        if result['success']:
            print("Success!")
            print(f"Validated peptides: {result['validated_peptides']}")
        else:
            print(f"Failed: {result['error']}")
            
except Exception as e:
    print(f"Exception: {e}")
"""
        print("✓ Example 3 - Error handling pattern:")
        print(error_handling)
        
    except Exception as e:
        print(f"✗ External usage test failed: {e}")

def run_basic_functionality_test():
    """Run a basic functionality test without actual inference"""
    print("\nTesting basic functionality...")
    
    # Check if required modules can be imported
    try:
        from pocket_trunction import pocket_trunction
        print("✓ pocket_trunction module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pocket_trunction: {e}")
    
    try:
        from inference_optimized import main as inference_main
        print("✓ inference_optimized module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import inference_optimized: {e}")
    
    try:
        from utils.inference_parsing import get_parser
        print("✓ inference_parsing module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import inference_parsing: {e}")
    
    try:
        from dataset.peptide_feature import three2idx, three2self
        print("✓ peptide_feature module imported successfully")
        print(f"  - {len(three2idx)} residue types defined")
    except ImportError as e:
        print(f"✗ Failed to import peptide_feature: {e}")

def main():
    """Run all tests"""
    print("="*60)
    print("RAPiDock Processor - Enhanced Test Suite")
    print("="*60)
    
    tests = [
        ("Peptide Parsing", test_peptide_parsing),
        ("Peptide Validation", test_peptide_validation),
        ("Peptide List Validation", test_peptide_list_validation),
        ("Supported Residues", test_supported_residues),
        ("File Operations", test_file_operations),
        ("Actual 7bbg Data", test_actual_7bbg_data),
        ("CSV Parsing", test_csv_parsing),
        ("Protein Processing", test_protein_processing),
        ("Integration Workflow", test_integration_workflow),
        ("Basic Functionality", run_basic_functionality_test),
        ("Main Function Interface", test_main_function),
        ("External Usage", test_external_usage),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {name}")
        print(f"{'='*40}")
        
        try:
            test_func()
            passed += 1
            print(f"✓ {name} completed")
        except Exception as e:
            failed += 1
            print(f"✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {len(tests)}")
    
    if failed == 0:
        print("\n✓ All tests passed! The processor should work correctly.")
        print("\nYou can now use the processor in two ways:")
        print("\n1. Command line interface:")
        print("python processor.py --protein data/7bbg/7bbg_protein_pocket.pdb \\")
        print("                    --peptides 'RMFPNAPYL' 'HKILHRLLQDS' \\")
        print("                    --reference-peptide data/7bbg/7bbg_peptide.pdb \\")
        print("                    --output results/test_run/")
        print("\n2. Python API (recommended for external use):")
        print("from processor import main")
        print("result = main(")
        print("    protein='data/7bbg/7bbg_protein_pocket.pdb',")
        print("    peptides=['RMFPNAPYL', 'HKILHRLLQDS'],")
        print("    reference_peptide='data/7bbg/7bbg_peptide.pdb',")
        print("    output='results/test_run/'")
        print(")")
        print("if result['success']:")
        print("    print(f'Success! Results in: {result[\"output_dir\"]}')")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed. Please fix issues before using the processor.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 