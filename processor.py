import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np
import tempfile
import shutil
import subprocess
from typing import List, Tuple, Optional, Union
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO
import re
import warnings
import traceback

# Try to import MDAnalysis for complex structure creation
try:
    import MDAnalysis
    MDANALYSIS_AVAILABLE = True
except ImportError:
    print("Warning: MDAnalysis not available. Complex structure creation will be limited.")
    MDANALYSIS_AVAILABLE = False

import os
import torch
import yaml
from argparse import Namespace
from torch_geometric.loader import DataListLoader
from argparse import Namespace
import yaml

# Import RAPiDock modules
from .utils.inference_utils import InferenceDataset
from .utils.utils import get_model, ExponentialMovingAverage
from .pocket_trunction import pocket_trunction
from .inference_optimized import main as inference_main
from .utils.inference_parsing import get_parser
from .dataset.peptide_feature import three2idx, three2self
from .inference_optimized import main_sequential_with_data
from .inference_optimized import main_optimized_with_data

warnings.filterwarnings("ignore")

# Supported residue types from RAPiDock
SUPPORTED_RESIDUES = set(three2idx.keys()) - {'X'}  # Remove 'X' as it's unknown

# Standard amino acids (single letter to three letter mapping)
STANDARD_AA = {
    'G': 'GLY', 'A': 'ALA', 'V': 'VAL', 'L': 'LEU', 'I': 'ILE', 'P': 'PRO',
    'F': 'PHE', 'Y': 'TYR', 'W': 'TRP', 'S': 'SER', 'T': 'THR', 'C': 'CYS',
    'M': 'MET', 'N': 'ASN', 'Q': 'GLN', 'D': 'ASP', 'E': 'GLU', 'K': 'LYS',
    'R': 'ARG', 'H': 'HIS'
}

def classify_capri_peptide(fnat, iRMS, lRMS):
    """
    Classify peptide docking results according to CAPRI official criteria.
    Borrowed from rmsd_bound_pyrosetta.py
    
    CAPRI overall model quality classification (applied in reverse order):
    - High quality: (fnat ≥ 0.5) and either (L-RMSD ≤ 1.0 Å) or (i-RMSDbb ≤ 1.0 Å)
    - Medium quality: (fnat ≥ 0.3) and either (L-RMSD ≤ 5.0 Å) or (i-RMSDbb ≤ 2.0 Å)
    - Acceptable quality: (fnat ≥ 0.1) and either (L-RMSD ≤ 10.0 Å) or (i-RMSDbb ≤ 4.0 Å)
    - Incorrect: (fnat < 0.1) or (L-RMSD > 10.0 Å) and (i-RMSDbb > 4.0 Å)
    
    Args:
        fnat: Fraction of native contacts (string or float)
        iRMS: Interface RMSD (i-RMSDbb) (string or float)
        lRMS: Ligand RMSD (L-RMSD) (string or float)  
        
    Returns:
        str: Classification ('High', 'Medium', 'Acceptable', 'Incorrect')
    """
    try:
        # Convert string values to float, handle None values
        fnat_val = float(fnat) if fnat is not None and str(fnat) != 'None' else 0.0
        iRMS_val = float(iRMS) if iRMS is not None and str(iRMS) != 'None' else float('inf')
        lRMS_val = float(lRMS) if lRMS is not None and str(lRMS) != 'None' else float('inf')  
        
        # Apply CAPRI criteria in reverse order as specified
        
        # First check for Incorrect (applied first in reverse order)
        if (fnat_val < 0.1) or (lRMS_val > 10.0 and iRMS_val > 4.0):
            return 'Incorrect'
        
        # Then check for Acceptable
        elif (fnat_val >= 0.1) and (lRMS_val <= 10.0 or iRMS_val <= 4.0):
            # Check if it meets higher quality criteria
            
            # Check for Medium quality
            if (fnat_val >= 0.3) and (lRMS_val <= 5.0 or iRMS_val <= 2.0):
                # Check for High quality
                if (fnat_val >= 0.5) and (lRMS_val <= 1.0 or iRMS_val <= 1.0):
                    return 'High'
                else:
                    return 'Medium'
            else:
                return 'Acceptable'
        
        # Default to Incorrect if none of the above conditions are met
        else:
            return 'Incorrect'
            
    except (ValueError, TypeError):
        # If any conversion fails, classify as Incorrect
        return 'Incorrect'

def parse_peptide_sequence(sequence: str) -> List[str]:
    """
    Parse peptide sequence and convert to three-letter codes.
    Supports both single-letter codes and [XXX] format for non-canonical amino acids.
    
    Args:
        sequence: Peptide sequence string
        
    Returns:
        List of three-letter residue codes
        
    Raises:
        ValueError: If sequence contains unsupported residues
    """
    residues = []
    i = 0
    
    while i < len(sequence):
        if sequence[i] == '[':
            # Find closing bracket
            end = sequence.find(']', i)
            if end == -1:
                raise ValueError(f"Unclosed bracket in sequence: {sequence[i:]}")
            
            # Extract three-letter code
            three_letter = sequence[i+1:end]
            if three_letter not in SUPPORTED_RESIDUES:
                raise ValueError(f"Unsupported residue: {three_letter}")
            
            residues.append(three_letter)
            i = end + 1
        else:
            # Single letter code
            single_letter = sequence[i].upper()
            if single_letter not in STANDARD_AA:
                raise ValueError(f"Unsupported single-letter residue: {single_letter}")
            
            three_letter = STANDARD_AA[single_letter]
            if three_letter not in SUPPORTED_RESIDUES:
                raise ValueError(f"Unsupported residue: {three_letter}")
            
            residues.append(three_letter)
            i += 1
    
    return residues

def validate_peptide_length(sequence: str, min_length: int = 3, max_length: int = 50) -> bool:
    """
    Validate peptide length constraints.
    
    Args:
        sequence: Peptide sequence
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Returns:
        True if length is valid
    """
    residues = parse_peptide_sequence(sequence)
    length = len(residues)
    
    if length < min_length:
        raise ValueError(f"Peptide too short: {length} residues (minimum: {min_length})")
    if length > max_length:
        raise ValueError(f"Peptide too long: {length} residues (maximum: {max_length})")
    
    return True

def validate_peptides(peptide_sequences: List[str]) -> List[str]:
    """
    Validate all peptide sequences.
    
    Args:
        peptide_sequences: List of peptide sequences
        
    Returns:
        List of validated peptide sequences
    """
    validated = []
    
    for i, seq in enumerate(peptide_sequences):
        try:
            validate_peptide_length(seq)
            parse_peptide_sequence(seq)  # Will raise error if invalid
            validated.append(seq)
            print(f"✓ Peptide {i+1}: {seq} - Valid")
        except ValueError as e:
            print(f"✗ Peptide {i+1}: {seq} - Invalid: {e}")
            continue
    
    return validated

def create_protein_pdb_from_sequence(sequence: str, output_path: str) -> str:
    """
    Create a PDB file from protein sequence using ESMFold (placeholder).
    In practice, this would call ESMFold or another structure prediction method.
    
    Args:
        sequence: Protein sequence
        output_path: Output PDB file path
        
    Returns:
        Path to created PDB file
    """
    # This is a placeholder - in practice you would call ESMFold
    print(f"Warning: Protein sequence provided but structure prediction not implemented")
    print(f"Please provide a PDB file instead of sequence: {sequence[:50]}...")
    raise NotImplementedError("Protein structure prediction from sequence not implemented")

def process_protein(protein_input: str, 
                   reference_peptide: Optional[str] = None,
                   docking_position: Optional[List[float]] = None,
                   threshold: float = 20.0,
                   level: str = "Chain",
                   work_dir: str = None) -> str:
    """
    Process protein input and perform pocket truncation if needed.
    
    Args:
        protein_input: Protein PDB file path or sequence string
        reference_peptide: Reference peptide PDB file path (for pocket truncation)
        docking_position: XYZ coordinates for docking center [x, y, z]
        threshold: Distance threshold for pocket truncation
        level: Truncation level ("Chain", "Residue", "Atom")
        work_dir: Working directory for temporary files
        
    Returns:
        Path to processed protein PDB file
    """
    # Check if input is a file path or sequence
    if os.path.exists(protein_input):
        protein_pdb = protein_input
        print(f"Using protein PDB file: {protein_pdb}")
    else:
        # Assume it's a sequence
        if work_dir is None:
            work_dir = tempfile.mkdtemp()
        
        protein_pdb = os.path.join(work_dir, "protein_from_sequence.pdb")
        protein_pdb = create_protein_pdb_from_sequence(protein_input, protein_pdb)
    
    # Perform pocket truncation if reference peptide or docking position provided
    if reference_peptide is not None or docking_position is not None:
        if work_dir is None:
            work_dir = tempfile.mkdtemp()
        
        truncated_pdb = os.path.join(work_dir, "protein_pocket.pdb")
        
        print(f"Performing pocket truncation...")
        print(f"  Threshold: {threshold} Å")
        print(f"  Level: {level}")
        
        if reference_peptide:
            print(f"  Reference peptide: {reference_peptide}")
        if docking_position:
            print(f"  Docking position: {docking_position}")
        
        # Call pocket truncation
        total_residues, pocket_residues = pocket_trunction(
            protein=protein_pdb,
            peptide=reference_peptide,
            threshold=threshold,
            save_name=truncated_pdb,
            xyz=docking_position,
            level=level,
            exclude_chain=None,
            threshold_keep=5.0
        )
        
        print(f"  Original residues: {total_residues}")
        print(f"  Pocket residues: {pocket_residues}")
        
        return truncated_pdb
    
    return protein_pdb



def create_complex_structure(protein_pdb: str, peptide_pdb: str, output_path: str, peptide_chain: str = 'p') -> bool:
    """
    Create a complex structure by merging protein and peptide PDB files.
    Borrowed approach from rmsd_bound_pyrosetta.py
    
    Args:
        protein_pdb: Path to protein PDB file
        peptide_pdb: Path to peptide PDB file  
        output_path: Path for output complex PDB file
        peptide_chain: Chain identifier for peptide (default: 'p')
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not MDANALYSIS_AVAILABLE:
        print("MDAnalysis not available for complex structure creation")
        return False
        
    try:
        # Load protein structure
        protein = MDAnalysis.Universe(protein_pdb)
        
        # Assign chain IDs to protein segments
        for idx, seg in enumerate(protein.segments):
            chain_id = chr(ord('A') + idx)
            seg.segid = chain_id
            for atom in seg.atoms:
                atom.chainID = chain_id
        
        # Load peptide structure
        peptide = MDAnalysis.Universe(peptide_pdb)
        peptide.segments[0].segid = peptide_chain
        for atom in peptide.atoms:
            atom.chainID = peptide_chain
            
        # Renumber peptide residues to start from 1
        peptide.residues.resids = peptide.residues.resids - peptide.residues.resids.min() + 1
        
        # Merge protein and peptide
        complex_structure = MDAnalysis.Merge(protein.atoms, peptide.atoms)
        
        # Write complex structure
        complex_structure.atoms.write(output_path)
        
        print(f"Successfully created complex structure: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating complex structure: {e}")
        return False

def run_dockq_evaluation(model_pdb: str, native_pdb: str, protein_pdb: str, native_chain: str = 'p') -> dict:
    """
    Run DockQ evaluation between model and native structures.
    Fixed version following rmsd_bound_pyrosetta.py approach.
    
    Args:
        model_pdb: Path to model peptide PDB file (predicted peptide only)
        native_pdb: Path to native peptide PDB file (reference peptide only)
        protein_pdb: Path to protein PDB file
        native_chain: Chain identifier for the native peptide (default: 'p')
        
    Returns:
        Dictionary with DockQ results including CAPRI classification
    """
    # Get the current script directory and construct relative path to DockQ
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dockq_script = os.path.join(script_dir, "tools", "DockQ-1.0", "DockQ.py")
    
    if not os.path.exists(dockq_script):
        print(f"Warning: DockQ script not found at {dockq_script}")
        return {}
    
    if not MDANALYSIS_AVAILABLE:
        print("Warning: MDAnalysis not available for DockQ evaluation")
        return {}
    
    try:
        # Load protein structure
        protein = MDAnalysis.Universe(protein_pdb)
        
        # Assign chain IDs to protein segments
        for idx, seg in enumerate(protein.segments):
            chain_id = chr(ord('A') + idx)
            seg.segid = chain_id
            for atom in seg.atoms:
                atom.chainID = chain_id
        
        # Create native complex (protein + native peptide)
        peptide_native = MDAnalysis.Universe(native_pdb)
        peptide_native.segments[0].segid = native_chain
        for atom in peptide_native.atoms:
            atom.chainID = native_chain
        peptide_native.residues.resids = peptide_native.residues.resids - peptide_native.residues.resids.min() + 1
        
        peptide_native_merge = MDAnalysis.Merge(protein.atoms, peptide_native.atoms)
        native_complex_path = native_pdb.replace('.pdb', '_complex_native.pdb')
        peptide_native_merge.atoms.write(native_complex_path)
        print(f"Created native complex: {native_complex_path}")
        
        # Create model complex (protein + model peptide)
        peptide_model = MDAnalysis.Universe(model_pdb)
        peptide_model.segments[0].segid = native_chain
        for atom in peptide_model.atoms:
            atom.chainID = native_chain
        
        peptide_model_merge = MDAnalysis.Merge(protein.atoms, peptide_model.atoms)
        model_complex_path = model_pdb.replace('.pdb', '_complex_model.pdb')
        peptide_model_merge.atoms.write(model_complex_path)
        print(f"Created model complex: {model_complex_path}")
        
        # Run DockQ with proper parameters for peptide docking
        cmd = f"python {dockq_script} {model_complex_path} {native_complex_path} -native_chain1 {native_chain} -no_needle"
        print(f"Running DockQ command: {cmd}")
        
        # Use os.popen like in rmsd_bound_pyrosetta.py for better output handling
        result = os.popen(cmd)
        
        # Initialize variables
        fnat = None
        iRMS = None
        lRMS = None
        DockQ = None
        
        # Parse DockQ output line by line
        for line in result.readlines():
            line = line.strip()
            print(f"DockQ output: {line}")
            
            if line.startswith('Fnat'):
                fnat = line.split()[1]
            elif line.startswith('iRMS'):
                iRMS = line.split()[1]
            elif line.startswith('LRMS'):
                lRMS = line.split()[1]
            elif line.startswith('DockQ'):
                DockQ = line.split()[1]
        
        result.close()
        
        # Calculate CAPRI-peptide classification
        capri_class = classify_capri_peptide(fnat, iRMS, lRMS)
        
        # Prepare results dictionary
        dockq_results = {
            'DockQ': float(DockQ) if DockQ is not None else None,
            'Fnat': float(fnat) if fnat is not None else None,
            'iRMS': float(iRMS) if iRMS is not None else None,
            'LRMS': float(lRMS) if lRMS is not None else None,
            'CAPRI_class': capri_class
        }
        
        # Clean up temporary files
        try:
            os.remove(model_complex_path)
            os.remove(native_complex_path)
        except:
            pass
        
        return dockq_results
    
    except Exception as e:
        print(f"Error running DockQ: {e}")
        traceback.print_exc()
        return {}

def run_dockq_evaluation_all_ranks(result_path: str, native_pdb: str, protein_pdb: str, native_chain: str = 'p') -> dict:
    """
    Run DockQ evaluation for all ranked models and return the best result.
    
    Args:
        result_path: Path to result directory containing ranked models
        native_pdb: Path to native peptide PDB file (reference peptide only)
        protein_pdb: Path to protein PDB file
        native_chain: Chain identifier for the native peptide (default: 'p')
        
    Returns:
        Dictionary with best DockQ results including CAPRI classification and rank info
    """
    import glob
    
    # Find all ranked model files
    rank_files = glob.glob(os.path.join(result_path, "rank*_ref2015.pdb"))
    
    if not rank_files:
        print(f"  No ranked model files found in {result_path}")
        return {}
    
    print(f"  Found {len(rank_files)} ranked models to evaluate")
    
    best_result = None
    best_dockq = -1.0  # DockQ scores range from 0 to 1, so -1 is worst possible
    best_rank = None
    all_results = []
    
    for rank_file in sorted(rank_files, key=lambda x: int(os.path.basename(x).split('rank')[1].split('_')[0])):
        rank_num = int(os.path.basename(rank_file).split('rank')[1].split('_')[0])
        print(f"    Evaluating rank {rank_num}...")
        
        # Run DockQ evaluation for this rank
        dockq_result = run_dockq_evaluation(rank_file, native_pdb, protein_pdb, native_chain)
        
        if dockq_result and 'DockQ' in dockq_result and dockq_result['DockQ'] is not None:
            current_dockq = dockq_result['DockQ']
            dockq_result['rank'] = rank_num
            dockq_result['model_file'] = rank_file
            all_results.append(dockq_result)
            
            print(f"      Rank {rank_num}: DockQ = {current_dockq:.3f}, CAPRI = {dockq_result.get('CAPRI_class', 'N/A')}")
            
            # Check if this is the best result so far
            if current_dockq > best_dockq:
                best_dockq = current_dockq
                best_result = dockq_result.copy()
                best_rank = rank_num
        else:
            print(f"      Rank {rank_num}: DockQ evaluation failed")
    
    if best_result:
        print(f"  Best result: Rank {best_rank} with DockQ = {best_dockq:.3f}")
        best_result['best_rank'] = best_rank
        best_result['total_ranks_evaluated'] = len(all_results)
        best_result['all_rank_results'] = all_results
        return best_result
    else:
        print(f"  No valid DockQ results obtained")
        return {}

def process_results(output_dir: str, 
                   reference_peptide: Optional[str] = None,
                   protein_pdb: Optional[str] = None,
                   work_dir: str = None,
                   peptide_sequences: Optional[List[str]] = None) -> dict:
    """
    Process and analyze results, including DockQ evaluation for reference peptide.
    
    Args:
        output_dir: Directory containing inference results
        reference_peptide: Reference peptide PDB file for DockQ evaluation
        protein_pdb: Protein PDB file for creating complex structures
        work_dir: Working directory
        peptide_sequences: List of peptide sequences used for inference
        
    Returns:
        Dictionary with detailed results including scores, files, and evaluation data
    """
    print(f"\n{'='*60}")
    print("PROCESSING RESULTS")
    print(f"{'='*60}")
    
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return {
            'success': False,
            'error': f"Output directory not found: {output_dir}",
            'results': []
        }
    
    # Find all result directories
    result_dirs = [d for d in os.listdir(output_dir) 
                   if os.path.isdir(os.path.join(output_dir, d))]
    
    summary_data = []
    detailed_results = []
    
    # Create peptide sequence mapping
    peptide_mapping = {}
    if peptide_sequences:
        for i, seq in enumerate(peptide_sequences):
            peptide_mapping[f'peptide_{i+1:03d}'] = seq
    
    for result_dir in sorted(result_dirs):
        result_path = os.path.join(output_dir, result_dir)
        print(f"\nProcessing results for: {result_dir}")
        
        # Initialize result structure for this complex
        complex_result = {
            'complex_name': result_dir,
            'result_path': result_path,
            'files': {
                'pdb_files': [],
                'score_files': [],
                'other_files': []
            },
            'scores': {
                'rosetta_scores': [],
                'best_rosetta_score': None,
                'dockq_results': None
            },
            'evaluation': {
                'dockq_score': None,
                'fnat': None,
                'irms': None,
                'lrms': None,
                'capri_class': None,
                'evaluation_type': 'none'
            },
            'peptide_info': {
                'sequence': peptide_mapping.get(result_dir, None),
                'is_reference': result_dir == 'reference_redocking'
            }
        }
        
        # Collect all files in the result directory
        if os.path.exists(result_path):
            for file in os.listdir(result_path):
                file_path = os.path.join(result_path, file)
                if file.endswith('.pdb'):
                    complex_result['files']['pdb_files'].append(file_path)
                elif file.endswith('.csv'):
                    complex_result['files']['score_files'].append(file_path)
                else:
                    complex_result['files']['other_files'].append(file_path)
        
        # Check for Rosetta scores
        score_file = os.path.join(result_path, "ref2015_score.csv")
        if os.path.exists(score_file):
            try:
                scores_df = pd.read_csv(score_file)
                
                # Enhance the scores with full file paths to match final_pdb_files
                enhanced_scores = []
                for _, row in scores_df.iterrows():
                    score_record = row.to_dict()
                    # Convert the simple filename to full path with .pdb extension
                    filename = score_record['file']
                    if not filename.endswith('.pdb'):
                        filename = f"{filename}.pdb"
                    full_path = os.path.join(result_path, filename)
                    score_record['file'] = full_path
                    score_record['filename'] = filename  # Keep original filename for compatibility
                    enhanced_scores.append(score_record)
                
                # Store enhanced scores
                complex_result['scores']['rosetta_scores'] = enhanced_scores
                best_score = scores_df.iloc[0]['ref2015score']
                complex_result['scores']['best_rosetta_score'] = best_score
                complex_result['evaluation']['evaluation_type'] = 'rosetta_only'
                
                print(f"  Best Rosetta REF2015 score: {best_score}")
                
                summary_data.append({
                    'complex_name': result_dir,
                    'best_rosetta_score': best_score,
                    'dockq_score': None,
                    'fnat': None,
                    'irms': None,
                    'lrms': None,
                    'capri_class': None,
                    'evaluation_type': 'rosetta_only'
                })
            except Exception as e:
                print(f"  Error reading Rosetta scores: {e}")
                complex_result['scores']['error'] = str(e)
        
        # Run DockQ evaluation for reference peptide re-docking
        if result_dir == 'reference_redocking' and reference_peptide and protein_pdb:
            print(f"  Running DockQ evaluation against reference for all ranks...")
            
            # Evaluate all ranked models and get the best result
            dockq_results = run_dockq_evaluation_all_ranks(result_path, reference_peptide, protein_pdb)
            
            if dockq_results:
                # Store DockQ results in complex_result
                complex_result['scores']['dockq_results'] = dockq_results
                complex_result['evaluation'].update({
                    'dockq_score': dockq_results.get('DockQ'),
                    'fnat': dockq_results.get('Fnat'),
                    'irms': dockq_results.get('iRMS'),
                    'lrms': dockq_results.get('LRMS'),
                    'capri_class': dockq_results.get('CAPRI_class'),
                    'best_rank': dockq_results.get('best_rank'),
                    'total_ranks_evaluated': dockq_results.get('total_ranks_evaluated'),
                    'evaluation_type': 'dockq_and_rosetta'
                })
                
                print(f"  Best DockQ Score: {dockq_results.get('DockQ', 'N/A')} (from Rank {dockq_results.get('best_rank', 'N/A')})")
                print(f"  Fnat: {dockq_results.get('Fnat', 'N/A')}")
                print(f"  iRMS: {dockq_results.get('iRMS', 'N/A')} Å")
                print(f"  LRMS: {dockq_results.get('LRMS', 'N/A')} Å")
                print(f"  CAPRI Class: {dockq_results.get('CAPRI_class', 'N/A')}")
                print(f"  Evaluated {dockq_results.get('total_ranks_evaluated', 0)} ranks")
                
                # Update summary data
                for item in summary_data:
                    if item['complex_name'] == result_dir:
                        item.update({
                            'dockq_score': dockq_results.get('DockQ'),
                            'fnat': dockq_results.get('Fnat'),
                            'irms': dockq_results.get('iRMS'),
                            'lrms': dockq_results.get('LRMS'),
                            'capri_class': dockq_results.get('CAPRI_class'),
                            'best_rank': dockq_results.get('best_rank'),
                            'total_ranks_evaluated': dockq_results.get('total_ranks_evaluated'),
                            'evaluation_type': 'dockq_and_rosetta'
                        })
                        break
            else:
                print(f"  DockQ evaluation failed for all ranks")
                complex_result['evaluation']['dockq_error'] = "DockQ evaluation failed for all ranks"
        
        # Add complex result to detailed results
        detailed_results.append(complex_result)
    
    # Save summary
    summary_file = None
    if summary_data:
        summary_file = os.path.join(output_dir, "evaluation_summary.csv")
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")
        
        # Display summary table
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(summary_df.to_string(index=False))
    
    # Return comprehensive results
    return {
        'success': True,
        'output_dir': output_dir,
        'summary_file': summary_file,
        'summary_data': summary_data,
        'detailed_results': detailed_results,
        'total_complexes': len(detailed_results),
        'complexes_with_scores': len([r for r in detailed_results if r['scores']['best_rosetta_score'] is not None]),
        'complexes_with_dockq': len([r for r in detailed_results if r['evaluation']['dockq_score'] is not None])
    }

def run_inference_directly(protein_pdb: str,
                         peptide_sequences: List[str],
                         reference_peptide: Optional[str] = None,
                         output_dir: str = None,
                         config: str = "default_inference_args.yaml",
                         n_samples: int = 10,
                         batch_size: int = 40,
                         cpu: int = 16) -> None:
    """
    Run inference directly without creating CSV files.
    
    Args:
        protein_pdb: Path to protein PDB file
        peptide_sequences: List of peptide sequences
        reference_peptide: Optional reference peptide (for re-docking)
        output_dir: Output directory
        config: Configuration file path
        n_samples: Number of samples per peptide
        batch_size: Batch size for inference
        cpu: Number of CPU cores
    """
    
    # Prepare data lists directly in memory
    complex_name_list = []
    protein_description_list = []
    peptide_description_list = []
    
    # Add reference peptide for re-docking if provided
    if reference_peptide:
        complex_name_list.append('reference_redocking')
        protein_description_list.append(protein_pdb)
        peptide_description_list.append(reference_peptide)
    
    # Add all target peptides
    for i, peptide_seq in enumerate(peptide_sequences):
        complex_name_list.append(f'peptide_{i+1:03d}')
        protein_description_list.append(protein_pdb)
        peptide_description_list.append(peptide_seq)
    
    print(f"Prepared {len(complex_name_list)} complexes for inference")
    
    # Create inference arguments directly
    inference_args = Namespace()
    
    # Load config file
    with open(config, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    # Set default values first
    inference_args.config = None  # We'll load config manually
    inference_args.protein_peptide_csv = None  # We're not using CSV
    inference_args.complex_name = None
    inference_args.protein_description = None
    inference_args.peptide_description = None
    inference_args.output_dir = output_dir
    inference_args.save_visualisation = False
    inference_args.N = n_samples
    inference_args.model_dir = None
    inference_args.ckpt = None
    inference_args.scoring_function = None
    inference_args.fastrelax = False
    inference_args.confidence_model_dir = None
    inference_args.confidence_ckpt = None
    inference_args.batch_size = batch_size
    inference_args.no_final_step_noise = False
    inference_args.inference_steps = None
    inference_args.actual_steps = None
    inference_args.conformation_partial = None
    inference_args.conformation_type = "H"
    inference_args.cpu = cpu
    
    # Update with config values
    for key, value in config_dict.items():
        setattr(inference_args, key, value)
    
    # Create a custom inference function that uses our prepared data
    run_inference_with_data(
        inference_args,
        complex_name_list,
        protein_description_list,
        peptide_description_list
    )

def run_inference_with_data(args, complex_name_list, protein_description_list, peptide_description_list):
    """
    Run inference with prepared data lists instead of CSV file.
    """

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model parameters
    with open(f"{args.model_dir}/model_parameters.yml") as f:
        score_model_args = Namespace(**yaml.full_load(f))
    
    # Create output directories
    for name in complex_name_list:
        write_dir = f"{args.output_dir}/{name}"
        os.makedirs(write_dir, exist_ok=True)
    
    # Create inference dataset directly with our data
    inference_dataset = InferenceDataset(
        output_dir=args.output_dir,
        complex_name_list=complex_name_list,
        protein_description_list=protein_description_list,
        peptide_description_list=peptide_description_list,
        lm_embeddings=score_model_args.esm_embeddings_path_train is not None,
        lm_embeddings_pep=score_model_args.esm_embeddings_peptide_train is not None,
        conformation_type=args.conformation_type,
        conformation_partial=args.conformation_partial,
    )
    
    # Create data loader
    inference_loader = DataListLoader(
        dataset=inference_dataset, batch_size=1, shuffle=False
    )
    
    # Load main model
    model = get_model(score_model_args, no_parallel=True)
    state_dict = torch.load(
        f"{args.model_dir}/{args.ckpt}", map_location=torch.device("cpu")
    )
    model.load_state_dict(state_dict["model"], strict=True)
    model = model.to(device)
    
    ema_weights = ExponentialMovingAverage(
        model.parameters(), decay=score_model_args.ema_rate
    )
    ema_weights.load_state_dict(state_dict["ema_weights"], device=device)
    ema_weights.copy_to(model.parameters())
    
    # Load confidence model if needed
    confidence_model = None
    if args.scoring_function == "confidence":
        with open(f"{args.confidence_model_dir}/model_parameters.yml") as f:
            confidence_args = Namespace(**yaml.full_load(f))

        confidence_model = get_model(
            confidence_args, no_parallel=True, confidence_mode=True
        )
        state_dict = torch.load(
            f"{args.confidence_model_dir}/{args.confidence_ckpt}",
            map_location=torch.device("cpu"),
        )
        confidence_model.load_state_dict(state_dict["model"], strict=True)
        confidence_model = confidence_model.to(device)
        confidence_model.eval()
    
    # Run inference using the optimized version
    if hasattr(args, 'disable_parallel_processing') and args.disable_parallel_processing:
        main_sequential_with_data(args, inference_dataset, inference_loader, model, confidence_model, score_model_args)
    else:
        main_optimized_with_data(args, inference_dataset, inference_loader, model, confidence_model, score_model_args)

def main(protein=None,
         peptides=None,
         reference_peptide=None,
         docking_position=None,
         threshold=12.0,
         level="Residue",
         min_length=3,
         max_length=50,
         output="results/",
         work_dir=None,
         keep_temp=False,
         config="/workdir/default_inference_args.yaml",
         n_samples=10,
         batch_size=40,
         cpu=16
    ):
    """
    Main entry point for the protein-peptide docking processor.
    
    Args:
        protein: Protein PDB file path or sequence string
        peptides: List of peptide sequences (supports both single-letter codes and [XXX] format)
        reference_peptide: Reference peptide PDB file for pocket truncation and DockQ evaluation
        docking_position: XYZ coordinates for docking center (for pocket truncation) as [x, y, z]
        threshold: Distance threshold for pocket truncation (default: 20.0 Å)
        level: Pocket truncation level (default: "Chain", choices: "Chain", "Residue", "Atom")
        min_length: Minimum peptide length (default: 3)
        max_length: Maximum peptide length (default: 50)
        output: Output directory for results (default: "results/")
        work_dir: Working directory for temporary files (default: auto-generated)
        keep_temp: Keep temporary files after processing (default: False)
        config: Configuration file for inference (default: "default_inference_args.yaml")
        n_samples: Number of samples to generate per peptide (default: 10)
        batch_size: Batch size for inference (default: 40)
        cpu: Number of CPU cores to use (default: 16)
        
    Returns:
        dict: Results summary with success status and output directory
        
    Examples:
        # Basic usage with protein PDB and peptide sequences
        main(protein="protein.pdb", 
             peptides=["HKILHRLLQDS", "EKHKILHRLLQDS"], 
             output="results/")

        # With reference peptide for pocket truncation and DockQ evaluation
        main(protein="protein.pdb", 
             peptides=["HKILHRLLQDS", "EKHKILHRLLQDS"],
             reference_peptide="reference.pdb", 
             output="results/")

        # With docking position for pocket truncation
        main(protein="protein.pdb", 
             peptides=["HKILHRLLQDS"],
             docking_position=[10.5, 20.3, 15.7], 
             output="results/")

        # Advanced options
        main(protein="protein.pdb", 
             peptides=["HK[HYP]RL[PTR]QDS"],
             reference_peptide="reference.pdb", 
             threshold=25.0, 
             level="Residue",
             config="custom_config.yaml", 
             output="results/")
    """
    
    # Validate required parameters
    if protein is None:
        raise ValueError("protein parameter is required")
    if peptides is None:
        raise ValueError("peptides parameter is required")
    if not isinstance(peptides, list):
        raise ValueError("peptides must be a list of sequences")
    
    # Validate level parameter
    if level not in ["Chain", "Residue", "Atom"]:
        raise ValueError(f"level must be one of 'Chain', 'Residue', 'Atom', got: {level}")
    
    print(f"{'='*60}")
    print("RAPiDock Processor - Protein-Peptide Docking")
    print(f"{'='*60}")
    
    # Create working directory
    if work_dir:
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir = tempfile.mkdtemp(prefix="rapidock_processor_")
    
    print(f"Working directory: {work_dir}")
    
    try:
        # Step 1: Validate peptides
        print(f"\nStep 1: Validating {len(peptides)} peptide sequences...")
        validated_peptides = validate_peptides(peptides)
        
        if not validated_peptides:
            raise ValueError("No valid peptides found!")
        
        print(f"✓ {len(validated_peptides)} peptides validated")
        
        # Step 2: Process protein
        print(f"\nStep 2: Processing protein...")
        processed_protein = process_protein(
            protein_input=protein,
            reference_peptide=reference_peptide,
            docking_position=docking_position,
            threshold=threshold,
            level=level,
            work_dir=work_dir
        )
        print(f"✓ Protein processed: {processed_protein}")
        
        # Step 3: Run inference directly (no CSV creation)
        print(f"\nStep 3: Running RAPiDock inference directly...")
        
        run_inference_directly(
            protein_pdb=processed_protein,
            peptide_sequences=validated_peptides,
            reference_peptide=reference_peptide,
            output_dir=output,
            config=config,
            n_samples=n_samples,
            batch_size=batch_size,
            cpu=cpu
        )
        
        # Step 4: Process results
        print(f"\nStep 4: Processing results...")
        results_data = process_results(
            output_dir=output,
            reference_peptide=reference_peptide,
            protein_pdb=processed_protein,
            work_dir=work_dir,
            peptide_sequences=validated_peptides
        )
        
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Results directory: {output}")
        print(f"Working directory: {work_dir}")
        
        # Prepare detailed peptide information
        peptide_info = []
        for i, seq in enumerate(validated_peptides):
            try:
                parsed_residues = parse_peptide_sequence(seq)
                peptide_info.append({
                    'index': i + 1,
                    'sequence': seq,
                    'parsed_residues': parsed_residues,
                    'length': len(parsed_residues),
                    'complex_name': f'peptide_{i+1:03d}'
                })
            except Exception as e:
                peptide_info.append({
                    'index': i + 1,
                    'sequence': seq,
                    'parsed_residues': None,
                    'length': None,
                    'complex_name': f'peptide_{i+1:03d}',
                    'error': str(e)
                })
        
        # Prepare protein information
        protein_info = {
            'input': protein,
            'processed_file': processed_protein,
            'is_sequence': not os.path.exists(protein),
            'pocket_truncation': {
                'performed': reference_peptide is not None or docking_position is not None,
                'reference_peptide': reference_peptide,
                'docking_position': docking_position,
                'threshold': threshold,
                'level': level
            }
        }
        
        # Prepare output results structure
        output_results = {
            'processed_protein': {
                'description': f"Processed protein from: {protein}",
                'file_path': processed_protein,
                'pocket_truncation_applied': reference_peptide is not None or docking_position is not None,
                'truncation_details': {
                    'threshold': threshold,
                    'level': level,
                    'reference_peptide': reference_peptide,
                    'docking_position': docking_position
                } if (reference_peptide is not None or docking_position is not None) else None
            },
            'docking_results': {
                'total_complexes': results_data.get('total_complexes', 0) if results_data else 0,
                'complexes_with_scores': results_data.get('complexes_with_scores', 0) if results_data else 0,
                'summary_file': results_data.get('summary_file') if results_data else None,
                'detailed_results': []
            },
            'reference_peptide_results': None
        }
        
        # Process detailed results for each peptide
        if results_data and results_data.get('detailed_results'):
            for result in results_data['detailed_results']:
                if result['peptide_info']['is_reference']:
                    # This is reference peptide re-docking
                    # Filter PDB files to only include final ranked results (rank*_ref2015.pdb)
                    final_pdb_files = [f for f in result['files']['pdb_files'] 
                                     if 'rank' in os.path.basename(f) and 'ref2015' in os.path.basename(f)]
                    
                    output_results['reference_peptide_results'] = {
                        'description': f"Reference peptide re-docking from: {reference_peptide}",
                        'file_path': reference_peptide,
                        'complex_name': result['complex_name'],
                        'result_path': result['result_path'],
                        'results': {
                            'final_pdb_files': final_pdb_files,
                            'score_files': result['files']['score_files'],
                            'best_rosetta_score': result['scores']['best_rosetta_score'],
                            'all_rosetta_scores': result['scores']['rosetta_scores'],
                            'dockq_evaluation': {
                                'dockq_score': result['evaluation']['dockq_score'],
                                'fnat': result['evaluation']['fnat'],
                                'irms': result['evaluation']['irms'],
                                'lrms': result['evaluation']['lrms'],
                                'capri_class': result['evaluation']['capri_class'],
                                'best_rank': result['evaluation'].get('best_rank'),
                                'total_ranks_evaluated': result['evaluation'].get('total_ranks_evaluated')
                            } if result['evaluation']['dockq_score'] is not None else None,
                            'evaluation_errors': result['evaluation'].get('dockq_error')
                        }
                    }
                else:
                    # This is target peptide docking
                    # Filter PDB files to only include final ranked results (rank*_ref2015.pdb)
                    final_pdb_files = [f for f in result['files']['pdb_files'] 
                                     if 'rank' in os.path.basename(f) and 'ref2015' in os.path.basename(f)]
                    
                    peptide_result = {
                        'peptide_sequence': result['peptide_info']['sequence'],
                        'complex_name': result['complex_name'],
                        'result_path': result['result_path'],
                        'results': {
                            'final_pdb_files': final_pdb_files,
                            'score_files': result['files']['score_files'],
                            'best_rosetta_score': result['scores']['best_rosetta_score'],
                            'all_rosetta_scores': result['scores']['rosetta_scores'],
                            'scoring_errors': result['scores'].get('error')
                        }
                    }
                    output_results['docking_results']['detailed_results'].append(peptide_result)
        
        # Combine input parameters and output results
        comprehensive_results = {
            'success': True,
            'input_parameters': {
                'protein': {
                    'input': protein,
                    'is_file': os.path.exists(protein),
                    'is_sequence': not os.path.exists(protein)
                },
                'peptides': {
                    'total_input': len(peptides),
                    'input_sequences': peptides,
                    'total_validated': len(validated_peptides),
                    'validated_sequences': validated_peptides,
                    'validation_details': peptide_info
                },
                'reference_peptide': {
                    'file_path': reference_peptide,
                    'exists': os.path.exists(reference_peptide) if reference_peptide else False,
                    'used_for_pocket_truncation': reference_peptide is not None,
                    'used_for_evaluation': reference_peptide is not None
                } if reference_peptide else None,
                'processing_options': {
                    'docking_position': docking_position,
                    'pocket_truncation_threshold': threshold,
                    'pocket_truncation_level': level,
                    'min_peptide_length': min_length,
                    'max_peptide_length': max_length
                },
                'inference_settings': {
                    'config_file': config,
                    'n_samples': n_samples,
                    'batch_size': batch_size,
                    'cpu_cores': cpu
                },
                'output_settings': {
                    'output_directory': output,
                    'work_directory': work_dir if keep_temp else None,
                    'keep_temp_files': keep_temp
                }
            },
            'output_results': output_results
        }
        
        return comprehensive_results
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'error_traceback': traceback.format_exc(),
            'input_parameters': {
                'protein': {
                    'input': protein,
                    'is_file': os.path.exists(protein) if protein else None,
                    'is_sequence': not os.path.exists(protein) if protein else None
                },
                'peptides': {
                    'total_input': len(peptides) if peptides else 0,
                    'input_sequences': peptides if peptides else [],
                    'total_validated': 0,
                    'validated_sequences': [],
                    'validation_details': []
                },
                'reference_peptide': {
                    'file_path': reference_peptide,
                    'exists': os.path.exists(reference_peptide) if reference_peptide else False,
                    'used_for_pocket_truncation': reference_peptide is not None,
                    'used_for_evaluation': reference_peptide is not None
                } if reference_peptide else None,
                'processing_options': {
                    'docking_position': docking_position,
                    'pocket_truncation_threshold': threshold,
                    'pocket_truncation_level': level,
                    'min_peptide_length': min_length,
                    'max_peptide_length': max_length
                },
                'inference_settings': {
                    'config_file': config,
                    'n_samples': n_samples,
                    'batch_size': batch_size,
                    'cpu_cores': cpu
                },
                'output_settings': {
                    'output_directory': output,
                    'work_directory': work_dir,
                    'keep_temp_files': keep_temp
                }
            },
            'output_results': {
                'processed_protein': {
                    'description': 'Processing failed',
                    'file_path': None,
                    'pocket_truncation_applied': False,
                    'truncation_details': None
                },
                'docking_results': {
                    'total_complexes': 0,
                    'complexes_with_scores': 0,
                    'summary_file': None,
                    'detailed_results': []
                },
                'reference_peptide_results': None,
                'processing_error': str(e)
            }
        }

def main_cli():
    """Command line interface for the processor."""
    parser = argparse.ArgumentParser(
        description="RAPiDock Processor: Main entry for protein-peptide docking with multiple peptides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with protein PDB and peptide sequences
  python processor.py --protein protein.pdb --peptides "HKILHRLLQDS" "EKHKILHRLLQDS" --output results/

  # With reference peptide for pocket truncation and DockQ evaluation
  python processor.py --protein protein.pdb --peptides "HKILHRLLQDS" "EKHKILHRLLQDS" \\
                      --reference-peptide reference.pdb --output results/

  # With docking position for pocket truncation
  python processor.py --protein protein.pdb --peptides "HKILHRLLQDS" \\
                      --docking-position 10.5 20.3 15.7 --output results/

  # Advanced options
  python processor.py --protein protein.pdb --peptides "HK[HYP]RL[PTR]QDS" \\
                      --reference-peptide reference.pdb --threshold 25.0 --level Residue \\
                      --config custom_config.yaml --output results/
        """
    )
    
    # Input arguments
    parser.add_argument("--protein", required=True,
                       help="Protein PDB file path or sequence string")
    parser.add_argument("--peptides", nargs="+", required=True,
                       help="List of peptide sequences (supports both single-letter codes and [XXX] format)")
    parser.add_argument("--reference-peptide", 
                       help="Reference peptide PDB file for pocket truncation and DockQ evaluation")
    parser.add_argument("--docking-position", nargs=3, type=float, metavar=("X", "Y", "Z"),
                       help="XYZ coordinates for docking center (for pocket truncation)")
    
    # Processing options
    parser.add_argument("--threshold", type=float, default=20.0,
                       help="Distance threshold for pocket truncation (default: 20.0 Å)")
    parser.add_argument("--level", choices=["Chain", "Residue", "Atom"], default="Chain",
                       help="Pocket truncation level (default: Chain)")
    parser.add_argument("--min-length", type=int, default=3,
                       help="Minimum peptide length (default: 3)")
    parser.add_argument("--max-length", type=int, default=50,
                       help="Maximum peptide length (default: 50)")
    
    # Output options
    parser.add_argument("--output", required=True,
                       help="Output directory for results")
    parser.add_argument("--work-dir",
                       help="Working directory for temporary files (default: auto-generated)")
    parser.add_argument("--keep-temp", action="store_true",
                       help="Keep temporary files after processing")
    
    # Inference options
    parser.add_argument("--config", default="default_inference_args.yaml",
                       help="Configuration file for inference (default: default_inference_args.yaml)")
    parser.add_argument("--n-samples", type=int, default=10,
                       help="Number of samples to generate per peptide (default: 10)")
    parser.add_argument("--batch-size", type=int, default=40,
                       help="Batch size for inference (default: 40)")
    parser.add_argument("--cpu", type=int, default=16,
                       help="Number of CPU cores to use (default: 16)")
    
    args = parser.parse_args()
    
    # Call main function with parsed arguments
    result = main(
        protein=args.protein,
        peptides=args.peptides,
        reference_peptide=args.reference_peptide,
        docking_position=args.docking_position,
        threshold=args.threshold,
        level=args.level,
        min_length=args.min_length,
        max_length=args.max_length,
        output=args.output,
        work_dir=args.work_dir,
        keep_temp=args.keep_temp,
        config=args.config,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        cpu=args.cpu
    )
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)

if __name__ == "__main__":
    import json
    
    result = main(
        protein='data/7bbg/7bbg_protein_pocket.pdb',
        peptides=['RMFPNAPYL', 'HKILHRLLQDS'],
        reference_peptide='data/7bbg/7bbg_peptide.pdb',
        output='results/test_run/'
    )

    print('\n' + "=" * 80)
    print("ACTUAL RESULT:")
    print("=" * 80)
    print(json.dumps(result, indent=2, ensure_ascii=False))
