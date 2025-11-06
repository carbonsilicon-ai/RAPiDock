##########################################################################
# File Name: pocket_trunction.py
# Author: huifeng
# mail: huifengzhao@zju.edu.cn
# Created Time: Thu 19 Oct 2023 09:31:15 AM CST
#########################################################################

"""
trunction of protein under three different level: Chain, Residue, Atom
"""

from Bio.PDB import NeighborSearch, PDBIO, Select, PDBParser
import argparse
import numpy as np


class PocketResidueSelect(Select):
    def __init__(self, residues):
        self.residues = residues

    def accept_residue(self, residue):
        return residue in self.residues
    
    def accept_atom(self, atom):
        # Filter out hydrogen atoms
        return atom.element != 'H'


def clean_pdb_structure(pdb_file_path: str, output_path: str = None) -> bool:
    """
    Clean PDB structure by removing residues with incomplete backbone atoms.
    This is necessary after pocket extraction to avoid Rosetta errors.
    
    Args:
        pdb_file_path: Path to input PDB file
        output_path: Path to output cleaned PDB file (if None, overwrites input)
        
    Returns:
        True if cleaning was successful, False otherwise
    """
    if output_path is None:
        output_path = pdb_file_path
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file_path)
        
        class CompleteResidueSelect(Select):
            """Only select residues with complete backbone atoms (N, CA, C, O)"""
            def accept_residue(self, residue):
                # Skip heteroatoms and water
                if residue.id[0] != ' ':
                    return False
                
                # Check for backbone atoms
                required_atoms = {'N', 'CA', 'C', 'O'}
                atom_names = {atom.name for atom in residue.get_atoms()}
                
                # Return True only if all required backbone atoms are present
                return required_atoms.issubset(atom_names)
            
            def accept_atom(self, atom):
                # Filter out hydrogen atoms
                return atom.element != 'H'
        
        # Count residues before and after
        total_residues = sum(1 for _ in structure.get_residues())
        
        # Save cleaned structure
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_path, CompleteResidueSelect())
        
        # Count cleaned residues
        cleaned_structure = parser.get_structure('cleaned', output_path)
        cleaned_residues = sum(1 for _ in cleaned_structure.get_residues())
        
        removed = total_residues - cleaned_residues
        if removed > 0:
            print(f"PDB cleaning: removed {removed} incomplete residues ({cleaned_residues}/{total_residues} retained)")
        
        return True
        
    except Exception as e:
        print(f"Error cleaning PDB structure: {e}")
        return False


def residues_saver(structure, residues, out_name, verbose=0):
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    pdbio.save(out_name, PocketResidueSelect(residues))
    if verbose:
        print("residues saved at {}".format(out_name))


def pocket_trunction(
    protein,
    peptide=None,
    threshold=20.0,  # 12
    save_name=None,
    xyz=None,
    level="Chain",  # residue
    exclude_chain=None,
    threshold_keep=5.0,
    clean_pdb=True,
):
    # Load Peptide Structures
    if xyz is not None:
        if isinstance(xyz[0], list):
            peptide_coords = np.array(xyz)
        else:
            peptide_coords = [np.array(xyz)]

    if peptide is not None:
        parser = PDBParser()
        peptide_structure = parser.get_structure("peptide", peptide)
        # Filter out hydrogen atoms
        peptide_coords = [atom.get_coord() for atom in peptide_structure.get_atoms() if atom.element != 'H']
    
    print('@@@@@!!@@@@@peptide_coords', peptide_coords, flush=True)

    other_chain = set(exclude_chain) if exclude_chain is not None else {}

    # Load Protein Sructures
    if type(protein) == str:
        parser = PDBParser()
        protein_structure = parser.get_structure("protein", protein)

    # Extract pocket residues (filter out hydrogen atoms)
    protein_atoms = [atom for atom in protein_structure.get_atoms() if atom.element != 'H']
    ns = NeighborSearch(protein_atoms)
    pocket_residues_far = {
        res.get_parent()
        for coord in peptide_coords
        for res in ns.search(coord, threshold)
    }  # 5-20A 
    pocket_residues_near = {
        res.get_parent()
        for coord in peptide_coords
        for res in ns.search(coord, threshold_keep)
    }  # 0-5-A 
    pocket_chains_far = {residue.get_parent() for residue in pocket_residues_far}
    pocket_chains_near = {residue.get_parent() for residue in pocket_residues_near}

    pocket_residues = set()
    if level == "Residue":
        pocket_residue_list_far = [
            [
                residue
                for residue in pocket_residues_far
                if residue.get_parent() == chain
            ]
            for chain in pocket_chains_far
        ]
        pocket_residue_list_near = [
            [
                residue
                for residue in pocket_residues_near
                if residue.get_parent() == chain
            ]
            for chain in pocket_chains_near
        ]

        for _ in pocket_residue_list_near:
            chain_id = _[0].get_parent()
            res_ids = [res.get_full_id()[-1][1] for res in _]
            res_id_min = min(res_ids)
            res_id_max = max(res_ids)
            # print(chain_id, res_id_min,res_id_max)
            for chain in protein_structure.get_chains():
                if chain == chain_id:
                    for residue in chain.get_residues():
                        if res_id_min <= residue.get_full_id()[-1][1] <= res_id_max:
                            pocket_residues.add(residue)
        for _ in pocket_residue_list_far:
            if (_[0].get_parent().id in other_chain) and (
                _[0].get_parent() in list(pocket_chains_near)
            ):
                continue
            chain_id = _[0].get_parent()
            res_ids = [res.get_full_id()[-1][1] for res in _]
            res_id_min = min(res_ids)
            res_id_max = max(res_ids)
            # print(chain_id, res_id_min,res_id_max)
            for chain in protein_structure.get_chains():
                if chain == chain_id:
                    for residue in chain.get_residues():
                        if res_id_min <= residue.get_full_id()[-1][1] <= res_id_max:
                            pocket_residues.add(residue)

    elif level == "Chain":
        for chain in protein_structure.get_chains():
            if chain in list(pocket_chains_near):
                for residue in chain.get_residues():
                    pocket_residues.add(residue)
            if (
                (chain in list(pocket_chains_far))
                and (chain.id not in other_chain)
                and (chain not in list(pocket_chains_near))
            ):
                for residue in chain.get_residues():
                    pocket_residues.add(residue)

    elif level == "Atom":
        pocket_residues = set(list(pocket_residues_far) + list(pocket_residues_near))
    
    # Convert set to list for saving
    pocket_residues = list(pocket_residues)

    # (Optional): save the pocket resides
    if save_name:
        residues_saver(protein_structure, pocket_residues, save_name)
        # Clean the saved PDB structure to remove incomplete residues
        if clean_pdb:
            clean_pdb_structure(save_name)
    return len([res for res in protein_structure.get_residues()]), len(pocket_residues)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protein_path", type=str, default=None, help="Path to the protein file"
    )
    parser.add_argument(
        "--peptide_path", type=str, default=None, help="Path to the peptide file"
    )
    parser.add_argument(
        "--xyz", nargs="+", type=float, default=None, help="Center of pocket"
    )
    parser.add_argument("--threshold", type=float, default=None, help="Cutoff threshold")
    parser.add_argument(
        "--save_name", type=str, default=None, help="Path to the output protein file"
    )
    parser.add_argument("--level", type=str, default=None, help="Chain or Residue")
    parser.add_argument(
        "--exclude_chain", nargs="+", type=str, default=None, help="Chain excluded"
    )
    parser.add_argument(
        "--threshold_keep",
        type=float,
        default=5,
        help="Cutoff threshold for keeping chains even they are included in exclude_chain",
    )
    args = parser.parse_args()

    pocket_trunction(
        args.protein_path,
        args.peptide_path,
        args.threshold,
        args.save_name,
        args.xyz,
        args.level,
        args.exclude_chain,
        args.threshold_keep,
    )
