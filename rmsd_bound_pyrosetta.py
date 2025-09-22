##########################################################################
# File Name: rmsd_bound.py
# Author: huifeng
# mail: 22219012@zju.edu.cn
# Created Time: Tue 27 Feb 2024 03:38:40 PM CST
#########################################################################

import os
import glob
import MDAnalysis
import sys


def classify_capri_peptide(fnat, iRMS, lRMS):
    """
    Classify peptide docking results according to CAPRI official criteria.
    
    CAPRI overall model quality classification (applied in reverse order):
    - High quality: (fnat ≥ 0.5) and either (L-RMSD ≤ 1.0 Å) or (i-RMSDbb ≤ 1.0 Å)
    - Medium quality: (fnat ≥ 0.3) and either (L-RMSD ≤ 5.0 Å) or (i-RMSDbb ≤ 2.0 Å)
    - Acceptable quality: (fnat ≥ 0.1) and either (L-RMSD ≤ 10.0 Å) or (i-RMSDbb ≤ 4.0 Å)
    - Incorrect: (fnat < 0.1) or (L-RMSD > 10.0 Å) and (i-RMSDbb > 4.0 Å)
    
    Args:
        fnat: Fraction of native contacts (string)
        iRMS: Interface RMSD (i-RMSDbb) (string)
        lRMS: Ligand RMSD (L-RMSD) (string)  
        
    Returns:
        str: Classification ('High', 'Medium', 'Acceptable', 'Incorrect')
    """
    try:
        # Convert string values to float, handle None values
        fnat_val = float(fnat) if fnat is not None and fnat != 'None' else 0.0
        iRMS_val = float(iRMS) if iRMS is not None and iRMS != 'None' else float('inf')
        lRMS_val = float(lRMS) if lRMS is not None and lRMS != 'None' else float('inf')  
        
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


def rmsd(i):
    if not os.path.exists(i+'/rmsd_pyrosetta.csv'):
        with open(i+'/rmsd_pyrosetta.csv', 'w') as f:
            # with open(i+'/rmsd_opt.csv','w') as f:
            f.writelines('id,pyrosetta,fnat,irms,lrms,dockq,capri_class\n')
            # f.writelines('id,optmized,fnat,irms,lrms,dockq,\n')
            f.close()
        protein_file = glob.glob(i+'/*protein_raw.pdb')[0]
        peptide_native_file: str = glob.glob(i+'/*peptide_raw.pdb')[0]
        protein = MDAnalysis.Universe(protein_file)
        for idx, seg in enumerate(protein.segments):
            seg.segid = chr(ord('A') + idx)
            for atom in seg.atoms:
                atom.chainID = chr(ord('A') + idx)
        peptide_native = MDAnalysis.Universe(peptide_native_file)
        peptide_native.segments[0].segid = 'p'
        for atom in peptide_native.atoms:
            atom.chainID = 'p'
        peptide_native.residues.resids = peptide_native.residues.resids - peptide_native.residues.resids.min() + 1
        peptide_native_merge = MDAnalysis.Merge(protein.atoms, peptide_native.atoms)
        peptide_native_merge_path = peptide_native_file.replace('peptide_raw.pdb', 'complex_native.pdb')
        try:
            peptide_native_merge.atoms.write(peptide_native_merge_path)
            print(f'Successfully created native complex: {peptide_native_merge_path}')
            print(f'Native complex exists: {os.path.exists(peptide_native_merge_path)}')
        except Exception as e:
            print(f'Error creating native complex: {e}')
            return

        # peptide_model_files = [j for j in glob.glob(i+'/rank*confidence*.pdb') if not 'opt' in j]
        peptide_model_files = glob.glob(i+'/rank*_ref2015.pdb')
        # return(peptide_model_files)
        # for peptide_model_file in sorted(peptide_model_files,key=lambda x : int(x.split('rank')[1].split('_')[0])):
        for peptide_model_file in sorted(peptide_model_files,key=lambda x : int(x.split('rank')[1].split('_')[0])):
            print('start to calculate DockQ')
            rank = os.path.basename(peptide_model_file).split('_')[0][4:]
            # confidence = os.path.basename(peptide_model_file).split('confidence')[1].split('.pdb')[0]
            confidence = os.path.basename(peptide_model_file).split('_')[1].split('.pdb')[0]
            peptide_model = MDAnalysis.Universe(peptide_model_file)
            peptide_model.segments[0].segid = 'p'
            for atom in peptide_model.atoms:
                atom.chainID = 'p'
            peptide_model_merge = MDAnalysis.Merge(protein.atoms, peptide_model.atoms)
            peptide_model_merge_path = peptide_model_file.replace('.pdb','_complex.pdb')
            try:
                peptide_model_merge.atoms.write(peptide_model_merge_path)
                print(f'Successfully created: {peptide_model_merge_path}')
                print(f'File exists: {os.path.exists(peptide_model_merge_path)}')
            except Exception as e:
                print(f'Error creating complex file: {e}')
                continue
            
            # Get the current script directory and construct relative path to DockQ
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dockq_path = os.path.join(script_dir, "tools", "DockQ-1.0", "DockQ.py")
            cmd = f"python {dockq_path} {peptide_model_merge_path} {peptide_native_merge_path} -native_chain1 p -no_needle"
            print('cmd: ', cmd)
            b = os.popen(cmd)
            
            fnat = None
            iRMS = None
            lRMS = None
            DockQ = None
            for _ in b.readlines():
                print('_: ', _)
                if _.startswith('Fnat'):
                    fnat = _.split()[1]
                if _.startswith('iRMS'):
                    iRMS = _.split()[1]
                if _.startswith('LRMS'):
                    lRMS = _.split()[1]
                if _.startswith('DockQ'):
                    DockQ = _.split()[1]
            # Calculate CAPRI-peptide classification
            capri_class = classify_capri_peptide(fnat, iRMS, lRMS)
            
            with open(i+'/rmsd_pyrosetta.csv','a') as f:
            # with open(i+'/rmsd_opt.csv','a') as f:
                f.writelines(f'{rank},{confidence},{fnat},{iRMS},{lRMS},{DockQ},{capri_class}\n')
            f.close()
            # os.remove(peptide_model_merge_path)
        # os.remove(peptide_native_merge_path)

if __name__ == "__main__":
    rmsd(sys.argv[1])
#a = glob.glob('results/valid_bound_local/*/rank*pyrosetta_opt*')
#pdbids = sorted(set([os.path.dirname(i) for i in a]))
#from multiprocessing import Pool
#p = Pool(10)
#map_fn = p.imap_unordered
#pdbids = ['results/valid_bound_local_short_199_P/7d6r_G','results/valid_bound_local_short_199_P/7ea1_B','results/valid_bound_local_short_199_P/7etu_B','results/valid_bound_local_short_199_P/7kco_D','results/valid_bound_local_short_199_P/7m51_A','results/valid_bound_local_short_199_P/7lg0_C','results/valid_bound_local_short_199_P/7mgs_B','results/valid_bound_local_short_199_P/7nab_C']
#map_fn(rmsd, pdbids)
#map_fn(rmsd, glob.glob('results/global_case/7M60'))
#p.close()
#p.join()
