from __future__ import annotations
from collections import Counter
import numpy as np
import copy
from ase.io import read
from ase.geometry.analysis import Analysis
from pathlib import Path
import os
import logging
from ase.calculators.calculator import PropertyNotImplementedError

from utils import *
from representation import Atom, AtomList

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def seq_from_filename(filename : Path, AAs_reference=None, cap = True):
    '''
    Returns a list of strings describing a sequence found in a filename.
    '''    
    components = filename.stem.split(sep='_')
    seq = []
    seq.append(components[0].upper() if components[1] == 'nat' else components[0].upper() + '_R')

    if cap:
        if len(components[0]) == 3:
            seq.insert(0,'ACE')
            seq.append('NME')
        elif not AAs_reference is None:
            if components[0][1:].upper() in AAs_reference.keys():
                if components[0].startswith('N'):
                    seq.append('NME')
                elif components[0].startswith('C'):
                    seq.insert(0,'ACE')
                else:
                    raise ValueError(f"Invalid filename {filename} for sequence conversion!")
        else:
            raise ValueError(f"Invalid filename {filename} for sequence conversion!")
    return seq 

def generate_radical_reference(AAs_reference: dict, AA: str, heavy_name: str):
    '''
    String that explains this function
    Assumes a hydrogen was abstracted from the radical atom
    '''
    logging.info(f"--- Generating radical reference entry for residue {AA} with a {heavy_name} radical ---")
    ## search for hydrogen attached to radical
    for bond in AAs_reference[AA]['bonds'][::-1]:
        bond_atoms = bond
        if heavy_name in bond_atoms:
            heavy_partner = bond_atoms[0] if heavy_name == bond_atoms[1] else bond_atoms[1]
            # assumes that it does not matter which hydrogen was abstracted
            if heavy_partner.startswith('H'):
                rmv_H = heavy_partner
                break

    AAs_reference[AA + '_R'] = copy.deepcopy(AAs_reference[AA])
    for atom in AAs_reference[AA + '_R']['atoms']:
        if rmv_H == atom[0]:
            rmv_atom = atom
            break
    for bond in AAs_reference[AA + '_R']['bonds']:
        if rmv_H in bond:
            rmv_bond = bond
            break
    AAs_reference[AA + '_R']['atoms'].remove(rmv_atom)
    AAs_reference[AA + '_R']['bonds'].remove(rmv_bond)
    return AAs_reference


def read_g09(file: Path, sequence: str, AAs_reference: dict):
    '''
    Returns a mol and an ASE trajectory.
    THIS ASSUMES THAT THE ATOMS ARE ORDERED BY RESIDUE!
    '''

    logging.info(f"--- Starting to read coordinates from file {file} with sequence {sequence} ---")
    trajectory = read(file,index=':')

    ## get formula and separation of atoms into residues ##
    formula = trajectory[-1].get_chemical_formula(mode='all')

    AA_delim_idxs = []              # index of first atom in new residue   
    pos = 0
    # this assignment of AA_delim_idx assumes the atoms to be ordered by residues
    for AA in sequence:
        AA_len = len(AAs_reference[AA]['atoms'])
        AA_delim_idxs.append([pos,pos+AA_len])
        pos += AA_len
    assert pos == len(formula) , f"Mismatch in number of atoms in {sequence}:{pos} and {formula}:{len(formula)}"

    ## split molecule into residues and get list of bonds. reconstruct atomnames using list ##
    ana = Analysis(trajectory[0])
    [bonds] = ana.unique_bonds
    
    res_AtomLists = []
    for _,AA_delim_idx in enumerate(AA_delim_idxs):
        AA_atoms = [[idx,formula[idx]] for idx in list(range(*AA_delim_idx))]
        AA_partners = [list(np.extract((np.array(bond) < AA_delim_idx[1]) & (AA_delim_idx[0] <= np.array(bond)),bond)) for bond in bonds[AA_delim_idx[0]:AA_delim_idx[1]]]
        AA_bonds = [[i+AA_delim_idx[0],partner] for i, partners in enumerate(AA_partners) for partner in partners]

        res_AtomLists.append(AtomList(AA_atoms,AA_bonds))

    return res_AtomLists, trajectory

def match_mol(mol: list, AAs_reference: dict, seq: list):
    '''
    String that explains this function
    '''

    logging.info(f"--- Matching molecule with sequence {seq} to reference ---")
    ## get mol_ref from function here
    mol_ref = _create_molref(AAs_reference, seq)
    logging.info(f"created reference: {mol_ref}")

    atom_order = []
    for i in range(len(seq)):
        AL = mol[i]
        AL_ref = mol_ref[i]
    
        ## create mapping scheme for sorting the atoms correctly ##
        c1 = sorted(AL.get_neighbor_elements(0))
        c2 = sorted(AL_ref.get_neighbor_elements(0))
        assert  c1 == c2 , f"Elements for molecule {seq} residue {i} ({c1}) don't map to FF reference ({c2}! Can't continue the mapping" 
        # assume both graphs are isomorphic after this assert

        mapdict = {}
        # #try simple matching by neighbor elements
        for order in range(4):
            n1 = AL.get_neighbor_elements(order)
            n2 = AL_ref.get_neighbor_elements(order)

            occurences = Counter(n1)
            logging.debug(order)
            logging.debug(occurences)
            logging.debug(mapdict)

            for key,val in occurences.items():
                if val == 1:
                    if AL.atoms[n1.index(key)].idx not in mapdict.keys():
                        logging.debug(AL.atoms[n1.index(key)].idx,mapdict)
                        mapdict[AL.atoms[n1.index(key)].idx] = AL_ref.atoms[n2.index(key)].idx

        rmapdict = dict(map(reversed, mapdict.items()))
        if len(mapdict.keys()) < len(AL):
            AA_remainder = sorted(list(set(AL.idxs)-set(mapdict.keys())))
            AA_refremainder = sorted(list(set(AL_ref.idxs)-set(rmapdict.keys())))
            logging.debug(f"{AA_remainder}/{AA_refremainder} remain unmatched after neighbor comparison!")
        logging.debug(mapdict)

        ## try matching of indistinguishable atoms bound to the same atom that has a mapping
        for _ in range(2):
            AA_match = list(mapdict.keys())
            for atom_idx in AA_match:
                neighbors = AL.by_idx(atom_idx).neighbors
                refneighbors = AL_ref.by_idx(mapdict[atom_idx]).neighbors
                for neighbor in neighbors:
                    if neighbor.idx in AA_remainder:
                        for refneighbor in refneighbors:
                            if refneighbor.idx in AA_refremainder:
                                mapdict[neighbor.idx] = refneighbor.idx
                                rmapdict[refneighbor.idx] = neighbor.idx
                                AA_remainder.remove(neighbor.idx)
                                AA_refremainder.remove(refneighbor.idx)
                                logging.debug(f"match {neighbor.idx}:{refneighbor.idx}")
                                break

            if len(AA_remainder):
                logging.info(f"{AA_remainder}/{AA_refremainder} remain unmatched; checking for indistinguishable atoms")
            else:
                logging.info(f"All atoms in were successfully matched:{mapdict}!")
                break
        if len(AA_remainder):
            raise ValueError(f"{AA_remainder}/{AA_refremainder} remain unmatched, cannot complete matching")
 
        logging.debug(mol_ref[i].idxs)
        logging.debug(rmapdict)
        atom_order.extend([rmapdict[atom] for atom in mol_ref[i].idxs])
        logging.debug(f"New atom order {atom_order}")
        logging.debug('\n\n')
    return atom_order

def write_trjtopdb(outfile: Path, trajectory, atom_order: list, seq: list, AAs_reference: dict):
    '''
    String that explains this function
    '''

    logging.info(f"--- Writing Trajectory of molecule with sequence {seq} to PDBs ---")
    outfile.parent.mkdir(exist_ok=True)

    ## write single pdb per opt step## 
    for i,step in enumerate(trajectory):
        writefile = outfile.parent / f"{outfile.stem}_{i}.pdb"
        with open(writefile,'w') as f:
            f.write(f"COMPND     step {i}\n")
            f.write(f"AUTHOR     generated by {__file__}\n")
            try:
                f.write(f"REMARK   0\nREMARK   0 energy: {step.get_total_energy()} eV\n")
            except PropertyNotImplementedError:
                f.write(f"REMARK   0\nREMARK   0 energy: nan eV\n")
            AA_pos = 0
            res_len = len(AAs_reference[seq[AA_pos]]['atoms'])
            for j,k in enumerate(atom_order):
                if j >= res_len:
                    AA_pos += 1
                    res_len += len(AAs_reference[seq[AA_pos]]['atoms'])
                atomname = AAs_reference[seq[AA_pos]]['atoms'][j-res_len][0]
                f.write('{0}  {1:>5d} {2:^4s} {3:<3s} {4:1s}{5:>4d}    {6:8.3f}{7:8.3f}{8:8.3f}{9:6.2f}{10:6.2f}          {11:>2s}\n'.format('ATOM',j,atomname ,seq[AA_pos].split(sep="_")[0],'A',AA_pos+1,*step.positions[k],1.00,0.00,atomname[0]))
            f.write("END")
    return


## general utils ##
'''
Construct a molecule, i.e. List of AtomLists, from whatr the force field expecs the residues to look like. 
'''
def _create_molref(AAs_reference: dict, sequence: list):
    mol_ref = []
    for res in sequence:
        AA_refatoms = [[atom[0],atom[0][0]] for atom in AAs_reference[res]['atoms']]
        AA_refbonds = [bond for bond in AAs_reference[res]['bonds'] if not any(idx.startswith(('-','+')) for idx in bond)]
        AA_refAtomList = AtomList(AA_refatoms,AA_refbonds)
        mol_ref.append(AA_refAtomList)
    return mol_ref


if __name__ == "__main__":
    dname = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dname)

    dataset = Path("../example/g09_dataset")
    FF_reference_path = Path("../example/amber99sb-star-ildnp.ff/aminoacids.rtp")
    targetdir = Path("../tmp")
    match_mol(dataset,FF_reference_path,targetdir)
