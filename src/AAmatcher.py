from __future__ import annotations
from collections import Counter
import numpy as np
import copy
from ase.io import read
from ase.geometry.analysis import Analysis
from pathlib import Path
from itertools import takewhile
import os
import logging
from ase.calculators.calculator import PropertyNotImplementedError

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def seq_from_filename(filename : Path, FF_refdict, cap = True):
    '''
    String that explains this function
    '''    
    components = filename.stem.split(sep='_')
    seq = []
    if not any([components[1][:3] in ['Ace','Nme']]):
        seq.append(components[0].upper() if components[1] == 'nat' else components[0].upper() + '_R')
    elif len(components[0]) == 3:
        if 'Ace' in components[1]:
            return ['ACE_R',components[0].upper(),'NME']
        else:
            return ['ACE',components[0].upper(),'NME_R']

    if cap:
        if len(components[0]) == 3:
            seq.insert(0,'ACE')
            seq.append('NME')
        elif components[0][1:].upper() in FF_refdict.keys():
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
    String that explains this function
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
        for order in range(5):
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
                        try:
                            mapdict[AL.atoms[n1.index(key)].idx] = AL_ref.atoms[n2.index(key)].idx
                        except ValueError:
                            continue
                        

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



if __name__ == "__main__":
    dname = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dname)

    dataset = Path("../example")
    FF_reference_path = Path("../example/amber99sb-star-ildnp.ff/aminoacids.rtp")
    targetdir = Path("../tmp")
    match_mol(dataset,FF_reference_path,targetdir)








## classes ##
class Atom:
    def __init__(self,idx,element):
        self.idx = idx
        self.element = element
        self.neighbors = []

    def add_neighbors(self, neighbors: list(Atom)):
        self.neighbors.extend(neighbors)

    def get_neighbor_elements(self,order):
        curr_neighbors = [self]
        for _ in range(order):
            prev_neighbors = curr_neighbors
            curr_neighbors = []
            for neighbor in prev_neighbors:
                curr_neighbors.extend(neighbor.neighbors)

        elements = ''.join(sorted([neighbor.element for neighbor in curr_neighbors]))
        return elements
    
    def get_neighbor_idxs(self,order):
        curr_neighbors = [self]
        for _ in range(order):
            prev_neighbors = curr_neighbors
            curr_neighbors = []
            for neighbor in prev_neighbors:
                curr_neighbors.extend(neighbor.neighbors)

        return [x.idx for x in curr_neighbors]

class AtomList:
    def __init__(self,atoms,bonds):
        self.idxs: list = []
        self.elements = []
        self.atoms: list[Atom] = []
        self.bonds = []

        for atom in atoms:
            self.atoms.append(Atom(*atom))

        self.set_indices()
        self.set_elements()

        for bond in bonds:
            self.atoms[self.idxs.index(bond[0])].add_neighbors([self.atoms[self.idxs.index(bond[1])]])
            self.atoms[self.idxs.index(bond[1])].add_neighbors([self.atoms[self.idxs.index(bond[0])]])

        self.set_bonds()

    def set_indices(self):
        self.idxs: list = []
        for atom in self.atoms:
            self.idxs.append(atom.idx)

    def set_elements(self):
        self.elements = []
        for atom in self.atoms:
            self.elements.append(atom.element)

    def set_bonds(self):
        self.bonds: list = []
        for atom in self.atoms:
            for neighbor in atom.neighbors:
                self.bonds.append([atom,neighbor])


    def get_neighbor_elements(self,order):
        elements = []
        for atom in self.atoms:
            elements.append(atom.get_neighbor_elements(order))
        return elements

    def by_idx(self,idx):
        return self.atoms[self.idxs.index(idx)]
        
    def __len__(self):
        return len(self.atoms)
    
    def __repr__(self):
        return f"AAmatcher.AtomList object with {len(self.atoms)} atoms and {len(self.bonds)} bonds"

## general utils ##
def _create_molref(AAs_reference: dict, sequence: list):
    mol_ref = []
    for res in sequence:
        AA_refatoms = [[atom[0],atom[0][0]] for atom in AAs_reference[res]['atoms']]
        AA_refbonds = [bond for bond in AAs_reference[res]['bonds'] if not any(idx.startswith(('-','+')) for idx in bond)]
        AA_refAtomList = AtomList(AA_refatoms,AA_refbonds)
        mol_ref.append(AA_refAtomList)
    return mol_ref

## utils (from kimmdy) ##
def read_rtp(path: Path) -> dict:
    # TODO: make this more elegant and performant
    with open(path, "r") as f:
        sections = _get_sections(f, "\n")
        d = {}
        for i, s in enumerate(sections):
            # skip empty sections
            if s == [""]:
                continue
            name, content = _extract_section_name(s)
            content = [c.split() for c in content if len(c.split()) > 0]
            if not name:
                name = f"BLOCK {i}"
            d[name] = _create_subsections(content)
            # d[name] = content

        return d

def _extract_section_name(ls):
    """takes a list of lines and return a tuple
    with the name and the lines minus the
    line that contained the name.
    Returns the empty string if no name was found.
    """
    for i, l in enumerate(ls):
        if l and l[0] != ";" and "[" in l:
            name = l.strip("[] \n")
            ls.pop(i)
            return (name, ls)
    else:
        return ("", ls)

def _is_not_comment(c: str) -> bool:
    return c != ";"

def _get_sections(seq, section_marker):
    data = [""]
    for line in seq:
        line = "".join(takewhile(_is_not_comment, line))
        if line.strip(" ").startswith(section_marker):
            if data:
                # first element will be empty
                # because newlines mark sections
                data.pop(0)
                # only yield section if non-empty
                if len(data) > 0:
                    yield data
                data = [""]
        data.append(line.strip("\n"))
    if data:
        yield data

def _create_subsections(ls):
    d = {}
    subsection_name = "other"
    for i, l in enumerate(ls):
        if l[0] == "[":
            subsection_name = l[1]
        else:
            if subsection_name not in d:
                d[subsection_name] = []
            d[subsection_name].append(l)

    return d
##


       