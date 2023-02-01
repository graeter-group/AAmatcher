from __future__ import annotations
from collections import Counter
import numpy as np
from ase.io import read,write
from ase.io.gaussian import read_gaussian_out
from ase.io.trajectory import Trajectory
from ase.build import molecule
from ase.geometry.analysis import Analysis
from pathlib import Path
import matplotlib.pyplot as plt
import re
from itertools import takewhile
import os

def convert_log_pdb(dataset,FF_reference_path,targetdir):
    '''
    String explaining the package
    '''

    ## from kimmdy
    def read_rtp(path: Path) -> dict:
        # TODO: make this more elegant and performant
        with open(path, "r") as f:
            sections = get_sections(f, "\n")
            d = {}
            for i, s in enumerate(sections):
                # skip empty sections
                if s == [""]:
                    continue
                name, content = extract_section_name(s)
                content = [c.split() for c in content if len(c.split()) > 0]
                if not name:
                    name = f"BLOCK {i}"
                d[name] = create_subsections(content)
                # d[name] = content

            return d

    def extract_section_name(ls):
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

    def is_not_comment(c: str) -> bool:
        return c != ";"

    def get_sections(seq, section_marker):
        data = [""]
        for line in seq:
            line = "".join(takewhile(is_not_comment, line))
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

    def create_subsections(ls):
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

    def get_str_chars(string,indices):
        return ''.join(sorted([string[x] for x in indices]))

    def seq_from_filename(filename : Path, FF_refdict, cap = True):
        components = filename.stem.split(sep='_')
        seq = []
        seq.append(components[0].upper() if components[1] == 'nat' else components[0].upper() + '_R')

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

        def set_indices(self):
            self.idxs: list = []
            for atom in self.atoms:
                self.idxs.append(atom.idx)

        def set_elements(self):
            self.elements = []
            for atom in self.atoms:
                self.elements.append(atom.element)

        def get_neighbor_elements(self,order):
            elements = []
            for atom in self.atoms:
                elements.append(atom.get_neighbor_elements(order))
            return elements

        def by_idx(self,idx):
            return self.atoms[self.idxs.index(idx)]
            

        def __len__(self):
            return len(self.atoms)


    ## read aminoacids.rtp of supplied force field to get list of bonds by atomname and list of atoms ##
    AAs_reference = read_rtp(FF_reference_path)

    files = dataset.glob("*nat*.log")
    for file in files:
        seq = seq_from_filename(file,AAs_reference)
        print(seq)
        trajectory = read(file,index=':')

        ## get formula and separation of atoms into residues ##
        formula = trajectory[-1].get_chemical_formula(mode='all')
        AA_delim_idxs = []              # index of first atom in new residue
        pos = 0
        # this assignment of AA_delim_idx assumes the atoms to be ordered by residues
        for AA in seq:
            AA_len = len(AAs_reference[AA]['atoms'])
            AA_delim_idxs.append([pos,pos+AA_len])
            pos += AA_len

        print(AA_delim_idxs)
        print(formula)
        assert pos == len(formula) , f"Mismatch in number of atoms in {seq}:{pos} and {formula}:{len(formula)}"


        ## split molecule into residues and get list of bonds. reconstruct atomnames using list ##
        ana = Analysis(trajectory[-1])
        [bonds] = ana.unique_bonds
        [allbonds] = ana.all_bonds
        #allbonds = [[int(idx) for idx in bond] for bond in allbonds]
        print(bonds)
        print(allbonds,type(allbonds[0]),type(allbonds[0][0]))
        atom_ordering = []
        for i,AA_delim_idx in enumerate(AA_delim_idxs):
            print(f"\n______{i+1}______")
            AA_atoms = [[idx,formula[idx]] for idx in list(range(*AA_delim_idx))]
            AA_partners = [list(np.extract((np.array(bond) < AA_delim_idx[1]) & (AA_delim_idx[0] <= np.array(bond)),bond)) for bond in bonds[AA_delim_idx[0]:AA_delim_idx[1]]]
            AA_bonds = [[i+AA_delim_idx[0],partner] for i, partners in enumerate(AA_partners) for partner in partners]
            print(AA_atoms,AA_partners,AA_bonds)

            AA_refatoms = [[atom[0],atom[0][0]] for atom in AAs_reference[seq[i]]['atoms']]
            AA_refbonds = [bond for bond in AAs_reference[seq[i]]['bonds'] if not any(idx.startswith(('-','+')) for idx in bond)]
            #AA_refbonds = [bond for bond in AA_reference[seq[i]]['bonds']]
            print(AA_refatoms,AA_refbonds)

            AA_AtomList = AtomList(AA_atoms,AA_bonds)
            AA_refAtomList = AtomList(AA_refatoms,AA_refbonds)

    
        ## create mapping scheme for sorting the atoms correctly ##
            c1 = sorted(AA_AtomList.get_neighbor_elements(0))
            c2 = sorted(AA_refAtomList.get_neighbor_elements(0))
            assert  c1 == c2 , f"Elements for molecule {seq} in {file} ({c1}) don't map to FF reference ({c2}! Can't continue the mapping" 
            # assume both graphs are isomorphic after this assert
            mapdict = {}
            # try simple matching by neighbor elements
            for order in range(4):
                n1 = AA_AtomList.get_neighbor_elements(order)
                n2 = AA_refAtomList.get_neighbor_elements(order)

                occurences = Counter(n1)
                print(occurences,mapdict)
                for key,val in occurences.items():
                    if val == 1:
                        if AA_AtomList.atoms[n1.index(key)].idx not in mapdict.keys():
                            print(AA_AtomList.atoms[n1.index(key)].idx,mapdict)
                            mapdict[AA_AtomList.atoms[n1.index(key)].idx] = AA_refAtomList.atoms[n2.index(key)].idx


            rmapdict = dict(map(reversed, mapdict.items()))
            if len(mapdict.keys()) < len(AA_AtomList):
                AA_remainder = sorted(list(set(AA_AtomList.idxs)-set(mapdict.keys())))
                AA_refremainder = sorted(list(set(AA_refAtomList.idxs)-set(rmapdict.keys())))
                print(f"{AA_remainder}/{AA_refremainder} remain unmatched after neighbor comparison!")
            print(mapdict,rmapdict)

            # try matching of indistinguishable atoms bound to the same atom that has a mapping
            for i in range(2):
                AA_match = list(mapdict.keys())
                for atom_idx in AA_match:
                    neighbors = AA_AtomList.by_idx(atom_idx).neighbors
                    refneighbors = AA_refAtomList.by_idx(mapdict[atom_idx]).neighbors
                    for neighbor in neighbors:
                        if neighbor.idx in AA_remainder:
                            for refneighbor in refneighbors:
                                if refneighbor.idx in AA_refremainder:
                                    mapdict[neighbor.idx] = refneighbor.idx
                                    rmapdict[refneighbor.idx] = neighbor.idx
                                    AA_remainder.remove(neighbor.idx)
                                    AA_refremainder.remove(refneighbor.idx)
                                    print(f"match {neighbor.idx}:{refneighbor.idx}")
                                    break
                if len(AA_remainder):
                    print(f"{AA_remainder}/{AA_refremainder} remain unmatched checking for indistinguishable atoms")
                else:
                    print(f"All atoms in were successfully matched:{mapdict}!")
                    break
            if len(AA_remainder):
                raise ValueError(f"{AA_remainder}/{AA_refremainder} remain unmatched checking for indistinguishable atoms")
            atom_ordering.extend([rmapdict[atom[0]] for atom in AA_refatoms])
        print(atom_ordering)
        print('\n\n')
        ## write single pdb per opt step## 
        targetdir.mkdir(exist_ok=True)
    
        for i,step in enumerate(trajectory):
            outfile = targetdir / (file.stem + f"_{i}.pdb")
            print(outfile)
            with open(outfile,'w') as f:
                f.write(f"COMPND    {file} step {i}\n")
                f.write(f"AUTHOR    generated by {__file__}\n")
                f.write(f"REMARK   0\nREMARK   0 energy: {step.get_total_energy()} eV\n")
                AA_pos = 0
                for j,k in enumerate(atom_ordering):
                    if j >= AA_delim_idxs[AA_pos][1]:
                        AA_pos += 1
                    atomname = AAs_reference[seq[AA_pos]]['atoms'][j-AA_delim_idxs[AA_pos][0]][0]
                    f.write('{0}  {1:>5d} {2:^4s} {3:<3s} {4:1s}{5:>4d}    {6:8.3f}{7:8.3f}{8:8.3f}{9:6.2f}{10:6.2f}          {11:>2s}\n'.format('ATOM',j,atomname ,seq[AA_pos],'A',AA_pos+1,*step.positions[k],1.00,0.00,atomname[0]))
                f.write("END")
    return

if __name__ == "__main__":
    dname = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dname)

    dataset = Path("../example")
    FF_reference_path = Path("../example/amber99sb-star-ildnp.ff/aminoacids.rtp")
    targetdir = Path("../tmp")
    convert_log_pdb(dataset,FF_reference_path,targetdir)
        