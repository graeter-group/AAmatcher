"""
Find the radical in a topology. Do this using the generate_missing... method from openmm forcefields. then collect all atoms names to find the missing name. then figure out to which atom this H is connected. this way we might not be able to differentiate between charge and radical, but for standard cases it should work.
"""

#%%
import openmm
from openmm import unit
from openmm.app import Simulation
from openmm.app import ForceField, topology

from openmm.unit import Quantity
import json
import numpy as np
from typing import List, Tuple, Dict, Union, Callable
from pathlib import Path
#%%

def get_radicals(topology:topology, forcefield:ForceField=ForceField("amber99sbildn.xml")):
    """
    Returns the indices of the residues in which radicals occur and  as residue_indices, radical_names: List[int], List[str]

    Find the radicals in a topology. Do this using the generate_missing... method from openmm forcefields. then collect all atoms names to find the missing name. then figure out to which atom this H is connected. this way we might not be able to differentiate between charge and radical, but for standard cases it should work.
    NOTE: currently only works for one radical per residue, i.e. one missing hydrogen per residue.
    """
    # get the missing hydrogens assuming that the ff cannot assign residues containing a radical
    
    residue_indices = []
    radical_names = []

    [templates, residues] = forcefield.generateTemplatesForUnmatchedResidues(topology)
    for t_idx, template in enumerate(templates):
        ref_template = forcefield._templates[template.name]
        # the atom names stored in the template of the residue
        ref_names = [a.name for a in ref_template.atoms]

        if len(template.atoms) != len(ref_template.atoms)-1:
            raise ValueError(f"Encountered {template.name} with {len(template.atoms)} atoms that cannot be matched to reference {len(ref_template.atoms)}. The reason for this cannot be a single missing hydrogen.")

        # delete the H indices since the Hs are not distinguishable anyways
        # in the template, the Hs are named HB2 and HB3 while in some pdb files they are named HB1 and HB2
        for i, n in enumerate(ref_names):
            if n[0] == 'H' and len(n) == 3:
                ref_names[i] = n[:-1]

        # find the atom name of the one that is not in the template
        names = [a.name[:-1] if a.name[0] == 'H' and len(a.name) == 3 else a.name for a in template.atoms]
        diff = []
        for n in ref_names:
            try:
                # search the names for the atom. we already know that names has less elements than ref_names
                # if it is found, remove it out of the names list.
                # this is necessary because we can have two HB atoms in the residue, but only one in the radical, i.e. number of occurence is important
                idx = names.index(n)
                names.pop(idx)
            except ValueError:
                diff.append(n)
        if len(diff) != 1:
            raise ValueError(f"Could not find the missing atom in {template.name}.\nAtom names are          {names}\nand reference names are {ref_names},\ndiff is {diff}.")
        
        # if the missing atom is a hydrogen, its neighbor is a radical.
        # thus, loop over all atoms in the residue and find the one that is connected to the missing atom
        for atom in ref_template.atoms:
            name = atom.name[:-1] if atom.name[0] == 'H' and len(atom.name) == 3 else atom.name
            if name == diff[0]:
                if len(atom.bondedTo)!=1:
                    raise ValueError(f"Found {len(atom.bonds)} bonds for atom {atom.name} in residue {template.name}, but expected 1 since we expect to find missing hydrogens (that would be bonded to exactly one atom).") 
                # append the atom name to the radical sequence
                radical_idx = atom.bondedTo[0]
                radical_names.append(ref_template.atoms[radical_idx].name)
                residue_indices.append(residues[t_idx].index)
                break

    return residue_indices, radical_names


#%%
if __name__=="__main__":
    from PDBData.PDBMolecule import PDBMolecule
    rad = PDBMolecule.from_pdb("./../../../mains/tests/radicals/F_rad.pdb")
    i, n = get_radicals(rad.to_openmm().topology)
    print(i, n)
    # %%
