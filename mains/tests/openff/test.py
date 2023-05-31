#%%
from openmm.app import PDBFile
from openff.toolkit.topology import Molecule
top = PDBFile('pep.pdb').topology
# %%

def assign_standard_charge(atom_name:str, residue:str):
    """
    Assigns standard charges to atoms based on their name and residue.
    """

    if residue in assign_standard_charge.atom_dict.keys():
        if atom_name == assign_standard_charge.atom_dict[residue]:
            return assign_standard_charge.charge_dict[residue]
    # else, the atom is not charged
    return 0
    
assign_standard_charge.atom_dict = {
    "GLU":"OE1", 
    "ASP":"OD1",
    "LYS":"NZ",
    "ARG":"NH1",
    "HIS":"ND1", # choose HIE as the standard protonation state
    "HIE":"ND1"
    }

assign_standard_charge.charge_dict = {
    "GLU":-1,
    "ASP":-1,
    "LYS":1,
    "ARG":1,
    "HIS":1,
    "HIE":1
    }

#%%

def some_fun(openmm_top):

    mol = Molecule()
    # zero charge used for all atoms, regardless what charge they actually have
    # NOTE: later, find a way to infer charge from pdb, by now we only have standard versions of AAs
    idx_lookup = {} # maps from openmm atom index to openff atom index (usually they agree)
    for i, atom in enumerate(openmm_top.atoms()):
        charge = assign_standard_charge(atom_name=atom.name, residue=atom.residue.name)
        print(atom.name, atom.residue.name, charge)
        mol.add_atom(atom.element.atomic_number, formal_charge=charge, is_aromatic=False, stereochemistry=None)
        idx_lookup[atom.index] = i

    # bond_order 1 used for all bonds, regardless what type they are
    for bond in openmm_top.bonds():
        mol.add_bond(idx_lookup[bond.atom1.index], idx_lookup[bond.atom2.index], bond_order=1, is_aromatic=False, stereochemistry=None)

    return mol
# %%
mol = some_fun(top)
# %%
mol.to_rdkit()
# %%
