#%%
from openff.toolkit.topology import Molecule
from PDBData.PDBDataset import PDBDataset
dspath = "/hits/fast/mbm/seutelf/data/datasets/PDBDatasets/spice_amber99sbildn"
# %%
ds = PDBDataset.load_npz(dspath, n_max=10)
# %%
import copy
mol = copy.deepcopy(ds[0])
o = mol.to_openff()
#%%
# BOND CHECK
ids = [[b.atom1_index, b.atom2_index] for b in o.bonds]
_ = [b.sort() for b in ids]
ids = [tuple(b) for b in ids]

m = mol.to_openmm()
ids_openmm = [[b[0].index, b[1].index] for b in m.topology.bonds()]
_ = [b.sort() for b in ids_openmm]
ids_openmm = [tuple(b) for b in ids_openmm]

assert set(ids) == set(ids_openmm)
#%%
# remove HB1
mol.pdb.pop(11)
mol.pdb
m = mol.to_openmm()
# %%
# mol.to_openff() # doesnt work!
# %%
by_hand = Molecule()

print("creating a bad molecule to test how restrictive openff is:")
by_hand.add_atom(
    atomic_number = 8, # Atomic number 8 is Oxygen
    formal_charge = 0, # Formal negative charge
    is_aromatic = False, # Atom is not part of an aromatic system
    stereochemistry = None, # Optional argument; "R" or "S" stereochemistry
    name = "O" # Optional argument; descriptive name for the atom
)
by_hand.add_atom(6,  0, False, name="C")
by_hand.add_atom(8,  0, False, name="O")
by_hand.add_atom(1,  0, False, name="H")
by_hand.add_atom(1,  0, False, name="H")

by_hand.add_bond( 
    atom1 = 0, # First (zero-indexed) atom specified above ("O-")  
    atom2 = 1, # Second atom specified above ("C")
    bond_order = 1, # Single bond
    is_aromatic = False, # Bond is not aromatic
    stereochemistry = None, # Optional argument; "E" or "Z" stereochemistry
    fractional_bond_order = None # Optional argument; Wiberg (or similar) bond order
)

by_hand.add_bond(1, 2, 1, False)
by_hand.add_bond(2, 3, 1, False)
by_hand.add_bond(1, 4, 1, False)
by_hand
# %%
import numpy as np
def angle_indices(offmol: Molecule) -> np.ndarray:
    return np.array(
        sorted(
            [
                tuple([atom.molecule_atom_index for atom in angle])
                for angle in offmol.angles
            ]
        )
    )
angle_indices(by_hand)
# %%
def proper_torsion_indices(offmol: Molecule) -> np.ndarray:
    return np.array(
        sorted(
            [
                tuple([atom.molecule_atom_index for atom in proper])
                for proper in offmol.propers
            ]
        )
    )
proper_torsion_indices(by_hand)
# %%

def improper_torsion_indices(offmol: Molecule) -> np.ndarray:


    ## Find all atoms bound to exactly 3 other atoms

    ## This finds all orderings, which is what we want for the espaloma case
    ##  but not for smirnoff
    improper_smarts = '[*:1]~[X3:2](~[*:3])~[*:4]'
    mol_idxs = offmol.chemical_environment_matches(improper_smarts)
    return np.array(mol_idxs)
improper_torsion_indices(by_hand)
# %%
print("conclusion: for radicals this works,\nbut the number of bonds cannot be greater than the valence of the atom")
# %%
from rdkit import Chem
atom = by_hand.to_rdkit().GetAtoms()[0]
atom
# %%
atom.GetMass()
# %%
from PDBData.utils.utils import pdb2openff_graph, pdb2openff

# %%
m = pdb2openff_graph("F.pdb")
m_rad = pdb2openff_graph("F_rad.pdb")
m2 = pdb2openff("F.pdb")
# %%
m
# %%
def check_bonds(m1, m2):
    ids = [[b.atom1_index, b.atom2_index] for b in m1.bonds]
    _ = [b.sort() for b in ids]
    ids = [tuple(b) for b in ids]

    ids_ = [[b.atom1_index, b.atom2_index] for b in m2.bonds]
    _ = [b.sort() for b in ids_]
    ids_ = [tuple(b) for b in ids_]

    assert set(ids) == set(ids_), f"in m1: but not in m2: {set(ids) - set(ids_)}\nin m2: but not in m1: {set(ids_) - set(ids)}"
#%%
check_bonds(m, m2)
check_bonds(m_rad, m2) # should fail
# %%
mr = m_rad.to_rdkit()
# aromatic canot be inferred, ring membership can
for atom in mr.GetAtoms():
    print(atom.IsInRing(), atom.IsInRingSize(6), atom.GetIsAromatic(), atom.GetTotalDegree(), atom.GetAtomicNum())
# %%


# CHECK FORCE FIELD COMPATIBILITY
from PDBData.PDBMolecule import PDBMolecule
new_rad = PDBMolecule.from_pdb("F_rad.pdb")
# %%
g = new_rad.to_dgl()
g.nodes["n1"].data["h0"].shape
# new_rad.parametrize()
# %%
