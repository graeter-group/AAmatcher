#%%
from openff.toolkit.topology import Molecule
from PDBData.PDBDataset import PDBDataset
dspath = "/hits/fast/mbm/seutelf/data/datasets/PDBDatasets/spice_amber99sbildn"
# %%
ds = PDBDataset.load_npz(dspath, n_max=10)
#%%
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
# bond check for radicals should fail:
try:
    check_bonds(m_rad, m2)
    raise RuntimeError
except AssertionError:
    pass
# %%

from PDBData.PDBMolecule import PDBMolecule
new_rad = PDBMolecule.from_pdb("F_rad.pdb")
# %%
import numpy as np
g = new_rad.to_dgl()
assert all(np.array(g.nodes["n1"].data["h0"].shape) == np.array(PDBMolecule.from_pdb("F.pdb").to_dgl().nodes["n1"].data["h0"].shape) - np.array([1, 0]))
# %%
