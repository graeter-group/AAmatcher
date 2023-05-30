#%%
spicepath = "/hits/fast/mbm/seutelf/data/datasets/SPICE-1.1.2.hdf5"

from pathlib import Path
from openmm.unit import bohr, Quantity, hartree, mole
import numpy as np
from PDBData.PDBMolecule import PDBMolecule

from PDBData.units import DISTANCE_UNIT, ENERGY_UNIT, FORCE_UNIT
import torch

#%%
m = PDBMolecule.from_pdb("F.pdb")
#%%
m.parametrize()
g = m.to_dgl()
#%%
g.nodes['n1'].data.keys()
#%%
g.nodes['n1'].data['h0'].shape
#%%

PARTICLE = mole.create_unit(
    6.02214076e23 ** -1,
    "particle",
    "particle",
)

HARTREE_PER_PARTICLE = hartree / PARTICLE

SPICE_DISTANCE = bohr
SPICE_ENERGY = HARTREE_PER_PARTICLE
SPICE_FORCE = SPICE_ENERGY / SPICE_DISTANCE

name = "hie-hie"
import h5py
with h5py.File(spicepath, "r") as f:
    elements = f[name]["atomic_numbers"]
    energies = f[name]["dft_total_energy"]
    xyz = f[name]["conformations"] # in bohr
    gradients = f[name]["dft_total_gradient"]
    # convert to kcal/mol and kcal/mol/angstrom
    elements = np.array(elements, dtype=np.int64)
    xyz = Quantity(np.array(xyz), SPICE_DISTANCE).value_in_unit(DISTANCE_UNIT)
    gradients = Quantity(np.array(gradients), SPICE_FORCE).value_in_unit(FORCE_UNIT)
    energies = Quantity(np.array(energies)-np.array(energies).mean(axis=-1), SPICE_ENERGY).value_in_unit(ENERGY_UNIT)

m = PDBMolecule.from_xyz(xyz, elements, energies, gradients)
m.parametrize()
g = m.to_dgl()
#%%
#%%
m.compress("tmp.npz")
loaded = PDBMolecule.load("tmp.npz")
# loading and saving works:
assert np.all(loaded.xyz == m.xyz)
assert np.all(loaded.energies == m.energies)
assert torch.allclose(g.nodes['g'].data['u_qm'], loaded.to_dgl().nodes['g'].data['u_qm']), f"{g.nodes['g'].data['u_qm']}\n{loaded.to_dgl().nodes['g'].data['u_qm']}"
assert torch.allclose(g.nodes['n1'].data['grad_total_amber99sbildn'], loaded.to_dgl().nodes['n1'].data['grad_total_amber99sbildn'])
assert "u_ref" in loaded.to_dgl().nodes['g'].data.keys(), loaded.to_dgl().nodes['g'].data.keys()
# %%
