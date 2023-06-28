#%%
from pathlib import Path
from PDBData.PDBMolecule import PDBMolecule
doppath = Path(__file__).parent / Path("AJ.pdb")
# doppath = Path(__file__).parent / Path("F.pdb")

mol = PDBMolecule.from_pdb(doppath)
#%%
# matching works fine:

xyz = mol.xyz
elements = mol.elements
mol_xyz = PDBMolecule.from_xyz(xyz=xyz, elements=elements)
mol_xyz.bond_check(perm=True)

# the forcefield not for some reason, even without permuting:

#%%
l = []
mol.energy_check(collagen=True, n_conf=100, permute=False, store_energies_ptr=l)
# %%
f1, f2 = l[2], l[3]
# %%
diff = f1 - f2
import numpy as np
np.argwhere(diff > 1e-1)

# %%
