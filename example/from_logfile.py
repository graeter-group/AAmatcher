#%%

from AAmatcher.PDBMolecule import PDBMolecule

from openmm.app import PDBFile
from openmm.unit import angstrom
import numpy as np

#%%

logfile = "g09_dataset/Ala_nat_opt.log"
mol_log = PDBMolecule.from_gaussian_log(logfile, logging=True)
# %%
xyz = mol_log.xyz[:5]
#%%
print(mol_log.energies)
# %%
