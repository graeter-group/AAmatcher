#%%

from pathlib import Path

from PDBData.PDBMolecule import PDBMolecule
#%%
DATASET_PATH = Path(__file__).parent.parent/Path("PDBData/xyz2res/scripts/data/pdbs/pep1")
print(DATASET_PATH.absolute())
PDBMolecule.do_energy_checks(DATASET_PATH)

# %%
