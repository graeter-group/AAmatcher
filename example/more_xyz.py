#%%

from AAmatcher.PDBMolecule import PDBMolecule

from openmm.app import PDBFile
from openmm.unit import angstrom
import numpy as np
#%%

pdb = PDBFile("some_peptides/CY/pep.pdb")

xyz = pdb.positions
xyz = xyz.value_in_unit(angstrom)
xyz = np.array([[np.array(v) for v in xyz]])

elements = []
for a in pdb.topology.atoms():
    elements.append(a.element.atomic_number)

elements = np.array(elements)

perm = np.arange(xyz.shape[1])
np.random.shuffle(perm)
xyz = xyz[:,perm]
elements = elements[perm]
print("elements[:5]: ", elements[:5])
print("xyz[:,:5]: ", xyz[:,:5])
#%%

mol = PDBMolecule.from_xyz(xyz=xyz, elements=elements, logging=True)

print(*mol.pdb)

# %%
