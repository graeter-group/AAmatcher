#%%

from AAmatcher.PDBMolecule import PDBMolecule

import numpy as np

#%%

# positions and elements describing capped Ala_2:
xyz = np.array([[[ 1.179,  2.188,  0.189],
        [ 5.396,  3.746,  1.707],
        [ 4.395,  2.218,  3.455],
        [ 1.048,  0.238, -0.245],
        [ 3.85 ,  2.297,  0.431],
        [-0.762,  1.11 ,  0.49 ],
        [-1.005, -0.17 , -0.69 ],
        [ 7.386,  4.677,  1.68 ],
        [ 9.281,  3.844,  1.119],
        [ 5.752,  2.702,  1.788],
        [ 4.164,  2.042, -2.353],
        [ 3.428,  3.387,  0.835],
        [ 3.5  ,  0.426, -0.502],
        [ 6.   ,  2.766,  3.976],
        [ 9.866,  3.59 ,  1.99 ],
        [ 5.363,  0.99 ,  0.537],
        [ 7.828,  1.662,  1.156],
        [ 3.126,  1.974, -1.98 ],
        [ 5.79 ,  1.121,  3.329],
        [ 7.222,  2.701,  1.443],
        [ 5.472,  2.175,  3.206],
        [ 7.866,  3.819,  1.448],
        [ 9.564,  4.83 ,  0.781],
        [ 1.662,  1.368, -0.15 ],
        [ 1.659, -0.755, -0.662],
        [ 3.068,  1.444, -0.538],
        [ 9.491,  3.131,  0.335],
        [ 4.995,  1.877,  0.851],
        [-0.41 ,  0.146,  0.154],
        [-0.529, -0.569,  0.955],
        [ 2.68 ,  2.984, -2.071],
        [ 2.578,  1.316, -2.682]]])

elements = np.array([1, 1, 1, 6, 6, 1, 1, 1, 6, 6, 1, 8, 1, 1, 1, 1, 8, 6, 1, 6, 6, 7,
       1, 7, 8, 6, 1, 7, 6, 1, 1, 1])

# use positiions and elements to create a pdb file
mol = PDBMolecule.from_xyz(xyz=xyz, elements=elements)

# print the content of the pdb file
print("\n", *mol.pdb, "\n")

# write the pdb file
# mol.write_pdb("my_pdb.pdb")

#%%
# this also works after shuffling the order of atoms randomly:
perm = np.arange(len(elements))
np.random.shuffle(perm)
elements = elements[perm]
xyz = xyz[:,perm]

mol = PDBMolecule.from_xyz(xyz=xyz, elements=elements)

print("\n", *mol.pdb[:5], "...", "\n")
#%%

# suppose we have energies and gradients, we can also use the class to store a dataset:
# create 100 energies and gradients
energies = np.random.randn(100)
gradients = np.random.randn(100, len(elements), 3)
# create 99 (unphysical) conformations of the molecule
# The zeroth conformation is used to create the molecular graph for the pdb and must therefore be physical
xyz = np.concatenate((xyz, np.random.randn(99, len(elements), 3)), axis=0)

mol = PDBMolecule.from_xyz(xyz=xyz, elements=elements, energies=energies, gradients=gradients)

# obtain the data of the 20th to 30th conformation
pos, en, grad = mol[20:30]
print("pos shape: ", pos.shape, "en shape: ", en.shape, "grad shape: ", grad.shape)

print("\n", *mol.pdb[:5], "...", "\n")
# %%
# we can also store and recover the dataset:
mol.save("data.npz")
new_mol = PDBMolecule.load("data.npz")
print(np.all(new_mol.xyz == mol.xyz))