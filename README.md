# AAmatcher

## General Information
Provides a class, PDBMolecule, to store and convert datasets (positions, energies, forces) of peptides, keeping a PDB file describing connectivity and residue membership. The PDB file is automatically generated upon initialization from positions and elements, without requireing a specific atom order. This is done using a graph neural network to classify c alpha atoms and an algorithmic approach to identify the residues from this information.

The class can also be used to create PDB files from xyz coordinates and elements alone. See examples/from_xyz.py for an instructive tutorial.

Currently, residue prediction is only possible for the standard versions of the amino acids and not guaranteed to work for molecules larger than capped dipeptides.

## Installation
- `git clone https://github.com/hits-mbm-dev/AAmatcher.git`
- `git checkout version-0.0.2`
- `cd AAmatcher`
- `pip install -e .`

## Requirements
- ase
- numpy
- pytorch
- deep graph library
- openmm
- openff-toolkit

## Usage

### Python
```
from AAmatcher.PDBMolecule import PDBMolecule
mol = PDBMolecule.from_xyz(xyz=xyz, elements=elements)
mol.write_pdb("my_pdb.pdb")
```

```
from AAmatcher.PDBMolecule import PDBMolecule
mol = PDBMolecule.from_xyz(xyz=xyz, elements=elements, energies=energies, gradients=gradients)
```
