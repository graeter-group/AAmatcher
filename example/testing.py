#%%
DATASET_PATH = "/hits/fast/mbm/seutelf/data/raw_pdb_files/pep5"
from pathlib import Path

from AAmatcher.PDBMolecule import PDBMolecule

from openmm.app import PDBFile
from openmm.unit import angstrom
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt


#%%
def testrun(testpath=DATASET_PATH):
    fine = 0
    crashed = []
    counter = 0
    counter

    for p in Path(testpath).rglob("*.pdb"):
        pdb = PDBFile(str(p))

        residues = []

        xyz = pdb.positions
        xyz = xyz.value_in_unit(angstrom)
        xyz = np.array([[np.array(v) for v in xyz]])

        elements = []
        for a in pdb.topology.atoms():
            elements.append(a.element.atomic_number)
            residues.append(a.residue.name)

        residues = list(set(residues))
        if "HIS" in residues or "GLY" in residues:
            continue

        elements = np.array(elements)

        perm = np.arange(xyz.shape[1])
        np.random.shuffle(perm)
        xyz = xyz[:,perm]
        elements = elements[perm]

        try:
            mol = PDBMolecule.from_xyz(xyz=xyz, elements=elements)
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, 'pep.pdb')
                mol.write_pdb(path)
                pdb = PDBFile(path)
            fine += 1
        except:
            crashed.append(residues)
        counter += 1
        print(counter, end="\r")

    print()
    print(f"ran through for {fine} of {fine+len(crashed)}")
    
    crashcount = {}
    for l in crashed:
        for res in l:
            if not res in crashcount.keys():
                crashcount[res] = 1
            else:
                crashcount[res] += 1
    if "ACE" in crashcount.keys():
        crashcount.pop("ACE")
    if "NME" in crashcount.keys():
        crashcount.pop("NME")
    print(f"errors occured n times for the resp. residues\n{crashcount}")
    plt.figure(figsize=(7,4))
    plt.bar(crashcount.keys(), crashcount.values())
    plt.show()
    return crashcount


# %%
c = testrun()

# seems to crash for pro and phe
# %%
