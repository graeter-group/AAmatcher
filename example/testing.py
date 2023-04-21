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
"""
For all pdb at the testpath, check wheter the workflow xyz -> PDBMolecule -> pdb -> openmm pdb runs without throwing an error
"""

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

"""
create a dataset using openmm, then shuffle positions, elements and forces, create a mol and use openmm to calculate the energies and forces belonging to the (re-ordered) positions of the mol, and compare these elementwise. 
"""

def data_testrun(pdbpath):
    from openmm.app import ForceField, PDBFile, topology, Simulation, PDBReporter, PME, HBonds, NoCutoff, CutoffNonPeriodic
    from openmm import LangevinMiddleIntegrator, Vec3
    from openmm.unit import Quantity, picosecond, kelvin, kilocalorie_per_mole, picoseconds, angstrom, nanometer, femtoseconds, newton


    pdb = PDBFile(str(pdbpath))

    residues = []

    elements = []
    for a in pdb.topology.atoms():
        elements.append(a.element.atomic_number)
        residues.append(a.residue.name)

    residues = list(set(residues))
    if "HIS" in residues or "GLY" in residues:
        raise RuntimeError("forbidden residue")

    elements = np.array(elements)

    # generate a dataset

    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.5*femtoseconds)

    forcefield=ForceField('amber99sb.xml')
    system = forcefield.createSystem(pdb.topology)

    simulation = Simulation(
        pdb.topology, system=system, integrator=integrator
    )

    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=int(10**6))

    amber_e = []
    amber_f = []
    xyz = []
    for t in range(20):
        simulation.step(100)
        state = simulation.context.getState(
                getEnergy=True,
                getPositions=True,
                getForces=True,
            )
        e = state.getPotentialEnergy()
        e = e.value_in_unit(kilocalorie_per_mole)
        f = state.getForces(True)
        f = f.value_in_unit(kilocalorie_per_mole/angstrom)
        f = np.array(f)
        pos = state.getPositions().value_in_unit(angstrom)

        xyz.append(np.array(pos))
        amber_e.append(e)
        amber_f.append(f)

    amber_e = np.array(amber_e)
    amber_f = np.array(amber_f)
    xyz = np.array(xyz)

    perm = np.arange(xyz.shape[1])
    np.random.shuffle(perm)
    xyz = xyz[:,perm]
    elements = elements[perm]
    amber_f = amber_f[:,perm]

    # generate a pdb molecule with our package
    mol = PDBMolecule.from_xyz(xyz=xyz, elements=elements, energies=amber_e, gradients=-amber_f)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'pep.pdb')
        mol.write_pdb(path)
        gen_pdb = PDBFile(path)

    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.5*femtoseconds)
    system = forcefield.createSystem(gen_pdb.topology)
    simulation = Simulation(
        gen_pdb.topology, system=system, integrator=integrator
    )

    # compare the datasets that should match
    new_forces = []
    new_energies = []
    for pos in mol.xyz:
        simulation.context.setPositions(pos/10) # go to nanometer
        state = simulation.context.getState(
                getEnergy=True,
                getParameters=True,
                getForces=True,
            )
        e = state.getPotentialEnergy()
        e = e.value_in_unit(kilocalorie_per_mole)
        f = state.getForces(True)
        f = f.value_in_unit(kilocalorie_per_mole/angstrom)
        f = np.array(f)
        new_energies.append(e)
        new_forces.append(f)

    new_forces = np.array(new_forces)
    new_energies = np.array(new_energies)

    diff = new_energies - mol.energies
    print("diff: ", np.sum(diff**2)/len(diff), "variance: ", np.sum((amber_e-np.mean(amber_e))**2)/len(diff))


    diff2 = -new_forces - mol.gradients
    print("diff:,", np.sum(diff2**2)/len(diff2), "~variance: ", np.sum(new_forces**2)/len(diff2))



#%%
p = Path("/hits/fast/mbm/seutelf/data/raw_pdb_files")/Path("pep2/AA/pep.pdb")
data_testrun(p)
    
# %%
c = testrun()
# seems to crash for pro and phe