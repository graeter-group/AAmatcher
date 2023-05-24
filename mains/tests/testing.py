#%%
DATASET_PATH = "./../PDBData/xyz2res/scripts/data/pdbs/pep2"
from pathlib import Path

from PDBData.PDBMolecule import PDBMolecule

from openmm.app import PDBFile
from openmm.unit import angstrom
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt

#%%
def testrun(testpath=DATASET_PATH, permute=True, test_energy=False, show=False, get_mols=False, fig = True):
    """
    For all pdb at the testpath, check whether the workflow shuffled_xyz -> PDBMolecule -> pdb -> openmm pdb runs without throwing an error
    """
    fine = 0
    crashed = []
    counter = 0
    if test_energy:
        e_counter = 0
        bad_energy = []

    if get_mols:
        moldict = {}

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

        elements = np.array(elements)

        perm = np.arange(xyz.shape[1])
        np.random.seed(0)
        if permute:
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

            if test_energy:
                mse, var = data_testrun(p)
                if mse > var/10.:
                    e_counter += 1
                    bad_energy.append(residues)

            if get_mols:
                moldict[p.parent.stem] = mol

        except:
            crashed.append(residues)
        counter += 1
        print(counter, end="\r")

    print()
    print(f"ran through for {fine} of {fine+len(crashed)}")
    
    crashcount = {}

    if len(crashed) > 0:
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
        if fig:
            plt.figure(figsize=(7,4))
            plt.title("Failed crash test occurences")
            plt.bar(crashcount.keys(), crashcount.values())
            plt.savefig(f"crashes_{str(Path(testpath).stem)}.png")
        if show:
            plt.show()


    if test_energy:
        ecrashcount = {}
        if len(bad_energy) > 0:
            for l in bad_energy:
                for res in l:
                    if not res in ecrashcount.keys():
                        ecrashcount[res] = 1
                    else:
                        ecrashcount[res] += 1
            if "ACE" in ecrashcount.keys():
                ecrashcount.pop("ACE")
            if "NME" in ecrashcount.keys():
                ecrashcount.pop("NME")
            print(f"energy errors occured n times for the resp. residues\n{ecrashcount}")
            plt.figure(figsize=(7,4))
            plt.title("Failed energy test occurences")
            plt.bar(ecrashcount.keys(), ecrashcount.values())
            plt.savefig(f"energy_test_{str(Path(testpath).stem)}.png")
            if show:
                plt.show()

    if get_mols:
        return moldict



def data_testrun(pdbpath, permute=True):
    """
    create a set of configurations using openmm, then shuffle positions, elements and forces, create a PDBMolecule and use openmm to calculate the energies and forces belonging to the (re-ordered) positions of the mol, and compare these elementwise. 
    """
    from openmm.app import ForceField, PDBFile, topology, Simulation, PDBReporter, PME, HBonds, NoCutoff, CutoffNonPeriodic
    from openmm import LangevinMiddleIntegrator, Vec3
    from openmm.unit import Quantity, picosecond, kelvin, kilocalorie_per_mole, picoseconds, angstrom, nanometer, femtoseconds, newton


    pdb = PDBFile(str(pdbpath))

    residues = []

    elements = []
    for a in pdb.topology.atoms():
        elements.append(a.element.atomic_number)
        residues.append(a.residue.name)
    elements = np.array(elements)

    residues = list(set(residues))

    # generate a dataset

    integrator = LangevinMiddleIntegrator(500*kelvin, 1/picosecond, 0.5*femtoseconds)

    forcefield=ForceField('amber99sbildn.xml')
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
    np.random.seed(0)
    if permute:
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

    mse, var = np.sum(diff**2)/len(diff), np.sum((amber_e-np.mean(amber_e))**2)/len(diff)
    # print("en diff: ", mse, "~variance: ", var)
    return mse, var

    # diff2 = -new_forces - mol.gradients
    # print("f  diff: ", np.sum(diff2**2)/len(diff2), "~variance: ", np.sum(new_forces**2)/len(diff2))



#%%
# p = Path("./../PDBData/xyz2res/scripts/data/pdbs")/Path("pep2/AF/pep.pdb")
# data_testrun(p, permute=True)
    
# %%
d = testrun(Path("./../PDBData/xyz2res/scripts/data/pdbs/pep1"), test_energy=True, show=False, permute=True, get_mols=True)

#%%

d.keys()

# %%
# p = Path("./../PDBData/xyz2res/scripts/data/pdbs")/Path("pep1/G/pep.pdb")
# data_testrun(p, permute=False)
# %%
# write to graph method and draw
