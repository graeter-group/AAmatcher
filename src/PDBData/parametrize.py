#%%
# inspired from espaloma:


# MIT License

# Copyright (c) 2020 Yuanqing Wang @ choderalab // MSKCC

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch

import openmm
from openmm import unit
from openmm.app import Simulation
from openmm.app import ForceField

from openmm.unit import Quantity
from . import units
import copy
import dgl

# simulation specs, only dummies for force and energy calculation
TEMPERATURE = 350 * unit.kelvin
STEP_SIZE = 1.0 * unit.femtosecond
COLLISION_RATE = 1.0 / unit.picosecond
EPSILON_MIN = 0.05 * unit.kilojoules_per_mole




# write all classical ff parameters in the graph (except for improper torsion)
# then, calculate the total, bonded and nonbonded energy contributions
# by gradually turning off parameters. the "nonbonded" entry corresponds
# to the energy with bond, angle and torsion params set to zero.
# set suffix to eg "_amber99sbildn"

# with forces: contains "grad_nonbonded"+suffix
# WRITES GRADIENTS, NOT FORCES!
# returns graph
# where conf data is a dictionary containing conformational data
# get_charges is a function that takes a topology and returns a list of charges as openmm Quantities in the order of the atoms in topology
def parametrize_amber(g, topology, forcefield=ForceField('amber99sbildn.xml'), n_max_periodicities=6, suffix="_amber99sbildn",
parameters=["charge", "LJ", "bond", "angle", "torsion"],
energies=["charge", "LJ", "bond", "angle", "torsion", "improper", "nonbonded", "total", "bonded"], forces=True, write_data=True, get_charges=None):
    """
    get_charges: Function that takes a topology and returns a list of charges as openmm Quantities in the order of the atoms in topology.
    """
    ZERO_CHARGE = 1e-20 # used to supress exception when setting charges to zero

    manual_charges = None
    if not get_charges is None:
        manual_charges = get_charges(topology)
        manual_charges = torch.tensor([c.value_in_unit(units.CHARGE_UNIT) for c in manual_charges], dtype=torch.float32)

    sys = forcefield.createSystem(topology=topology)


    # PARAMETRISATION START

    atom_lookup = {
        idxs.detach().numpy().item(): position
        for position, idxs in enumerate(g.nodes["n1"].data["idxs"])
    }

    bond_lookup = {
        tuple(idxs.detach().numpy()): position
        for position, idxs in enumerate(g.nodes["n2"].data["idxs"])
    }

    angle_lookup = {
        tuple(idxs.detach().numpy()): position
        for position, idxs in enumerate(g.nodes["n3"].data["idxs"])
    }

    torsion_lookup = {
        tuple(idxs.detach().numpy()): position
        for position, idxs in enumerate(g.nodes["n4"].data["idxs"])
    }

    improper_lookup = {
        tuple(idxs.detach().numpy()): position
        for position, idxs in enumerate(
            g.nodes["n4_improper"].data["idxs"]
        )
    }

    torsion_phases = torch.zeros(
        g.number_of_nodes("n4"),
        n_max_periodicities,
    )

    torsion_periodicities = torch.zeros(
        g.number_of_nodes("n4"),
        n_max_periodicities,
    )

    torsion_ks = torch.zeros(
        g.number_of_nodes("n4"),
        n_max_periodicities,
    )

    improper_phases = torch.zeros(
        g.number_of_nodes("n4_improper"),
        n_max_periodicities,
    )

    improper_periodicities = torch.zeros(
        g.number_of_nodes("n4_improper"),
        n_max_periodicities,
    )

    improper_ks = torch.zeros(
        g.number_of_nodes("n4_improper"),
        n_max_periodicities,
    )

    if not len(parameters) == 0:

        for force in sys.getForces():
            name = force.__class__.__name__
            if "NonbondedForce" in name:
                assert (
                    force.getNumParticles()
                    == g.number_of_nodes("n1")
                )

                if "LJ" in parameters:

                    g.nodes["n1"].data["epsilon%s"%suffix] = torch.zeros(
                        force.getNumParticles(), 1, dtype=torch.float32,
                    )

                    g.nodes["n1"].data["sigma%s"%suffix] = torch.zeros(
                        force.getNumParticles(), 1, dtype=torch.float32,
                    )

                if "charge" in parameters:

                    g.nodes["n1"].data["q%s"%suffix] = torch.zeros(
                        force.getNumParticles(), 1, dtype=torch.float32,
                    )

                for idx in range(force.getNumParticles()):
                    charge, sigma, epsilon = force.getParticleParameters(idx)

                    position = atom_lookup[idx]

                    if "LJ" in parameters:

                        g.nodes["n1"].data["epsilon%s"%suffix][position] = epsilon.value_in_unit(
                            units.ENERGY_UNIT,
                        )
                        g.nodes["n1"].data["sigma%s"%suffix][position] = sigma.value_in_unit(
                            units.DISTANCE_UNIT,
                        )

                    if "charge" in parameters:
                        if not manual_charges is None:
                            g.nodes["n1"].data["q%s"%suffix][position] = manual_charges[position].value_in_unit(units.CHARGE_UNIT)
                        else:
                            g.nodes["n1"].data["q%s"%suffix][position] = charge.value_in_unit(
                                units.CHARGE_UNIT,
                            )


            if "HarmonicBondForce" in name:
                assert (
                    force.getNumBonds() * 2
                    == g.number_of_nodes("n2")
                )

                if not "bond" in parameters:
                    continue

                g.nodes["n2"].data["eq%s"%suffix] = torch.zeros(
                    force.getNumBonds() * 2, 1, dtype=torch.float32,
                )

                g.nodes["n2"].data["k%s"%suffix] = torch.zeros(
                    force.getNumBonds() * 2, 1, dtype=torch.float32,
                )

                for idx in range(force.getNumBonds()):
                    idx0, idx1, eq, k = force.getBondParameters(idx)

                    position = bond_lookup[(idx0, idx1)]
                    g.nodes["n2"].data["eq%s"%suffix][position] = eq.value_in_unit(
                        units.DISTANCE_UNIT,
                    )
                    g.nodes["n2"].data["k%s"%suffix][position] = k.value_in_unit(
                        units.FORCE_CONSTANT_UNIT,
                    )

                    position = bond_lookup[(idx1, idx0)]
                    g.nodes["n2"].data["eq%s"%suffix][position] = eq.value_in_unit(
                        units.DISTANCE_UNIT,
                    )
                    g.nodes["n2"].data["k%s"%suffix][position] = k.value_in_unit(
                        units.FORCE_CONSTANT_UNIT,
                    )

            if "HarmonicAngleForce" in name:
                assert (
                    force.getNumAngles() * 2
                    == g.number_of_nodes("n3")
                )

                if not "angle" in parameters:
                    continue

                g.nodes["n3"].data["eq%s"%suffix] = torch.zeros(
                    force.getNumAngles() * 2, 1, dtype=torch.float32,
                )

                g.nodes["n3"].data["k%s"%suffix] = torch.zeros(
                    force.getNumAngles() * 2, 1, dtype=torch.float32,
                )

                for idx in range(force.getNumAngles()):
                    idx0, idx1, idx2, eq, k = force.getAngleParameters(idx)

                    position = angle_lookup[(idx0, idx1, idx2)]
                    g.nodes["n3"].data["eq%s"%suffix][position] = eq.value_in_unit(
                        units.ANGLE_UNIT,
                    )
                    g.nodes["n3"].data["k%s"%suffix][position] = k.value_in_unit(
                        units.ANGLE_FORCE_CONSTANT_UNIT,
                    )

                    position = angle_lookup[(idx2, idx1, idx0)]
                    g.nodes["n3"].data["eq%s"%suffix][position] = eq.value_in_unit(
                        units.ANGLE_UNIT,
                    )
                    g.nodes["n3"].data["k%s"%suffix][position] = k.value_in_unit(
                        units.ANGLE_FORCE_CONSTANT_UNIT,
                    )

            if "PeriodicTorsionForce" in name:
                if not "torsion" in parameters:
                    continue

                for idx in range(force.getNumTorsions()):
                    (
                        idx0,
                        idx1,
                        idx2,
                        idx3,
                        periodicity,
                        phase,
                        k,
                    ) = force.getTorsionParameters(idx)
                    
                    if (idx0, idx1, idx2, idx3) in torsion_lookup.keys():
                        # implement symmetry under reversal
                        for index_tuple in [(idx0, idx1, idx2, idx3), (idx3, idx2, idx1, idx0)]:
                            position = torsion_lookup[index_tuple]
    
                            torsion_ks[position, periodicity-1] = k.value_in_unit(
                                units.ENERGY_UNIT
                            )

                            torsion_phases[position, periodicity-1] = phase.value_in_unit(units.ANGLE_UNIT)

                            torsion_periodicities[position, periodicity-1] = periodicity


                    # for improper torsion, put amber index 2 at espaloma position 1 (see below)
                    elif (idx0, idx2, idx1, idx3) in improper_lookup.keys():
                        # seems like the improper torsions are ordered such that the middle atom is at index 1 in original espaloma while it is at 2 in amber (acc to https://github.com/openmm/openmm/issues/220 amber is not unique with the order of the non-centering atoms. this does not matter, one can just pick the fitting one from a cyclic permutation of params in the espaloma graph when replacing parameters since the dihedral angle and therefore impr torsion is invariant under cyclic permutation of the outer atoms)

                        # implement symmetry under cyclic permutations
                        for index_tuple in [(idx0, idx2, idx1, idx3), (idx3, idx2, idx0, idx1), (idx1, idx2, idx3, idx0)]:
                            position = improper_lookup[index_tuple]
    
                            improper_ks[position, periodicity-1] = k.value_in_unit(
                                units.ENERGY_UNIT
                            )

                            improper_phases[position, periodicity-1] = phase.value_in_unit(units.ANGLE_UNIT)

                            improper_periodicities[position, periodicity-1] = periodicity

                    else:
                        raise RuntimeError("A torsion parameter could not be assigned since the participating atom indices are not listed in the input graph")
            
            g.nodes["n4"].data["k_original"+suffix] = torsion_ks
            g.nodes["n4"].data["periodicity"+suffix] = torsion_periodicities
            g.nodes["n4"].data["phases"+suffix] = torsion_phases

            g.nodes["n4"].data["k"+suffix] = torch.where(torsion_phases==0., torsion_ks, -torsion_ks)


            g.nodes["n4_improper"].data["k_original"+suffix] = improper_ks
            g.nodes["n4_improper"].data["periodicity"+suffix] = improper_periodicities
            g.nodes["n4_improper"].data["phases"+suffix] = improper_phases
            
            g.nodes["n4_improper"].data["k"+suffix] = torch.where(improper_phases==0., improper_ks, -improper_ks)


    # PARAMETRISATION END

    if not write_data:
        return g

    if not len(energies) == 0 and "xyz" in g.nodes["n1"].data.keys():

        # initialize an integrator to be able to initialize a simulation to calculate energies:
        integrator = openmm.LangevinIntegrator(
            TEMPERATURE, COLLISION_RATE, STEP_SIZE
        )
            # create simulation
        simulation = Simulation(
            topology=topology, system=sys, integrator=integrator
        )

        # the snapshots
        xs =Quantity(
                g.nodes["n1"].data["xyz"].detach().numpy().transpose((1, 0, 2)),
                units.DISTANCE_UNIT,
            ).value_in_unit(unit.nanometer)
    
        if not manual_charges is None:
            # set charges:
            for force in sys.getForces():
                name = force.__class__.__name__
                # calculate charge contribution
                if "NonbondedForce" in name:
                    # COULOMB
                    for idx in range(force.getNumParticles()):
                        q = manual_charges[idx]
                        _, sigma, epsilon = force.getParticleParameters(idx)
                        force.setParticleParameters(idx, q, sigma, epsilon)
                    for idx in range(force.getNumExceptions()):
                        (
                            idx0,
                            idx1,
                            q,
                            sigma,
                            epsilon,
                        ) = force.getExceptionParameters(idx)
                        force.setExceptionParameters(
                            idx, idx0, idx1, q, sigma, epsilon
                        )

                    force.updateParametersInContext(simulation.context)

        total_energies, total_gradients = get_energies_(simulation, xs)


        # turn off angle:
        for force in sys.getForces():
            name = force.__class__.__name__
            if "Angle" in name:
                for idx in range(force.getNumAngles()):
                    id1, id2, id3, angle, k = force.getAngleParameters(idx)
                    force.setAngleParameters(idx, id1, id2, id3, angle, 0.0)

                force.updateParametersInContext(simulation.context)
        energies_now, _ = get_energies_(simulation, xs)
        energies_angle = total_energies - energies_now
        
        # turn off bond:
        for force in sys.getForces():
            name = force.__class__.__name__
            if "Bond" in name:
                for idx in range(force.getNumBonds()):
                    id1, id2, length, k = force.getBondParameters(idx)
                    force.setBondParameters(
                        idx,
                        id1,
                        id2,
                        length,
                        0.0,
                    )

                force.updateParametersInContext(simulation.context)
        energies_prev = copy.deepcopy(energies_now)
        energies_now, _ = get_energies_(simulation, xs)
        energies_bond = energies_prev - energies_now
        
        # turn off torsion:
        for force in sys.getForces():
            name = force.__class__.__name__
            # also contains improper torsions
            if "Torsion" in name:
                for idx in range(force.getNumTorsions()):
                    (
                        id1,
                        id2,
                        id3,
                        id4,
                        periodicity,
                        phase,
                        k,
                    ) = force.getTorsionParameters(idx)
                    if (id1, id2, id3, id4) in torsion_lookup.keys():
                        force.setTorsionParameters(
                            idx,
                            id1,
                            id2,
                            id3,
                            id4,
                            periodicity,
                            phase,
                            0.0,
                        )
                force.updateParametersInContext(simulation.context)

        energies_prev = copy.deepcopy(energies_now)
        energies_now, _ = get_energies_(simulation, xs)
        energies_torsion = energies_prev - energies_now

        # turn off improper torsion:
        for force in sys.getForces():
            name = force.__class__.__name__
            # also contains improper torsions
            if "Torsion" in name:
                for idx in range(force.getNumTorsions()):
                    (
                        id1,
                        id2,
                        id3,
                        id4,
                        periodicity,
                        phase,
                        k,
                    ) = force.getTorsionParameters(idx)
                    force.setTorsionParameters(
                        idx,
                        id1,
                        id2,
                        id3,
                        id4,
                        periodicity,
                        phase,
                        0.0,
                    )
                force.updateParametersInContext(simulation.context)

        energies_prev = copy.deepcopy(energies_now)
        energies_now, gradients_nonbonded = get_energies_(simulation, xs)
        energies_improper_torsion = energies_prev - energies_now
        energies_nonbonded = energies_now

        # turn off charges:
        for force in sys.getForces():
            name = force.__class__.__name__
            # calculate charge contribution
            if "NonbondedForce" in name:
                # COULOMB
                for idx in range(force.getNumParticles()):
                    q, sigma, epsilon = force.getParticleParameters(idx)
                    force.setParticleParameters(idx, q * ZERO_CHARGE, sigma, epsilon)
                for idx in range(force.getNumExceptions()):
                    (
                        idx0,
                        idx1,
                        q,
                        sigma,
                        epsilon,
                    ) = force.getExceptionParameters(idx)
                    force.setExceptionParameters(
                        idx, idx0, idx1, q * ZERO_CHARGE, sigma, epsilon
                    )

                force.updateParametersInContext(simulation.context)
        energies_prev = copy.deepcopy(energies_now)
        energies_now, _ = get_energies_(simulation, xs)
        energies_coulomb = energies_prev - energies_now

        # turn off LJ:
        for force in sys.getForces():
            name = force.__class__.__name__
            # calculate charge contribution
            if "NonbondedForce" in name:
                # LJ
                for idx in range(force.getNumParticles()):
                    q, sigma, epsilon = force.getParticleParameters(idx)
                    force.setParticleParameters(idx, q, 0, 0)
                for idx in range(force.getNumExceptions()):
                    (
                        idx0,
                        idx1,
                        q,
                        sigma,
                        epsilon,
                    ) = force.getExceptionParameters(idx)
                    force.setExceptionParameters(
                        idx, idx0, idx1, q, 0, 0
                    )

                force.updateParametersInContext(simulation.context)
        energies_prev = copy.deepcopy(energies_now)
        energies_now, _ = get_energies_(simulation, xs)
        energies_LJ = energies_prev - energies_now


        # write the energies to the graph:
        if "total" in energies:
            g.nodes["g"].data["u_total"+suffix] = total_energies
            if forces:
                g.nodes["n1"].data["grad_total"+suffix] = total_gradients

        if "nonbonded" in energies:
            g.nodes["g"].data["u_nonbonded"+suffix] = energies_nonbonded
            if forces:
                g.nodes["n1"].data["grad_nonbonded"+suffix] = gradients_nonbonded

        if "bond" in energies:
            g.nodes["g"].data["u_bond"+suffix] = energies_bond

        if "angle" in energies:
            g.nodes["g"].data["u_angle"+suffix] = energies_angle

        if "torsion" in energies:
            g.nodes["g"].data["u_torsion"+suffix] = energies_torsion

        if "improper" in energies:
            g.nodes["g"].data["u_improper"+suffix] = energies_improper_torsion

        if "charge" in energies:
            g.nodes["g"].data["u_charge"+suffix] = energies_coulomb

        if "LJ" in energies:
            g.nodes["g"].data["u_LJ"+suffix] = energies_LJ
        
        if "bonded" in energies:
            bonded = total_energies - energies_nonbonded
            g.nodes["g"].data["u_bonded"+suffix] = bonded

    return g


# helper function:
def get_energies_(simulation, xs):
    import numpy as np

    # loop through the snapshots
    energies = []
    derivatives = []

    for x in xs:
        simulation.context.setPositions(x)

        state = simulation.context.getState(
            getEnergy=True,
            getParameters=True,
            getForces=True,
        )

        energy = state.getPotentialEnergy().value_in_unit(
            units.ENERGY_UNIT,
        )


        forces = state.getForces(asNumpy=True).value_in_unit(
            units.FORCE_UNIT,
        )

        energies.append(energy)
        derivatives.append(-forces)

    energies = torch.tensor(np.expand_dims(np.array(energies),axis=0), dtype=torch.float32)
    derivatives = torch.tensor(np.array(derivatives), dtype=torch.float32).transpose(0,1)

    return energies, derivatives

#%%
