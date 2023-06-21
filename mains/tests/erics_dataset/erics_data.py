#%%

from PDBData.PDBDataset import PDBDataset
from PDBData.PDBMolecule import PDBMolecule
from openmm.unit import kilocalorie_per_mole, angstrom
from pathlib import Path
from PDBData.charge_models.charge_models import model_from_dict
import copy
import shutil
import os

from pathlib import Path
dpath = "/hits/fast/mbm/share/datasets_Eric"
storepath = "/hits/fast/mbm/seutelf/data/datasets/PDBDatasets"
#%%

overwrite = False
n_max = None

for pathname in [
                # "AA_opt_nat",
                # "AA_scan_nat",
                "AA_opt_rad",
                # "AA_scan_rad",
                ]:
    ds = PDBDataset([])
    counter = 0


    for p in (Path(dpath)/Path(pathname)).rglob("*.log"):
        try:
            mol = PDBMolecule.from_gaussian_log_rad(p, e_unit=kilocalorie_per_mole*23.0609, dist_unit=angstrom, force_unit=kilocalorie_per_mole*23.0609/angstrom)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Failed to load {p}:\n  e: {e}")
            print()
            if not any([skip_res in str(p) for skip_res in ["Glu", "Asp", "Ile", "Trp", "NmeCH3", "Nme", "Hie", "AceCH3", "Hyp", "Dop", "Nala"]]):
                raise
            continue

        # filter out conformations with energies above 200 kcal/mol away from the minimum
        mol.filter_confs(max_energy=200, max_force=500)

        try:
            mol.bond_check()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Failed bond check for {p}:\n  e: {e}")
            print()
            continue

        try:
            mol_copy = copy.deepcopy(mol)
            mol_copy.parametrize(allow_radicals=True, get_charges=model_from_dict(tag="bmk"))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Failed to parametrize {p}:\n  e: {e}")
            print()
            raise
            continue

        if mol.energies is None:
            print(f"No energies present for {p}.")
            print()
            continue

        if mol.gradients is None:
            print(f"No energies present for {p}.")
            print()
            continue

        print(f"appending {counter}")
        ds.append(mol)
        counter += 1
        if not n_max is None:
            if counter > n_max:
                break

    print(f"Created {len(ds)} molecules for {pathname}.")
# %%


#%%
ds.filter_confs(max_energy=60)
#%%
ds.parametrize(allow_radicals=True, get_charges=model_from_dict(tag="bmk"))

#%%
graphs = ds.to_dgl()

#%%
import numpy as np
energies = np.array([])
ff_energies = np.array([])
for g in graphs:
    e = g.nodes['g'].data['u_qm'].flatten().detach().clone().numpy()
    e -= e.mean()
    r = g.nodes['g'].data['u_total_amber99sbildn'].flatten().detach().clone().numpy()
    r -= r.mean()
    energies = np.concatenate([energies, e], axis=0)
    ff_energies = np.concatenate([ff_energies, r], axis=0)
#%%
energies = energies.flatten()
ff_energies = ff_energies.flatten()
import matplotlib.pyplot as plt
plt.scatter(energies, ff_energies)
plt.plot(energies, energies, color="black", linestyle="--")
plt.xlabel("QM Energies")
plt.ylabel("FF Energies")
plt.ylim(-50, 50)
plt.xlim(-50, 50)
# %%
import numpy as np
n_max = 1e3
g = graphs[0]
f = g.nodes['n1'].data['grad_qm'].flatten().numpy()
af = g.nodes['n1'].data['grad_total_amber99sbildn'].flatten().numpy()
if n_max < len(f):
    perm = np.random.permutation(len(f))[:int(n_max)]
    f = f[perm]
    af = af[perm]


plt.scatter(f, af)
plt.plot(f,f, color="black", linestyle="--")
plt.xlabel("QM Grads")
plt.ylabel("FF Grads")
# %%
# test:
from openmm.app import ForceField
# %%
rad = PDBMolecule.from_pdb("./../radicals/F_rad.pdb")
topology = rad.to_openmm().topology
# %%
forcefield=ForceField('amber99sbildn.xml')
[templates, residues] = forcefield.generateTemplatesForUnmatchedResidues(topology)
for t_idx, template in enumerate(templates):
    ref_template = forcefield._templates[template.name]
    for a in template.atoms:
        print(a.bondedTo)
        break

# %%

# %%
p = "/hits/fast/mbm/share/datasets_Eric/AA_opt_rad/Cys_SG_opt.log"
p = "/hits/fast/mbm/share/datasets_Eric/AA_opt_rad/Arg_NH1_opt.log"

# p = "/hits/fast/mbm/share/datasets_Eric/AA_opt_rad/Lys_NZ_opt.log"
mol = PDBMolecule.from_gaussian_log_rad(p)
top = mol.to_openmm().topology
forcefield=ForceField('amber99sbildn.xml')

#%%
templates = forcefield.getMatchingTemplates(top)
# %%
for t in templates:
    print(t.name)
# %%
for atom in top.atoms():
    print(atom.name)
# %%
forcefield._templates
# %%
