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
                "AA_opt_nat",
                # "AA_scan_nat",
                # "AA_opt_rad",
                # "AA_scan_rad",
                ]:
    ds = PDBDataset([])
    counter = 0


    for p in (Path(dpath)/Path(pathname)).rglob("*.log"):
        
        try:
            mol = PDBMolecule.from_gaussian_log_rad(p, e_unit=kilocalorie_per_mole*23.0609, dist_unit=angstrom, force_unit=kilocalorie_per_mole*23.0609/angstrom)
        except:
            print(f"Failed to load {p}.")
            #raise
            continue
        if any([forbidden_res in mol.sequence for forbidden_res in ["HYP", "DOP", "NALA", 
        # "TRP_R", "HIE_R", "ILE_R", "LEU_R", "CYS_R", "TYR_R"
        ]]):
            continue
        # filter out conformations with energies above 200 kcal/mol away from the minimum
        mol.filter_confs(max_energy=200, max_force=500)

        try:
            mol.bond_check()
        except:
            print(f"Failed bond check for {p}.")
            #continue

        try:
            mol_copy = copy.deepcopy(mol)
            mol_copy.parametrize(allow_radicals=True, get_charges=model_from_dict(tag="bmk"))
        except:
            print(f"Failed to parametrize {p}.")
            continue

        if mol.energies is None:
            print(f"No energies present for {p}.")
            continue

        if mol.gradients is None:
            print(f"No energies present for {p}.")
            continue

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
