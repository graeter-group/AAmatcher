#%%
store = True

from PDBData.PDBDataset import PDBDataset
from PDBData.PDBMolecule import PDBMolecule
from openmm.unit import kilocalorie_per_mole, angstrom
from pathlib import Path
from PDBData.charge_models.charge_models import model_from_dict
import copy
import os
import shutil
import json

from pathlib import Path
dpath = "/hits/fast/mbm/share/datasets_Eric"
storepath = "/hits/fast/mbm/seutelf/data/datasets/PDBDatasets"
#%%

overwrite = True
n_max = 20

MAX_ENERGY = 200
MAX_FORCE = 400


for pathname in [
                "AA_opt_nat",
                "AA_scan_nat",
                "AA_opt_rad",
                "AA_scan_rad",
                ]:
    print(f"Starting {pathname}...")
    counter = 0
    expected_fails = {}
    unexpected_fails = {}
    bond_check_fails = {}

    ds = PDBDataset([])

    if overwrite:
        if os.path.exists(str(Path(storepath)/Path(pathname)/Path("base"))):
            shutil.rmtree(str(Path(storepath)/Path(pathname))/Path("base"))

    for p in (Path(dpath)/Path(pathname)).rglob("*.log"):
        ###########################################################
        # INIT
        try:
            mol = PDBMolecule.from_gaussian_log_rad(p, e_unit=kilocalorie_per_mole*23.0609, dist_unit=angstrom, force_unit=kilocalorie_per_mole*23.0609/angstrom)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if not any([skip_res in str(p) for skip_res in ["Glu", "Asp", "Ile", "Trp", "NmeCH3", "Nme", "Hie", "AceCH3", "Hyp", "Dop", "Nala"]]):
                # unexpected fail!
                unexpected_fails[str(p)] = str(e)
            else:
                # expected fail
                expected_fails[str(p)] = str(e)
            continue

        # filter out extreme conformations with energies above 200 kcal/mol away from the minimum
        mol.filter_confs(max_energy=MAX_ENERGY, max_force=MAX_FORCE)

        ###########################################################
        # BOND CHECK
        try:
            mol.bond_check()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            bond_check_fails[str(p)] = str(e)
            continue
        ###########################################################
        # PARAMETRIZE CHECK
        try:
            mol_copy = copy.deepcopy(mol)
            mol_copy.parametrize(allow_radicals=True, get_charges=model_from_dict(tag="bmk"))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Failed to parametrize {p}:\n  e: {e}")
            print()
            continue

        if mol.energies is None:
            print(f"No energies present for {p}.")
            print()
            continue

        if mol.gradients is None:
            print(f"No energies present for {p}.")
            print()
            continue

        ds.append(mol)
        counter += 1
        if not n_max is None:
            if counter > n_max:
                break

    print(f"Finished {pathname} with {counter} molecules.")
    print(f"  {len(expected_fails)} expected fails.")
    print(f"  {len(unexpected_fails)} unexpected fails.")
    print(f"  {len(bond_check_fails)} bond check fails.")
    print()
    print(f"expected fails:\n{json.dumps(expected_fails, indent=4)}\n")
    print(f"bond check fails:\n{json.dumps(bond_check_fails, indent=4)}\n")
    print(f"unexpected fails:\n{json.dumps(unexpected_fails, indent=4)}\n\n")


    if store:
        ds.save_npz(Path(storepath)/Path(pathname)/Path("base"), overwrite=True)
        print(f"Stored {len(ds)} molecules for {pathname}.")
# %%


#%%
do_scatter = False
if not do_scatter:
    exit()
#%%
# p = Path(storepath)/Path("spice/base")
p = Path(storepath)/Path("AA_scan_nat/base")
p = Path(storepath)/Path("AA_opt_rad/base")
p = Path(storepath)/Path("AA_scan_rad/base")
p = Path(storepath)/Path("AA_opt_nat/base")

ds = PDBDataset.load_npz(p)
# ds = PDBDataset.load_npz(p, n_max=20)
ds.filter_confs(max_energy=60)
ds.parametrize(allow_radicals=True, get_charges=model_from_dict(tag="bmk"))
#%%
graphs = ds.to_dgl()

#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
ds.filter_confs(max_energy=60)
ds.energy_hist(filename=None, show=True)
ds.grad_hist(filename=None, show=True)
#%%
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
plt.ylabel("FF Grads")
plt.xlabel("QM Grads")
plt.plot(f,f, color="black", linestyle="--")
# %%
