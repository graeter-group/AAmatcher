# FORCES ARE BAD, ENERGIES ARE FINE
#%%
from PDBData.PDBDataset import PDBDataset
from PDBData.PDBMolecule import PDBMolecule
from openmm.unit import kilocalorie_per_mole, angstrom


from pathlib import Path
dpath = "/hits/fast/mbm/share/datasets_Eric"
storepath = "/hits/fast/mbm/seutelf/data/datasets/PDBDatasets/AA_opt_nat/base"
#%%
p = Path(dpath)/Path("AA_scan_nat/Ala_nat/Ala_nat_Phi_10.log")
#p = Path(dpath)/Path("AA_opt_nat/Phe_nat_opt.log")
mol = PDBMolecule.from_gaussian_log(p, force_unit=kilocalorie_per_mole*23.0609/angstrom)
# %%
print(*mol.pdb)
print(mol.energies.shape)

# %%
mol.filter_confs(max_energy=60)
mol.parametrize()
g = mol.to_dgl()
e = g.nodes['g'].data['u_qm'].flatten().numpy()
a = g.nodes['g'].data['u_total_amber99sbildn'].flatten().numpy()

e-=e.min()
a-=a.min()
import matplotlib.pyplot as plt
plt.scatter(e, a)
plt.ylim(0,40)
plt.ylabel("amber")
plt.xlabel("qm")
plt.plot(e,e, color="black")
# %%
import numpy as np
n_max = 1e5
f = g.nodes['n1'].data['grad_qm'].flatten().numpy()
af = g.nodes['n1'].data['grad_total_amber99sbildn'].flatten().numpy()
if n_max < len(f):
    perm = np.random.permutation(len(f))[:int(n_max)]
    f = f[perm]
    af = af[perm]
plt.scatter(f, af)
# plt.ylim(-40,40)
plt.ylabel("amber")
plt.xlabel("qm")
plt.plot(f,f, color="black")
# %%
g.nodes['n1'].data['grad_qm'].shape
# %%
