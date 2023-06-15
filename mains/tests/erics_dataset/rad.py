
#%%
from PDBData.PDBDataset import PDBDataset
from PDBData.PDBMolecule import PDBMolecule
from openmm.unit import kilocalorie_per_mole, angstrom

from PDBData.charge_models.charge_models import model_from_dict


from pathlib import Path
dpath = "/hits/fast/mbm/share/datasets_Eric"
storepath = "/hits/fast/mbm/seutelf/data/datasets/PDBDatasets"
#%%
p = Path(dpath)/Path("AA_scan_rad/Ala_CA/Ala_CA_Phi_00.log")
mol = PDBMolecule.from_gaussian_log_rad(p, force_unit=kilocalorie_per_mole*23.0609/angstrom)
# %%
print(*mol.pdb)
print(mol.energies.shape)
# %%
mol.filter_confs(max_energy=30)
mol.parametrize(allow_radicals=True, get_charges=model_from_dict(tag="bmk"))
#%%
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
f = g.nodes['n1'].data['grad_qm'].flatten().numpy()
af = g.nodes['n1'].data['grad_total_amber99sbildn'].flatten().numpy()
plt.scatter(f, af)
plt.ylabel("amber")
plt.xlabel("qm")
plt.plot(f,f, color="black")
# %%
g.nodes['n1'].data['grad_qm'].shape
# %%
