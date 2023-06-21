#%%
from openmm.app import ForceField, Simulation, PDBFile
from openmm import XmlSerializer

#%%
p = "/hits/fast/mbm/share/hartmaec/FF99SBILDNPX_OpenMM/amber99sbildn.xml"
ff = ForceField(p)

# %%
path = "/hits/fast/mbm/seutelf/software/AAmatcher/mains/tests/pdbmol/F.pdb"
pdb = PDBFile(path)

t = pdb.topology
# %%
sys = ff.createSystem(t)
# %%
import numpy as np
from PDBData.PDBMolecule import PDBMolecule
from PDBData.PDBDataset import PDBDataset
import copy
from pathlib import Path

# %%
ds = PDBDataset.load_npz("/hits/fast/mbm/seutelf/data/datasets/PDBDatasets/spice/base", n_max=300)
ff2 = ForceField("amber99sbildn.xml")

def get_energies(ds, ff="amber99sbildn.xml", component="u_total", lvl="g"):
    ds.filter_confs(50)
    if len(ff) > 0:
        ds.parametrize(ForceField(f"{ff}"), suffix = f"_{Path(ff).stem}")
    energies_ref = np.array([])
    for mol in ds.to_dgl():
        suffix = ""
        if component == "q":
            u = mol.nodes[lvl].data["q_ref"].numpy().flatten()
        else:
            if len(ff) > 0:
                suffix = f"_{Path(ff).stem}"
            u = mol.nodes[lvl].data[f"{component}{suffix}"].numpy().flatten()
        u -= u.min()
        energies_ref = np.concatenate((energies_ref, u))

    return energies_ref


def compare(component="u_total", lvl="g"):
    energies_ref = get_energies(ds, ff="amber99sbildn.xml", component=component, lvl=lvl)
    energies = get_energies(ds, ff=p, component=component, lvl=lvl)
    print()
    print("new ", energies.mean(), energies.std())
    #print("new ", energies)
    print("ref ", energies_ref.mean(), energies_ref.std())
    #print("ref ", energies_ref)
    #print(energies-energies_ref)
#%%
compare(lvl="n4_improper", component="k")
# %%
compare(lvl="g", component="u_improper")

# %%
e_ref = get_energies(ds, ff="", component="u_qm", lvl="g")
#%%
e_new = get_energies(ds, ff=p, component="u_total", lvl="g")

# %%
import matplotlib.pyplot as plt
plt.scatter(e_ref, e_new)
plt.show()
print("new")
print("mae= ", np.abs(e_new-e_ref).mean(), "rmse= ", (e_new-e_ref).std())
# %%
e_new = get_energies(ds, ff="amber99sbildn.xml", component="u_total", lvl="g")
#%%
import matplotlib.pyplot as plt
plt.scatter(e_ref, e_new)
plt.show()
print("ref")
print("mae= ", np.abs(e_new-e_ref).mean(), "rmse= ", (e_new-e_ref).std())
# %%
