#%%
import openmm
from openmm import unit
from openmm.app import Simulation
from openmm.app import ForceField, topology

from openmm.unit import Quantity
import json
import numpy as np
from typing import List, Tuple, Dict, Union, Callable
from pathlib import Path
#%%
def from_dict(d:dict, top:topology):
    charges = []
    for atom in top.atoms():
        res = atom.residue.name
        name = atom.name

        # unfortunately, ACE and NME are not in the dictionary yet
        if res in ["ACE", "NME"]:
            charges.append(None)
            continue

        # NOTE: how to do this properly, cannot be directly inferred from topology
        if res == "HIS":
            res = "HIE"

        if res not in d.keys():
            raise ValueError(f"Residue {res} not in dictionary, residues are {d.keys()}")
        if name not in d[res]:
            raise ValueError(f"(Atom {name}, Residue {res}) not in dictionary, atom names are {d[res].keys()}")
        charge = d[res][name]
        charge = openmm.unit.Quantity(charge, unit.elementary_charge)
        charges.append(charge)
    return charges


#%%
if __name__=="__main__":
    from PDBData.PDBDataset import PDBDataset
    # get_charges = model_from_dict()
    # manual_charges = get_charges(topology)
    # manual_charges = torch.tensor([c.value_in_unit(units.CHARGE_UNIT) for c in manual_charges], dtype=torch.float32)

    # def f(Callable[openmm.app.Topology]):
    #     return 0
    #%%
    spicepath = Path(__file__).parent.parent.parent.parent / Path("mains/small_spice")
    dspath = Path(spicepath)/Path("small_spice.hdf5")
    ds = PDBDataset.from_spice(dspath, n_max=5)
    # %%
    ds.filter_validity()
    len(ds)

    ds.filter_confs()
    len(ds)
    ds.parametrize()
    #%%
    g = ds[2].to_dgl()
    g.nodes["n1"].data["q_amber99sbildn"]
    # %%
    m = ds[2]
    top = m.to_openmm().topology
    #%%
    with open(Path(__file__).parent/Path("charge_dicts/BMK/AAs_nat.json"), "r") as f:
        d = json.load(f)
    charges = from_dict(d, top)
    #%%
    print([(g.nodes["n1"].data["q_amber99sbildn"][i], charges[i]) for i in range(len(charges))])
    print(sum([c.value_in_unit(unit.elementary_charge) for c in charges if not c is None]))
    print(g.nodes["n1"].data["q_amber99sbildn"].sum())
    # %%
    print(*m.pdb)
    print(*[(a.name, a.residue.name) for a in top.atoms()])
    #%%
