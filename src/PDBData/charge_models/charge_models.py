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
from PDBData.charge_models.esp_charge import get_espaloma_charge_model
#%%


def model_from_dict(tag:str=None, d_path:Union[str, Path]=None):
    """
    Returns a function that takes a topology and returns a list of charges for each atom in the topology. Uses the dictionary stored in the json file a d_path.
    tags: ["bmk"]
    random: randomly assigns charges, keeping the total charge obtained with the tagged model invariant
    """

    basepath = Path(__file__).parent / Path("charge_dicts")
    if not tag is None:
        if tag == "bmk":
            d_path = basepath / Path("BMK/AAs_nat.json")
        elif tag == "bmk_rad":
            d_path = basepath / Path("ref/AAs_rad.json")
        elif tag == "ref":
            d_path = basepath / Path("ref/AAs_nat.json")
        elif tag == "ref_rad":
            d_path = basepath / Path("ref/AAs_avg.json")
        elif tag == "amber99sbildn":
            return None
        else:
            raise ValueError(f"tag {tag} not recognized")


    with open(str(d_path), "r") as f:
        d = json.load(f)
    
    # add ACE and NME from ref (this corresponds to amber99sbildn) if not present
    if (not "ACE" in d.keys()) or (not "NME" in d.keys()):
        with open(str(Path(__file__).parent / Path("charge_dicts/ref/AAs_nat.json")), "r") as f:
            ref = json.load(f)
        for key in ["ACE", "NME"]:
            if not key in d.keys():
                d[key] = ref[key]

    def get_charges(top:topology):
        return from_dict(d, top)
    
    return get_charges


def randomize_model(get_charges, noise_level:float=0.1):
    def get_random_charges(top):
        charges = get_charges(top)
        unit = charges[0].unit
        charges = np.array([q.value_in_unit(unit) for q in charges])
        charge = np.sum(charges)
        charges = np.array(charges)

        noise = np.random.normal(0, noise_level, size=charges.shape)
        # create noise with zero mean, i.e. the total charge is conserved
        noise -= np.mean(noise)
        charges += noise

        charges = [openmm.unit.Quantity(q, unit) for q in charges]
        return charges
    return get_random_charges


def from_dict(d:dict, top:topology, charge_unit=unit.elementary_charge):
    charges = []
    for atom in top.atoms():
        res = atom.residue.name
        name = atom.name

        # hard code that we only have HIE atm
        if res == "HIS":
            res = "HIE"

        # hard code the H naming of caps (maybe do this in the to_openmm function instead?)
        if res == "ACE":
            if "H" in name and not "C" in name:
                name = "HH31"
        elif res == "NME":
            if "H" in name and not "C" in name:
                if len(name) > 1:
                    name = "HH31"
            if "C" in name:
                name = "CH3"


        if res not in d.keys():
            raise ValueError(f"Residue {res} not in dictionary, residues are {d.keys()}")
        if name not in d[res]:
            raise ValueError(f"(Atom {name}, Residue {res}) not in dictionary, atom names are {d[res].keys()}")
        charge = float(d[res][name])
        charge = openmm.unit.Quantity(charge, charge_unit)
        charges.append(charge)
    return charges



#%%
if __name__=="__main__":
    from PDBData.PDBDataset import PDBDataset
    get_charges_bmk = model_from_dict()
    spicepath = Path(__file__).parent.parent.parent.parent / Path("mains/small_spice")
    dspath = Path(spicepath)/Path("small_spice.hdf5")
    ds = PDBDataset.from_spice(dspath, n_max=1)
    # %%
    ds.filter_validity()
    len(ds)

    ds.filter_confs()
    len(ds)
    #%%
    for mol in ds.mols:
        top = mol.to_openmm().topology
        charges = get_charges_bmk(top)
        print(charges)
    #%%
    ds.parametrize(get_charges=get_charges_bmk, suffix="_bmk")
    #%%
    i = 0
    g = ds[i].to_dgl()
    print(g.nodes["n1"].data["q_bmk"])
    # %%
    import copy
    ds2 = copy.deepcopy(ds)
    ds2.parametrize()
    #%%
    g = ds2[i].to_dgl()
    print(g.nodes["n1"].data["q_amber99sbildn"])
    #%%
    ds3 = copy.deepcopy(ds)
    ds3.parametrize(get_charges=model_from_dict())
    g = ds3[i].to_dgl()
    print(g.nodes["n1"].data["q_amber99sbildn"])


# %%

