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
from PDBData.utils.find_radical import get_radicals
from PDBData.utils.utils import replace_h23_to_h12
#%%


def model_from_dict(tag:str=None, d_path:Union[str, Path]=None):
    """
    Returns a function that takes a topology and returns a list of charges for each atom in the topology. Uses the dictionary stored in the json file a d_path.
    possible tags: ['bmk', 'avg', 'heavy, 'amber99sbildn']
    random: randomly assigns charges, keeping the total charge obtained with the tagged model invariant
    """
    if d_path is None and tag is None:
        raise ValueError("Either d_path or tag must be given")
    
    if d_path is None:
        basepath = Path(__file__).parent / Path("charge_dicts")
        if not tag is None:
            if tag == "bmk":
                d_path = basepath / Path("BMK/AAs_nat.json")
                drad_path = basepath / Path("BMK/AAs_rad.json")
            elif tag == "avg":
                d_path = basepath / Path("ref/AAs_nat.json")
                drad_path = basepath / Path("ref/AAs_rad_avg.json")
            elif tag == "heavy":
                d_path = basepath / Path("ref/AAs_nat.json")
                drad_path = basepath / Path("ref/AAs_rad_heavy.json")
            elif tag == "amber99sbildn":
                return None
            else:
                raise ValueError(f"tag {tag} not recognized")


    with open(str(d_path), "r") as f:
        d = json.load(f)

    with open(str(drad_path), "r") as f:
        d_rad = json.load(f)
    
    # add ACE and NME from ref (this corresponds to amber99sbildn) if not present
    if (not "ACE" in d.keys()) or (not "NME" in d.keys()):
        with open(str(Path(__file__).parent / Path("charge_dicts/ref/AAs_nat.json")), "r") as f:
            ref = json.load(f)
        for key in ["ACE", "NME"]:
            if not key in d.keys():
                d[key] = ref[key]

    def get_charges(top:topology):
        return from_dict(d=d, top=top, d_rad=d_rad)
    
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


def from_dict(d:dict, top:topology, d_rad:dict=None, charge_unit=unit.elementary_charge):
    charges = []
    rad_indices, rad_names = get_radicals(topology=top)

    assert len(set(rad_indices)) == len(rad_indices), f"radical indices are not unique: {rad_indices}"

    if len(rad_indices) != 0 and d_rad is None:
        raise ValueError("radical indices are given but no radical dictionary is given")


    for atom in top.atoms():
        rad_name = None
        res_idx = atom.residue.index

        num_atoms = len([a for a in atom.residue.atoms()])

        if res_idx in rad_indices:
            rad_name = rad_names[rad_indices.index(res_idx)]


        # IF ID IN RADICAL INDICES, DICT IS GIVEN BY THE RAD NAME
        res = atom.residue.name
        name = atom.name

        # hard code that we only have HIE atm
        if res == "HIS":
            res = "HIE"

        H23_dict = replace_h23_to_h12.d

        if not rad_name is None:
            # rename all HB2 to HB1 because these are in the dict for radicals and have the same parameters since they are indisinguishable
            if not name in d_rad[res][rad_name].keys():
                if res in H23_dict.keys():
                    orig_name = name
                    if name in [f"H{lvl}2" for lvl in H23_dict[res]]:
                        name = name.replace("2", "1")
                    elif name in [f"H{lvl}1" for lvl in H23_dict[res]]:
                        name = name.replace("1", "2")
            if not name in d_rad[res][rad_name].keys():
                raise ValueError(f"Neither {name} nor {orig_name} in radical dictionary for {res} {rad_name}.\nStored atom names are {d_rad[res][rad_name].keys()}")



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

        if not res in (d.keys() if rad_name is None else d_rad.keys()):
            raise ValueError(f"Residue {res} not in dictionary, residues are {d.keys() if rad_name is None else d_rad.keys()}")
            
        if not rad_name is None and not rad_name in d_rad[res].keys():
            raise ValueError(f"Radical {rad_name} not in dictionary for residue {res}, radicals are {d_rad[res].keys()}")

        if name not in (d[res] if rad_name is None else d_rad[res][rad_name]):
            raise ValueError(f"(Atom {name}, Residue {res}, Radical {rad_name}) not in dictionary, atom names are {d[res].keys() if rad_name is None else d_rad[res][rad_name].keys()}")

        charge = float(d[res][name]) if rad_name is None else float(d_rad[res][rad_name][name])

        if len((d[res] if rad_name is None else d_rad[res][rad_name]).keys()) != num_atoms:
            raise RuntimeError(f"Residue {res} has {num_atoms} atoms, but dictionary has {len((d[res] if rad_name is None else d_rad[res][rad_name]).keys())} atoms.\ndictionary entries: {(d[res] if rad_name is None else d_rad[res][rad_name]).keys()},\natom names: {[a.name for a in atom.residue.atoms()]},\nrad name is {rad_name},\nres_idx is {res_idx}, \nrad_indices are {rad_indices},\nrad_names are {rad_names}")

        charge = openmm.unit.Quantity(charge, charge_unit)
        charges.append(charge)

    return charges


#%%
if __name__=="__main__":
    from PDBData.PDBDataset import PDBDataset
    get_charges_bmk = model_from_dict("bmk")
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
    ds3.parametrize(get_charges=model_from_dict("amber99sbildn"))
    g = ds3[i].to_dgl()
    print(g.nodes["n1"].data["q_amber99sbildn"])
    #%%
    from PDBData.PDBMolecule import PDBMolecule
    rad = PDBMolecule.from_pdb("./../../../mains/tests/radicals/F_radA.pdb")
    mol = PDBMolecule.from_pdb("./../../../mains/tests/radicals/F.pdb")
    fail_rad = PDBMolecule.from_pdb("./../../../mains/tests/radicals/F_rad.pdb")
    # print(*mol.pdb)
    # print(*rad.pdb)
    charges = get_charges_bmk(rad.to_openmm().topology)
    # print(charges)
    ds = PDBDataset()
    ds.mols = [rad, mol]
    #%%
    ds.parametrize(get_charges=get_charges_bmk, allow_radicals=True, suffix="_bmk")
    #%%
    g_rad = ds[0].to_dgl()
    g_mol = ds[1].to_dgl()
    print(g_rad.nodes["n1"].data["q_bmk"])
    print(g_mol.nodes["n1"].data["q_bmk"])
    #%%
    # print both charges next to each other:
    mol_atoms = [a for a in mol.to_openmm().topology.atoms()]
    rad_atoms = [a for a in rad.to_openmm().topology.atoms()]

    print("radical charge  molecule charge  radical atom  molecule atom")

    i = 0
    j = 0
    while i < len(mol_atoms):
            
        if mol_atoms[i].name != rad_atoms[j].name:
            print(f"   ---   {g_mol.nodes['n1'].data['q_bmk'][i].item():>3.3f}   - {mol_atoms[i].name}".replace(" -", "-"))
            i += 1
        else:
            print(f" {g_rad.nodes['n1'].data['q_bmk'][j].item():>3.3f}  {g_mol.nodes['n1'].data['q_bmk'][i].item():>3.3f}  {rad_atoms[j].name} {mol_atoms[i].name}".replace(" -", "-"))
            i += 1
            j += 1


    # %%

# %%
