#%%
spicepath = "/hits/fast/mbm/seutelf/data/datasets/SPICE-1.1.2.hdf5"

from pathlib import Path
from openmm.unit import bohr, Quantity, hartree, mole
import numpy as np
from PDBData.PDBMolecule import PDBMolecule

from PDBData.units import DISTANCE_UNIT, ENERGY_UNIT, FORCE_UNIT
import torch

from PDBData.PDBDataset import PDBDataset
import dgl
import json

#%%
ds_old = dgl.load_graphs("/hits/fast/mbm/seutelf/data/datasets/old_PDBDatasets/spice_amber99sbildn_dgl.bin")[0]
with open("/hits/fast/mbm/seutelf/data/datasets/old_PDBDatasets/spice_amber99sbildn_dgl_seq.json", "r") as f:
    seqs = json.load(f)
#%%
ds_new = PDBDataset.load_npz("/hits/fast/mbm/seutelf/data/datasets/PDBDatasets/spice")
ds_new.mols = ds_new.mols
ds_new.parametrize()
#%%
mol = None
for i in range(10):
    s = seqs[i]
    g_ref = ds_old[i]
    for m in ds_new:
        if m.sequence == s:
            mol = m
            break
    g = mol.to_dgl()

    for lvl in g.ntypes:
        for ft in g.nodes[lvl].data.keys():
            if g.nodes[lvl].data[ft].shape != g_ref.nodes[lvl].data[ft].shape:
                print(f"shape of {lvl} {ft} is {g.nodes[lvl].data[ft].shape} in new dataset and {g_ref.nodes[lvl].data[ft].shape} in old dataset")
    # %%