#%%

# run this to "train" the residue classifier by generating a lookup

import dgl
import torch

from AAmatcher.xyz2res.get_residues import write_residues

from AAmatcher.xyz2res.constants import RESIDUES

graphs, _ = dgl.load_graphs("./../data/pep2.bin")

#%%
def get_residue(subgraph):
    elems = torch.argmax(subgraph.ndata["atomic_number"], dim=-1)
    # just take the square because it works as opposed to the sum
    res_hash = (elems*elems).sum().item()
    res = RESIDUES[subgraph.ndata["residue"][0].item()]
    get_residue.lookup[res_hash] = res
    return get_residue.lookup[res_hash]
get_residue.lookup = {}
#%%
for g in graphs:
    g = write_residues(g, get_residue)
    
#%%
import json
hash_storage = "hashed_residues.json"
with open(hash_storage, "w") as f:
    json.dump(get_residue.lookup, f)

#%%
# check that the mapping is bijective except for ILE and LEU
set(RESIDUES) - set(get_residue.lookup.values())
# %%
