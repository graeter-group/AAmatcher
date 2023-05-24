#%%

# run this to "train" the residue classifier by generating a lookup
if __name__=="__main__":

    import dgl
    import torch

    from PDBData.xyz2res.get_residues import write_residues

    from PDBData.xyz2res.constants import RESIDUES
    from pathlib import Path

    dspath = str(Path(__file__).parent/Path('data')/Path("pep1_dgl"+'.bin'))
    graphs, _ = dgl.load_graphs(dspath)

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
    # check that the mapping is bijective except for ILE and LEU
    set(RESIDUES) - set(get_residue.lookup.values())
    # %%

    # histidin is a special case, with its tautomers, we set the default to HIE
    # non-protonated his hash:
    h = [ha for ha, res in get_residue.lookup.items() if res == "HIS"][0]

    get_residue.lookup[h] = "HIE"


    #%%
    import json
    hash_storage = str(Path(__file__).parent/Path("hashed_residues.json"))
    with open(hash_storage, "w") as f:
        json.dump(get_residue.lookup, f)