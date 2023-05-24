#%%
if __name__ == "__main__":
    from PDBData.PDBDataset import PDBDataset
    from preprocess_spice import dipeppath, spicepath
    from pathlib import Path
    from openmm.app import ForceField

    import os
    dspath = Path(spicepath).parent/Path("PDBDatasets/spice")
    param_dspath = Path(spicepath).parent/Path("PDBDatasets/spice_amber99sbildn")
    os.makedirs(dspath, exist_ok=True)
    ds = None
    # %%
    # this fails for a lot of molecules, because PDBMolecule cannot deal with charges or non-standard conformations
    ds = PDBDataset.from_spice(dipeppath, info=True)
    # %%
    ds.filter_validity(forcefield=ForceField("amber99sbildn.xml"))
    # %%
    ds.save_npz(dspath)
    #%%
    if ds is None:
        ds = PDBDataset.load_npz(dspath)
    ds.parametrize(forcefield=ForceField("amber99sbildn.xml"))
    #%%
    ds.save_npz(param_dspath)
    #%%
    ds.save_dgl(str(param_dspath)+"_dgl.bin")
    #%%
    # remove conformations with energy > 60 kcal/mol from the amber99sbildn minimum
    if ds is None:
        ds = PDBDataset.load_npz(dspath)
    ds.filter_confs(max_energy=60)
    # unfortunately, have to parametrize again to get the right shapes
    ds.parametrize(forcefield=ForceField("amber99sbildn.xml"))
    ds.save_dgl(str(param_dspath)+"_60_dgl.bin")
    #%%
    ds.save_npz(str(param_dspath)+"_60")
# %%
