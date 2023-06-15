from PDBData.PDBDataset import PDBDataset
from pathlib import Path
spicepath = "/hits/fast/mbm/seutelf/data/datasets/SPICE-1.1.2.hdf5"
dipeppath = str(Path(spicepath).parent/Path("dipeptides_spice.hdf5"))

storepath = Path(spicepath).parent/Path("PDBDatasets/spice/base")

if __name__=="__main__":
    ds = PDBDataset.from_spice(dipeppath, info=True, n_max=None)

    # remove conformations with energy > 200 kcal/mol from min energy in ds[i]
    ds.filter_confs(max_energy=200) # remove crashed conformations

    ds.save_npz(storepath, overwrite=True)
    ds.save_dgl(str(storepath)+"_dgl.bin", overwrite=True)