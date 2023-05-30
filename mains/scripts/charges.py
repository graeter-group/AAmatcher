# run to generate parametrized spice data

from PDBData.PDBDataset import PDBDataset
from preprocess_spice import dipeppath, spicepath
from pathlib import Path
from openmm.app import ForceField
from PDBData.charge_models.charge_models import model_from_dict, randomize_model, get_espaloma_charge_model
import os
import warnings
warnings.filterwarnings("ignore")


def make_ds(get_charges, storepath, dspath, n_max=None, openff_charge_flag=False, overwrite=False):
    ds = None

    # try to load dataset directly if present
    if n_max is None and not overwrite:
        try:
            ds = PDBDataset.load_npz(dspath)
            ds.remove_names_spice()
            # set to none if nmax was small for testing before (a bit hacky...)
            if len(ds) < 10:
                ds = None
        except:
            ds = None
            pass

    if ds is None:
        ds = PDBDataset.from_spice(dipeppath, info=True, n_max=n_max)
    # %%
    # remove conformations with energy > 200 kcal/mol from min energy in ds[i]
    ds.filter_confs(max_energy=200) # remove crashed conformations
    
    ds.parametrize(forcefield=ForceField("amber99sbildn.xml"), get_charges=get_charges, openff_charge_flag=openff_charge_flag)
    
    ds.save_npz(storepath, overwrite=True)
    ds.save_dgl(str(storepath)+"_dgl.bin", overwrite=True)
    #%%
    
    # remove conformations with energy > 60 kcal/mol from min energy in ds[i]
    ds.filter_confs(max_energy=60)
    # unfortunately, have to parametrize again to get the right shapes
    ds.parametrize(forcefield=ForceField("amber99sbildn.xml"), get_charges=get_charges, openff_charge_flag=openff_charge_flag)
    ds.save_npz(str(storepath)+"_60", overwrite=True)
    ds.save_dgl(str(storepath)+"_60_dgl.bin", overwrite=True)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", "-t", type=str, default=["bmk"], nargs="+", help="tag of the charge model to use, see model_from_dict in charge_models.py. If the tag is 'esp_charge', use espaloma_charge to parametrize")
    parser.add_argument("--overwrite", "-o", action="store_true", default=False, help="overwrite existing .npz dataset")
    parser.add_argument("--noise_level", type=float, default=[None], nargs="+")
    args = parser.parse_args()
    for tag in args.tag:
        print()
        print(f"tag: {tag}")
        for noise_level in args.noise_level:
            NMAX = None
            # NMAX = 2
            dspath = Path(spicepath).parent/Path("PDBDatasets/spice")
            storepath = Path(spicepath).parent/Path(f"PDBDatasets/spice_{tag}")
            if not noise_level is None:
                storepath = Path(spicepath).parent/Path(f"PDBDatasets/noisy_charges/spice_{tag}_{noise_level}")

            openff_charge_flag = False
            if tag == 'esp_charge':
                get_charges = get_espaloma_charge_model()
                openff_charge_flag = True
            else:
                get_charges = model_from_dict(tag=tag)

            if not noise_level is None:
                get_charges = randomize_model(model_from_dict(tag=tag), noise_level=noise_level)
                print()
                print(f"noise level: {noise_level}")

            make_ds(get_charges=get_charges, storepath=storepath, dspath=dspath, n_max=NMAX, openff_charge_flag=openff_charge_flag)