# run to generate parametrized spice data

from PDBData.PDBDataset import PDBDataset
from pathlib import Path
from openmm.app import ForceField
from PDBData.charge_models.charge_models import model_from_dict, randomize_model, get_espaloma_charge_model
import os
import warnings
warnings.filterwarnings("ignore") # for esp_charge


def make_ds(get_charges, storepath, dspath, openff_charge_flag=False, overwrite=False):
    ds = None

    ds = PDBDataset.load_npz(dspath)

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
    parser.add_argument("--tag", "-t", type=str, default=["bmk"], nargs="+", help="tag of the charge model to use, see model_from_dict in charge_models.py. If the tag is 'esp_charge', use espaloma_charge to parametrize.\npossible tags: ['bmk', 'bmk_rad', 'ref', 'ref_rad_avg', 'ref_rad_heavy, 'amber99sbildn']")
    parser.add_argument("--overwrite", "-o", action="store_true", default=False, help="overwrite existing .npz dataset")
    parser.add_argument("--noise_level", type=float, default=[None], nargs="+")
    parser.add_argument("--ds_base", type=str, default=str(Path("/hits/fast/mbm/seutelf/data/datasets/PDBDatasets")))
    parser.add_argument("--ds_name", type=str, default="spice")

    args = parser.parse_args()
    for tag in args.tag:
        print()
        print(f"tag: {tag}")
        for noise_level in args.noise_level:
            dspath = Path(args.ds_base)/Path(args.ds_name)
            storepath = str(dspath)+"_"+tag # with charge model
            if not noise_level is None:
                storepath += f"_{noise_level}"

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

            make_ds(get_charges=get_charges, storepath=storepath, dspath=dspath, openff_charge_flag=openff_charge_flag)