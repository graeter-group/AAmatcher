# run to generate parametrized spice data

from PDBData.PDBDataset import PDBDataset
from pathlib import Path
from openmm.app import ForceField
from PDBData.charge_models.charge_models import model_from_dict, randomize_model, get_espaloma_charge_model
import os
import warnings
warnings.filterwarnings("ignore") # for esp_charge

FILTER_REF = True # whether to apply the filtering to the reference energies or the qm energies

def make_ds(get_charges, storepath, dspath, openff_charge_flag=False, overwrite=False, allow_radicals=False, n_max=None, collagen=False):

    print("starting...")
    print("will write to ", storepath)

    ds = PDBDataset()

    ds = PDBDataset.load_npz(dspath, n_max=n_max)

    ff = ForceField("amber99sbildn.xml")

    if not collagen:
        ds.remove_names(patterns=["HYP", "DOP"], upper=True)

    ds.parametrize(forcefield=ff, get_charges=get_charges, openff_charge_flag=openff_charge_flag, allow_radicals=allow_radicals, collagen=collagen)

    # filter out conformations that are way out of equilibrium:
    ds.filter_confs(max_energy=200, max_force=500, reference=FILTER_REF)

    
    ds.save_npz(storepath, overwrite=overwrite)
    ds.save_dgl(str(storepath)+"_dgl.bin", overwrite=overwrite)
    #%%

    ds.parametrize(forcefield=ff, get_charges=get_charges, openff_charge_flag=openff_charge_flag, allow_radicals=allow_radicals, collagen=collagen)
    
    # remove conformations with energy > 60 kcal/mol from min energy in ds[i]
    ds.filter_confs(max_energy=60, max_force=200, reference=FILTER_REF)

    # unfortunately, have to parametrize again to get the right shapes
    ds.save_npz(str(storepath)+"_60", overwrite=overwrite)
    ds.save_dgl(str(storepath)+"_60_dgl.bin", overwrite=overwrite)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", type=str, default=["spice/base"], nargs='+', help="name of the dataset to be loaded. will load from the directory base_path/ds_name default: spice/base")
    parser.add_argument("--ds_base", type=str, default=str(Path("/hits/fast/mbm/seutelf/data/datasets/PDBDatasets")), help="base path to the dataset, default: /hits/fast/mbm/seutelf/data/datasets/PDBDatasets")
    parser.add_argument("--charge", "-c", type=str, dest="tag", default=["bmk"], nargs="+", help="tag of the charge model to use, see model_from_dict in charge_models.py. If the tag is 'esp_charge', use espaloma_charge to parametrize.\npossible tags: ['bmk', 'avg', 'heavy, 'amber99sbildn', 'esp_charge'] (esp_charge is only for non-radicals!)")
    parser.add_argument("--overwrite", "-o", action="store_true", default=False, help="overwrite existing .npz dataset, default:False")
    parser.add_argument("--noise_level", type=float, default=[None], nargs="+", help="noise level to add to the charges, default:None")
    parser.add_argument("--storage_dir", "-s", type=str, default=None, help="directory path relative to ds_base. in this folder, the parametrizes datasets are stored, named by the tag and noise_level. if None, this is the parent of ds_name. default: None")
    parser.add_argument("--allow_radicals", "-r", action="store_true", default=False)

    parser.add_argument("--n_max", type=int, default=None, help="maximum number of conformations to load from the dataset, default: None")

    parser.add_argument("--collagen", "-col", action="store_true", default=False, help="use collagen the forcefield instead of amber99sbildn. Will add '_col' to the name. Also remove any molecules with DOP or HYP in the name. default: False")

    args = parser.parse_args()
    for ds_name in args.ds_name:
        for tag in args.tag:
            print()
            print(f"charge tag: {tag}")
            for noise_level in args.noise_level:
                dspath = Path(args.ds_base)/Path(ds_name)
                storebase = str(dspath.parent) if args.storage_dir is None else str(Path(args.ds_base)/Path(args.storage_dir))
                
                storepath = os.path.join(str(storebase),tag) # tagged with charge model
                if args.collagen:
                    storepath += "_col"
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

                make_ds(get_charges=get_charges, storepath=storepath, dspath=dspath, openff_charge_flag=openff_charge_flag, overwrite=args.overwrite, allow_radicals=args.allow_radicals, n_max=args.n_max, collagen=args.collagen)