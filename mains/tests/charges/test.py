from PDBData.PDBDataset import PDBDataset
from PDBData.charge_models.charge_models import model_from_dict
from pathlib import Path


spicepath = "/hits/fast/mbm/seutelf/data/datasets/SPICE-1.1.2.hdf5"
dipeppath = str(Path(spicepath).parent/Path("dipeptides_spice.hdf5"))
smallspicepath = str(Path(spicepath).parent/Path("small_spice.hdf5"))

def do_test(get_charges):
    ds = PDBDataset.from_spice(dipeppath, n_max=5, randomize=True)

    ds.filter_validity()

    ds.filter_confs()

    ds.parametrize(get_charges=get_charges, suffix="_chargemodel")

    import copy
    ds2 = copy.deepcopy(ds)
    ds2.parametrize()

    graphs1 = ds.to_dgl()
    graphs2 = ds2.to_dgl()

    import torch

    # shouldnt change bonded energy
    for i in range(len(graphs1)):
        assert torch.allclose(graphs1[i].nodes["g"].data["u_bonded_chargemodel"], graphs2[i].nodes["g"].data["u_bonded_amber99sbildn"], atol=1e-5)

        # sum of charges should be the same
        assert torch.allclose(graphs1[i].nodes["n1"].data["q_chargemodel"].sum(), graphs2[i].nodes["n1"].data["q_amber99sbildn"].sum(), atol=1e-3), f"charge sums {i}: {graphs1[i].nodes['n1'].data['q_chargemodel'].sum()}, {graphs2[i].nodes['n1'].data['q_amber99sbildn'].sum()}"
        # if not torch.allclose(graphs1[i].nodes["n1"].data["q_chargemodel"].sum(), graphs2[i].nodes["n1"].data["q_amber99sbildn"].sum(), atol=1e-3):
        #     print(f"charge sums {i}: {graphs1[i].nodes['n1'].data['q_chargemodel'].sum()}, {graphs2[i].nodes['n1'].data['q_amber99sbildn'].sum()}")

    print("Test over!")


get_charges_ref = model_from_dict(tag="ref")
do_test(get_charges_ref)

get_charges_bmk = model_from_dict(tag="bmk")
do_test(get_charges_bmk)