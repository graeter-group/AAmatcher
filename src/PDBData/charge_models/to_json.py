#%%
if __name__=="__main__":
    import numpy as np
    from pathlib import Path
    import json
    import os

    #%%
    old_path = Path(__file__).parent/Path("npy_charge_dicts")
    new_path = Path(__file__).parent/Path("charge_dicts")
    for p in old_path.rglob("*.npy"):
        d = np.load(p, allow_pickle=True)
        d = d.item()

        # rename all keys (ie residues) to uppercase
        old_keys = list(d.keys())
        for k in old_keys:
            d[k.upper()] = d.pop(k)

        # convert all values to floats
        for k in d.keys():
            for k2 in d[k].keys():
                try:
                    d[k][k2] = float(d[k][k2])
                except:
                    for k3 in d[k][k2].keys():
                        try:
                            d[k][k2][k3] = float(d[k][k2][k3])
                        except:
                            pass
        os.makedirs(str(new_path/Path(p.parent.stem)), exist_ok=True)

        with open(str(new_path/Path(p.parent.stem)/Path(p.stem))+".json", "w") as f:
            json.dump(d, f, indent=4, sort_keys=True)
#%%
