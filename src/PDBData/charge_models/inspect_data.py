#%%
import numpy as np
from pathlib import Path
#%%
d = np.load(Path(__file__).parent/Path("npy_charge_dicts/BMK/AAs_nat.npy"), allow_pickle=True)
# %%
d = d.item()
d.keys()
# %%
sum([c for c in d['Asp'].values()])
# %%
