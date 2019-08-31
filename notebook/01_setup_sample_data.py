# %%
DEBUG = True
import numpy as np
import pandas as pd
from pathlib import Path
import os
%load_ext autoreload
%autoreload 2

HOME = Path('../')
DATA = HOMEDIR/'input'
if DEBUG:
    DATA_DIR = DATA/'sample'
    DATA_DIR.mkdir(exist_ok=True)
else:
     DATA_DIR = DATA
     DATA_DIR.mkdir(exist_ok=True)
     
df = pd.read_csv(DATA/'train.csv')

# %%


if DEBUG:
    # take 2% sample data for setup
    from shutil import copyfile
    n_frac = 0.02
    n = int(len(df) * n_frac)
    sample_idx = np.random.permutation(len(df))[:n]
    (DATA_DIR/'train').mkdir(exist_ok=True)
    (DATA_DIR/'test').mkdir(exist_ok=True)
    train_fn_list = list((DATA/'train').iterdir())[:n]
    test_fn_list = list((DATA/'test').iterdir())[:n]
    for fn in train_fn_list:
        f = fn.parts[-1]
        copyfile(DATA/'train'/f, DATA_DIR/'train'/f)
        # print(fn)
    print(f'{len(train_fn_list)} files are copied')
    for fn in test_fn_list:
        f = fn.parts[-1]
        copyfile(DATA/'test'/f, DATA_DIR/'test'/f)
    print(f'{len(test_fn_list)} files are copied')
# %%
df.head()


#%%
list((DATA_DIR/'train').iterdir())
#%%
list((DATA/'train').iterdir())

#%%
list((DATA/'test').glob('*'))


