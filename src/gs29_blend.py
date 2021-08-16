from numpy.core.fromnumeric import mean
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
import pandas as pd
import optuna
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pathlib import Path, PurePath
import pdb
import warnings
from datetime import datetime
from gsdc.gsdc import *
import pretty_errors

# =========================================================================
t1 = datetime.now()
print(f'\033[1;93m{t1}\033[0m')

pd.options.display.max_colwidth = 999
pd.options.display.max_rows = 999
warnings.filterwarnings('ignore')
# optuna.logging.set_verbosity(optuna.logging.WARNING)
# warnings.filterwarnings('default')
# =========================================================================
# preamble
# =========================================================================
elias_kaggle = Path('/Users/meg/ka/k6/')

if Path('.').cwd() != elias_kaggle:
    os.chdir(elias_kaggle)
Path('.').cwd()
# ========================================================
# directory
# ========================================================
ROOT = Path('../input/google-smartphone-decimeter-challenge/')
TRAIN = ROOT/'train'
TEST = ROOT/'test'

TRAIN_BASELINE = ROOT / 'baseline_locations_train.csv'
TEST_BASELINE = ROOT / 'baseline_locations_test.csv'
SAMPLE_SUBMISSION = ROOT / 'sample_submission.csv'

sample_sub = pd.read_csv(SAMPLE_SUBMISSION)
sub_cols = sample_sub
lat_cols = ['latDeg', 'lngDeg']
phone_cols = ['phone', 'millisSinceGpsEpoch']
# ========================================================
# v7 : v6 with weight
# ========================================================
sub_cols
# PRED_LIST = [Path('./sub/sub_4.970.csv'),
#              Path('./sub/sub_4.980.csv'),
#              #             Path('./sub/sub_4.995.csv'),
#              #             Path('./sub/sub_4.993.csv'),
#              Path('./sub/sub_4.985.csv')]

PRED_LIST = [Path('./sub/sub_4.970.csv'),
             Path('./sub/sub_4.962.csv'),
             #             Path('./sub/sub_4.995.csv'),
             #             Path('./sub/sub_4.993.csv'),
             Path('./sub/sub_4.963.csv')]

# ========================================================
# PRED_STORE = Path('./sub/sub_s3.csv')
SUB_PATH = Path('./submission.csv')
weight = [0.2, 0.2, 0.2, 0.2, 0.2]
weight = [0.25, 0.25, 0.25, 0.25]
weight = [0.5, 0.3, 0.2]
weight = [1/3, 1/3, 1/3]


# ========================================================
# pred_site_col = ['site_path_timestamp', 'site', 'path']
# pred_site_col = ['site_path_timestamp', 'site', 'path', 'timestamp', 'floor']

# ========================================================
row_check = 4048


pred_list = []
for i_pred, pred_path in enumerate(PRED_LIST):

    pred_one = pd.read_csv(pred_path)
    pred_one = pred_one[lat_cols]
    pred_one['latDeg'] = pred_one['latDeg'].mul(weight[i_pred], fill_value=0)
    pred_one['lngDeg'] = pred_one['lngDeg'].mul(weight[i_pred], fill_value=0)

    pred_list.append(pred_one)
    print(pred_path)
    print(weight[i_pred])
    print(pred_one.iloc[row_check][lat_cols])

#    pred_site_test = pd.concat(pred_list, axis=1)
pred_stack = pd.concat(pred_list, axis=0)
pred_agg = pred_stack.groupby(pred_stack.index)

pred_agg = pred_agg.sum()
sample_sub = sample_sub[phone_cols]
sub = pd.concat([sample_sub, pred_agg], axis=1)
print(sub.head(10))

print(f'\033[36m{sub.iloc[row_check][lat_cols]}\033[0m')

#    print(pred_site.head(5))
# get string columns
print(f'\033[1;33mshape {sub.shape}')
print(f'\033[1;92mnan\n{sub.isna().sum()}\033[0m')


sub.to_csv(str(SUB_PATH), index=False)

print(f'\033[35mstored in \033[0m{str(SUB_PATH)}')

# pred_site_stack[['x']].head(3)
# pred_site_test[['x']].head(3)
# pred_site.head(3)


# ===========================================================================
# process summary
# ===========================================================================
