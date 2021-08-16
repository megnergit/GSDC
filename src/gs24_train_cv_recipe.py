import re
from numpy.core.fromnumeric import mean
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from folium import plugins
import folium
from scipy import signal
import pandas as pd
from pyproj.transformer import TransformerFromCRS
import optuna
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pathlib import Path, PurePath
from pyproj import Proj, transform
import pdb
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from tqdm import tqdm
# from IPython.display import display
import simdkalman
import warnings
from datetime import datetime
from sklearn import preprocessing
import math
import sys
from gsdc.gsdc import *
import pretty_errors

# =========================================================================
t1 = datetime.now()
print(f'\033[1;93m{t1}\033[0m')

pd.options.display.max_colwidth = 999
pd.options.display.max_rows = 999
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
# warnings.filterwarnings('default')
# =========================================================================
# preamble
# =========================================================================
elias_kaggle = Path('/Users/meg/ka/k6/')

if Path('.').cwd() != elias_kaggle:
    os.chdir(elias_kaggle)
Path('.').cwd()
# =========================================================================
t = datetime.now()
sub1 = './sub/sub_'+t.date().__str__() + '.csv'
log1 = './log/d_'+t.date().__str__() + '.csv'

msge = 'millisSinceGpsEpoch'
msge_sample = 'millisSinceGpsEpoch_sample'
# =========================================================================
# ROOT.exists()
ROOT = Path('../input/google-smartphone-decimeter-challenge/')
TRAIN = ROOT/'train'
TEST = ROOT/'test'

TRAIN_BASELINE = ROOT / 'baseline_locations_train.csv'
TEST_BASELINE = ROOT / 'baseline_locations_test.csv'
SAMPLE_SUBMISSION = ROOT / 'sample_submission.csv'
DLAT_INFO = './log/d_2021-07-09.csv'
PAIRS_PHONES = './log/pairs_phones.csv'
PAIRS_ALL = './log/pairs_all_v3.csv'

GNSS_LOG = sorted(list(TRAIN.glob('*/*/*GnssLog.txt')))

sub_sample = pd.read_csv(SAMPLE_SUBMISSION)
sub_cols = sub_sample.columns

# pairs_all = road_matching(TEST, TRAIN, TEST_BASELINE, TRAIN_BASELINE)
# pairs_phones = get_pairs_phones(pairs_all)
pairs_all = pd.read_csv(PAIRS_ALL)
pairs_phones = pd.read_csv(PAIRS_PHONES)

path_test_routes = sorted(list(TEST.glob('*/*/*_derived.csv')))

path_test_routes = list(TEST.glob('*/*/*_derived.csv'))
path_sjc = [p for p in path_test_routes if str(p).find('SJC') != -1]
path_mtv = [p for p in path_test_routes if str(p).find('SJC') == -1]

collection_all = [(i, c) for i, c in enumerate(pairs_all['test'])]

# # =========================================================================
'''
- offset => (do not apply if no matching phone)
- spline
- snap large course off to gt
(- snap small off to nearest gt) => oversampling necessary
'''
dlat_info = pd.read_csv(DLAT_INFO)
# dlat_info

# # =========================================================================
# collectoin base
# # =========================================================================
score_list = []
score_train_list = []

# -------------------------------------------
# menu
do_clip = True
do_kalman = True
do_zero = True
do_position_shift = True

do_gaussian = True
do_course_off = True
do_spline = True
do_savgol = True
do_mean_offset = True

# -------------------------------------------
# params
# clip
abs_clip, d_cutoff = True, 20.0  # absolute
abs_clip, d_cutoff = False, 2.0

# zero
rolling_window = 16
rolling_window = 8
dist_crit = 12.5  # stationary criterion in m
d_crit = 0.005  # stationary criterion in deg
d_crit = 0.1  # stationary criterion in deg
n_idle = 4  # stationary criterion

detection = ''
detection_train = ''
n_replaced = 0
n_replaced_train = 0
n_total = 1
n_total_train = 1

# course off
off_crit = 12.5  # correct course off [m]
# off_crit = 8  # correct course off [m]

# spline
smooth_factor = 1e-7

# savgol
n_window, n_order = 7, 4

# # =========================================================================
detection = ''
detection_train = ''
n_replaced = 0
n_replaced_train = 0
n_total = 1
n_total_train = 1
abs_clip = False
n_tail = 4
n_head = 4

params_mtv = dict(d_cutoff=2.0,
                  nn=3,
                  n_tail=12,
                  n_head=12,
                  rolling_window=8,
                  dist_crit=12.5,
                  d_crit=0.005,  # stationary criterion in deg
                  n_idle=8,
                  off_crit=12.5,
                  kf_obs_noise=5e-5,
                  n_window=7,
                  n_order=3,
                  phone_mean=[])  # correct course off [m]

params_mtv0 = dict(d_cutoff=2.5,
                   nn=3,
                   n_tail=12,
                   #                   n_head=12,
                   n_head=12,
                   rolling_window=8,
                   dist_crit=12.5,
                   d_crit=0.01,  # stationary criterion in deg
                   n_idle=4,
                   off_crit=12.5,
                   kf_obs_noise=1e-5,
                   n_window=7,
                   n_order=3,
                   phone_mean=['Pixel4XL'])  # correct course off [m]

params_mtv1 = dict(d_cutoff=2.5,
                   nn=3,
                   n_tail=48,
                   n_head=24,
                   rolling_window=8,
                   dist_crit=12.5,
                   d_crit=0.01,  # stationary criterion in deg
                   n_idle=4,
                   off_crit=12.5,
                   kf_obs_noise=5e-5,
                   n_window=7,
                   n_order=3,
                   phone_mean=[])  # correct course off [m]

params_mtv2 = dict(d_cutoff=2.5,
                   nn=3,
                   n_tail=12,
                   n_head=12,
                   rolling_window=8,
                   dist_crit=12.5,
                   d_crit=0.01,  # stationary criterion in deg
                   n_idle=4,
                   off_crit=12.5,
                   kf_obs_noise=5e-5,
                   n_window=7,
                   n_order=3,
                   phone_mean=[])  # correct course off [m]

params_mtv3 = dict(d_cutoff=2.0,
                   d_crit=0.005,  # stationary criterion in deg
                   n_tail=12,
                   n_head=12,
                   n_idle=8,
                   nn=3,
                   rolling_window=8,
                   dist_crit=12.5,
                   off_crit=12.5,
                   kf_obs_noise=1e-5,
                   n_window=7,
                   n_order=3,
                   phone_mean=[])  # correct course off [m]


params_mtv4 = dict(d_cutoff=2.5,
                   n_tail=12,
                   n_head=12,
                   d_crit=0.005,  # stationary criterion in deg
                   n_idle=8,

                   rolling_window=8,
                   dist_crit=12.5,
                   nn=3,
                   off_crit=16,
                   kf_obs_noise=5e-5,
                   n_window=7,
                   n_order=3,
                   phone_mean=[])  # correct course off [m]


params_mtv5 = dict(d_cutoff=2.0,
                   nn=3,
                   n_tail=12,
                   n_head=16,
                   rolling_window=8,
                   dist_crit=12.5,
                   d_crit=0.01,  # stationary criterion in deg
                   n_idle=8,
                   off_crit=12.5,
                   kf_obs_noise=5e-5,
                   n_window=7,
                   n_order=3,
                   phone_mean=[])  # correct course off [m]


params_mtv6 = dict(d_cutoff=2.0,
                   nn=3,
                   n_tail=16,
                   n_head=32,
                   rolling_window=8,
                   dist_crit=12.5,
                   d_crit=0.01,  # stationary criterion in deg
                   n_idle=8,
                   off_crit=12.5,
                   kf_obs_noise=5e-5,
                   n_window=7,
                   n_order=3,
                   phone_mean=[])  # correct course off [m]
#                   phone_mean=['Pixel4XLModded', 'Pixel4'])  # correct course off [m]

params_mtv7 = dict(
    #                   d_cutoff=1.5,
    d_cutoff=2.5,
    n_tail=24,
    n_head=24,
    # params_mtv7 = dict(d_cutoff=1.5,
    d_crit=0.05,  # stationary criterion in deg
    n_idle=2,
    nn=3,
    rolling_window=32,
    dist_crit=18,
    off_crit=16,
    kf_obs_noise=5e-6,
    n_window=7,
    n_order=3,
    #                   phone_mean=['Mi8'])  # correct course off [m]
    phone_mean=[])  # correct course off [m]


params_mtv8 = dict(d_cutoff=2.5,
                   n_tail=24,
                   n_head=0,
                   d_crit=0.005,  # stationary criterion in deg
                   n_idle=8,
                   nn=5,
                   rolling_window=8,
                   dist_crit=12.5,
                   off_crit=16,
                   kf_obs_noise=5e-5,
                   n_window=7,
                   n_order=3,
                   phone_mean=[])  # correct course off [m]

params_mtv9 = dict(d_cutoff=2.5,
                   n_tail=12,
                   n_head=12,
                   d_crit=0.005,  # stationary criterion in deg
                   n_idle=8,
                   nn=3,
                   rolling_window=8,
                   dist_crit=12.5,
                   off_crit=16,
                   kf_obs_noise=1e-5,
                   n_window=7,
                   n_order=3,
                   phone_mean=[])  # correct course off [m]


params_mtv10 = dict(d_cutoff=2.0,
                    nn=3,
                    n_tail=16,
                    n_head=16,
                    rolling_window=8,
                    dist_crit=12.5,
                    d_crit=0.02,  # stationary criterion in deg
                    n_idle=4,
                    off_crit=12.5,
                    kf_obs_noise=5e-5,
                    n_window=7,
                    n_order=3,
                    phone_mean=[])  # correct course off [m]


params_mtv11 = dict(d_cutoff=2.0,
                    n_tail=8,
                    n_head=56,
                    d_crit=0.025,  # stationary criterion in deg
                    n_idle=4,
                    nn=3,
                    rolling_window=8,
                    dist_crit=12.5,
                    off_crit=16,
                    n_window=7,
                    kf_obs_noise=5e-5,
                    n_order=3,
                    phone_mean=[])  # correct course off [m]

params_mtv12 = dict(d_cutoff=2.0,
                    nn=3,
                    n_tail=12,
                    n_head=12,
                    rolling_window=8,
                    dist_crit=12.5,
                    d_crit=0.005,  # stationary criterion in deg
                    n_idle=8,
                    off_crit=12.5,
                    kf_obs_noise=5e-5,
                    n_window=7,
                    n_order=3,
                    #                    phone_mean=[])  # correct course off [m]
                    phone_mean=['Pixel4', 'Pixel4Modded', 'Pixel5'])  # correct course off [m]

# ------------------------------------------------------------------
params_sjc13 = dict(d_cutoff=1.5,
                    n_tail=12,
                    n_head=12,
                    d_crit=0.01,  # stationary criterion in deg
                    n_idle=8,
                    nn=5,
                    rolling_window=8,
                    dist_crit=12.5,
                    off_crit=12.5,
                    kf_obs_noise=5e-5,
                    n_window=7,
                    n_order=3,
                    phone_mean=['Pixel5'])  # correct course off [m]
#                    phone_mean=[])  # correct course off [m]

params_mtv14 = dict(d_cutoff=2.5,
                    n_tail=12,
                    n_head=12,
                    d_crit=0.05,  # stationary criterion in deg
                    n_idle=8,
                    nn=3,
                    rolling_window=8,
                    dist_crit=12.5,
                    off_crit=16,
                    kf_obs_noise=5e-5,
                    n_window=7,
                    n_order=3,
                    phone_mean=['Pixel5'])  # correct course off [m]

params_mtv15 = dict(d_cutoff=2.0,
                    nn=3,
                    n_tail=12,
                    n_head=12,
                    rolling_window=8,
                    dist_crit=12.5,
                    d_crit=0.005,  # stationary criterion in deg
                    n_idle=8,
                    off_crit=12.5,
                    kf_obs_noise=5e-5,
                    n_window=7,
                    n_order=3,
                    phone_mean=[])  # correct course off [m]

# ------------------------------------------------------------------
params_sjc1 = dict(d_cutoff=1.5,
                   n_tail=12,
                   n_head=12,
                   d_crit=0.01,  # stationary criterion in deg
                   n_idle=8,
                   nn=5,
                   rolling_window=8,
                   dist_crit=12.5,
                   off_crit=12.5,
                   kf_obs_noise=5e-5,
                   n_window=7,
                   n_order=3,
                   phone_mean=[])  # correct course off [m]

params_sjc2 = dict(d_cutoff=2.0,
                   n_tail=12,
                   n_head=12,
                   d_crit=0.01,  # stationary criterion in deg
                   n_idle=8,
                   nn=3,
                   rolling_window=8,
                   dist_crit=12.5,
                   off_crit=18,
                   kf_obs_noise=1e-5,
                   n_window=7,
                   n_order=3,
                   phone_mean=[])  # correct course off [m]

params_sjc3 = dict(d_cutoff=1.5,
                   n_tail=12,
                   n_head=12,
                   d_crit=0.02,  # stationary criterion in deg
                   n_idle=8,
                   nn=5,
                   rolling_window=8,
                   dist_crit=12.5,
                   off_crit=6.4,  # correct course off [m]
                   kf_obs_noise=1e-5,
                   n_window=7,
                   n_order=2,
                   phone_mean=[])  # correct course off [m]

# ------------------------------------------------------------------
# 5 6 8 10 11 leave t/h

# recipe_mtv = ['c', 'k', 'p', 'p3']

recipe_mtv = ['c', 'k', 'p']
recipe_mtv0 = ['t', 'h', 'c', 'z', 'k', 'p', 'p3']
recipe_mtv1 = ['t', 'h', 'c', 'z', 'k', 'p', 'p3']
recipe_mtv2 = ['t', 'h', 'c', 'z', 'k', 'p']
recipe_mtv3 = ['c', 'k', 'p', 'p3']
recipe_mtv4 = ['c', 'z', 'k', 'p', 'p3']
recipe_mtv5 = ['t', 'h', 'c', 'z', 'k', 'p', 'p3']
recipe_mtv6 = ['t', 'h', 'c', 'z', 'k', 'p', 'p3']
recipe_mtv7 = ['t', 'h', 'c', 'z', 'k',  'p', 'p3']  # tail

recipe_mtv8 = ['t', 'h', 'c', 'z', 'k', 'p', 'p3']
recipe_mtv9 = ['c', 'z', 'k', 's', 'p']

recipe_mtv10 = ['t', 'h', 'c', 'z', 'k', 'p', 'p3']
recipe_mtv11 = ['t', 'h', 'c', 'z', 'k', 'p', 'p3']
recipe_mtv12 = ['c', 'k', 'p']

recipe_sjc13 = ['c', 'k', 'p', 'p3', 'p3']
recipe_mtv14 = ['c',  'k', 's', 'p', 'p3']
# recipe_mtv14 = ['c']
recipe_sjc1 = ['c', 'x', 'z', 'k', 'v', 'p']
recipe_sjc2 = ['c', 'x', 'z', 'k', 'v', 'p']
recipe_sjc3 = ['c', 'x', 'z', 'k', 'c', 'v', 'p']

# recipe_mtv0 = ['t', 'h', 'c', 'z', 'k', 'p', 'p3']
# recipe_mtv1 = ['t', 'h', 'c', 'z', 'k', 'p', 'p3']
# recipe_mtv2 = ['t', 'h', 'c', 'z', 'k', 'p', 'p3']
# recipe_mtv3 = ['c', 'k', 'p', 'p3']
# recipe_mtv4 = ['c', 'z', 'k', 'p', 'p3']
# recipe_mtv5 = ['t', 'h', 'c', 'z', 'k', 'p', 'p3']
# recipe_mtv6 = ['t', 'h', 'c', 'z', 'k', 'p', 'p3']
# recipe_mtv7 = ['t', 'h', 'c', 'z', 'k',  'p', 'p3']  # tail

# recipe_mtv8 = ['t', 'h', 'c', 'z', 'k', 'p', 'p3']
# recipe_mtv9 = ['c', 'z', 'k', 's', 'p']

# recipe_mtv10 = ['t', 'h', 'c', 'z', 'k', 'p', 'p3']
# recipe_mtv11 = ['t', 'h', 'c', 'z', 'k', 'p', 'p3']

# recipe_sjc13 = ['c', 'k', 'p', 'p3', 'p3']
# recipe_mtv14 = ['c',  'k', 's', 'p', 'p3']
# # recipe_mtv14 = ['c']
# recipe_sjc1 = ['c', 'x', 'z', 'k', 'v', 'p']
# recipe_sjc2 = ['c', 'x', 'z', 'k', 'v', 'p']
# recipe_sjc3 = ['c', 'x', 'z', 'k', 'c', 'v', 'p']

# recipe_mtv = ['c', 'k', 'p']
# recipe_mtv0 = ['t', 'h', 'c', 'z', 'k', 'p']
# recipe_mtv1 = ['t', 'h', 'c', 'z', 'k', 'p']
# recipe_mtv2 = ['c', 'z', 'k', 'p']
# recipe_mtv3 = ['c', 'k', 'p']
# recipe_mtv4 = ['c', 'z', 'k', 'p']
# recipe_mtv5 = ['c', 'z', 'k', 'p']
# recipe_mtv6 = ['c', 'z', 'k', 'p']
# recipe_mtv7 = ['t', 'h', 'c', 'z', 'k',  'p']

# recipe_mtv8 = ['c', 'z', 'k', 'p']
# recipe_mtv9 = ['c', 'z', 'k', 's', 'p']
# recipe_mtv10 = ['c', 'z', 'k', 'p']

# recipe_mtv11 = ['c', 'z', 'k', 'p']
# recipe_mtv14 = ['c', 'k', 's', 'p']
# # recipe_mtv14 = ['c']
# recipe_sjc1 = ['c', 'x', 'z', 'k', 'v', 'p']
# recipe_sjc2 = ['c', 'x', 'z', 'k', 'v', 'p']
# recipe_sjc3 = ['c', 'x', 'z', 'k', 'c', 'v', 'p']

recipe_sets = [
    (recipe_mtv0, params_mtv0),  # 0
    (recipe_mtv1, params_mtv1),  # 1
    (recipe_mtv2, params_mtv2),  # 2
    (recipe_mtv3, params_mtv3),  # 3
    (recipe_mtv4, params_mtv4),  # 4
    (recipe_mtv5, params_mtv5),  # 5
    (recipe_mtv6, params_mtv6),  # 6
    (recipe_mtv7, params_mtv7),  # 7
    (recipe_mtv8, params_mtv8),  # 8
    (recipe_mtv9, params_mtv9),  # 9
    (recipe_mtv10, params_mtv10),  # 10
    (recipe_mtv11, params_mtv11),  # 11
    (recipe_mtv12, params_mtv12),  # 12

    (recipe_sjc13, params_sjc13),  # 13
    (recipe_mtv14, params_mtv14),  # 14
    (recipe_mtv, params_mtv),  # 15

    (recipe_sjc1, params_sjc1),  # 16
    (recipe_sjc2, params_sjc2),  # 17
    (recipe_sjc3, params_sjc3)]  # 18

recipe_sets = pd.DataFrame(recipe_sets, columns=['recipe', 'params'])
# =========================================================================

# message = "zero 0.005 dist_crit 24 clip cutoff 20"
# print(f'\033[0m{message}\033[0m')
# =========================================================================
# -------------------------------------------
df_train_bs = pd.read_csv(TEST_BASELINE)
df_test_bs = pd.read_csv(TEST_BASELINE)
df_train_bs['in_motion'] = 'm'
# -------------------------------------------
# HERE1
# -------------------------------------------
n_cols = 4
n_rows = 4
i_route = 4
collection_all[i_route:i_route+1]
data, layout = visualize_tile_init(
    #    n_rows, n_cols, collection_all[i_route:i_route+1])
    n_rows, n_cols, collection_all)
for i, (i_c, c) in enumerate(collection_all[i_route:i_route+16]):
    # for i, (i_c, c) in enumerate(collection_all):
    # -------------------------------------------

    c_t = pairs_all.loc[pairs_all['test'] == c, 'train'].values[0]
#    print(i_c, recipe_sets)

    print('')
    print(
        f'\033[31m==============================================================\033[0m')
    print(f'\033[1;92m{i_c} \033[1;33m{c}\033[0m / \033[1;32m{c_t}')
    phone_list = pairs_phones.loc[pairs_phones['test']
                                  == c, 'phoneName'].values

    ex_list = []
    idx_list = []
    # =========================================================================
    #  recipe
    # =========================================================================

    recipe_set = recipe_sets.iloc[i_c]
    recipe = recipe_set['recipe']
    params = recipe_set['params']

    n_recipe = len(recipe)

    d_cutoff = params['d_cutoff']
    d_crit = params['d_crit']
    n_idle = params['n_idle']
    nn = params['nn']

    rolling_window = params['rolling_window']
    dist_crit = params['dist_crit']

    off_crit = params['off_crit']

    n_window = params['n_window']
    n_order = params['n_order']

    phone_selection = params['phone_mean']
    n_tail = params['n_tail']
    n_head = params['n_head']
    kf_obs_noise = params['kf_obs_noise']

#    print(phone_selection)
#    if len(phone_selection) == 0:
#        phone_selection = phone_list
#    print(f'phone_selection {phone_selection}')

# -------------------------------------------
# HERE2
# -------------------------------------------

    # data, layout = visualize_tile_init(
    #     n_rows, n_cols, list(enumerate(phone_list)))

    for phone in phone_list:
        #    p = p_list[0]
        p = TEST/c/phone/(phone + '_derived.csv')
        #    print(c, p_list[0], p, p.exists())

        df_test_routes = pd.read_csv(p)
        df_test_ex, idx_ex = extract_baseline(df_test_bs, df_test_routes)
        df_test_ex['in_motion'] = 'm'
        idx_list.append(idx_ex)

        df_train_gt = get_matching_gt(df_test_ex, pairs_all)
        df_train_ex = get_matching_bs(df_test_ex, pairs_all)
        phone_train = df_train_ex['phoneName'].drop_duplicates().values[0]

        # # =========================================================================
        print(
            f'\033[34m--------------------------------------------------------------\033[0m')
        print(f'\033[1;96m{phone}\033[1;0m / \033[1;92m{phone_train}\033[0m')
        print(
            f'\033[34m--------------------------------------------------------------\033[0m')
        # # =========================================================================

        p_gnss = str(p).replace('_derived.csv', '_GnssLog.txt')
        sample = gnss_log_to_dataframes(p_gnss)

        df_test_ex_0 = df_test_ex.copy()  # before
        df_train_ex_0 = df_train_ex.copy()  # before

        acce_test, mag_test = get_acce_test(sample, df_test_ex)
        acce_train, mag_train = get_acce_train(sample, df_train_gt)

        show_score_proc_double(df_test_ex_0, df_test_ex_0,
                               df_train_ex_0, df_train_ex_0, df_train_gt, 'START ')

        # # =========================================================================
        # start processing according to recipe
        # # --------------------------------------------------------------------------

        for i_recipe in range(n_recipe):
            if recipe[i_recipe] == '-':
                pass

            elif recipe[i_recipe] == 'c':
                df_test_ex, df_train_ex = clip_module(
                    df_test_ex, df_train_ex, df_train_gt, d_cutoff, abs_clip, nn)

            elif recipe[i_recipe] == 'z':
                df_test_ex, df_train_ex, detection, detection_train, n_replaced, n_total, n_replaced_train, n_total_train = zero_module(
                    df_test_ex, df_train_ex, df_train_gt, acce_test, acce_train,  rolling_window, dist_crit, d_crit, n_idle)

            elif recipe[i_recipe] == 'k':
                #                df_test_ex, df_train_ex = kalman_module(
                df_test_ex, df_train_ex = kalman_module2(
                    df_test_ex, df_train_ex, df_train_gt, kf_obs_noise, n_head)

            elif recipe[i_recipe] == 'x':

                df_test_ex, df_train_ex = course_off_module(
                    df_test_ex, df_train_ex, df_train_gt, off_crit)

            elif recipe[i_recipe] == 'g':

                df_test_ex, df_train_ex = gaussian_module(
                    df_test_ex, df_train_ex, df_train_gt)

            elif recipe[i_recipe] == 's':
                df_test_ex, df_train_ex = spline_module(
                    df_test_ex, df_train_ex, df_train_gt, smooth_factor)

            elif recipe[i_recipe] == 'v':
                df_test_ex, df_train_ex = savgol_module(
                    df_test_ex, df_train_ex, df_train_gt, n_window, n_order)

            elif recipe[i_recipe] == 'p':

                df_test_ex, df_train_ex = position_shift_module(
                    df_test_ex, df_train_ex, df_train_gt, sub_cols)

            elif recipe[i_recipe] == 'p3':
                df_test_ex, df_train_ex = position_shift_module3(
                    df_test_ex, df_train_ex, df_train_gt)

            elif recipe[i_recipe] == 'o':
                df_test_ex, df_train_ex = mean_offsets_module(
                    df_test_ex, df_train_ex, df_train_gt, pairs_all, DLAT_INFO)

            elif recipe[i_recipe] == 't':
                df_test_ex, df_train_ex = tail_module(
                    df_test_ex, df_train_ex, df_train_gt, n_tail)

            elif recipe[i_recipe] == 'h':
                df_test_ex, df_train_ex = head_module(
                    df_test_ex, df_train_ex, df_train_gt, n_head)
            else:
                pass

        # ===========================================================================
        # summary
        # ===========================================================================

        s_before, s_after = return_scores(
            df_test_ex, df_test_ex_0, df_train_gt)
        # --------------------------------------------------------------------------
        score_dict = dict(
            collectionName=[c],
            phoneName=[phone],
            after=[s_after[0]],
            score_after_p50=[s_after[1]],
            score_after_p95=[s_after[2]],
            before=[s_before[0]],
            score_before_p50=[s_before[1]],
            score_before_p95=[s_before[2]],
            diff=[s_before[0] - s_after[0]])

        score_df = pd.DataFrame(score_dict)
        score_list.append(score_df)
        # --------------------------------------------------------------------------
        sb_train, sa_train = return_scores(
            df_train_ex, df_train_ex_0, df_train_gt)

        score_train_dict = dict(
            collectionName=[c_t],
            phoneName=[phone_train],
            after=[sa_train[0]],
            score_after_p50=[sa_train[1]],
            score_after_p95=[sa_train[2]],
            before=[sb_train[0]],
            score_before_p50=[sb_train[1]],
            score_before_p95=[sb_train[2]],
            diff=[sb_train[0] - sa_train[0]])

        score_train_df = pd.DataFrame(score_train_dict)
        score_train_list.append(score_train_df)

# -------------------------------------------
# HERE3
# -------------------------------------------

        # data, layout = visualize_tile_append(
        #     df_test_ex_0, df_test_ex, data, layout, i)
        # i = i + 1

#        --------------------------------------------------------------------------
        df_test_bs.loc[idx_ex, 'latDeg'] = df_test_ex['latDeg'].to_numpy()
        df_test_bs.loc[idx_ex, 'lngDeg'] = df_test_ex['lngDeg'].to_numpy()
        df_test_bs.loc[idx_ex,
                       'in_motion'] = df_test_ex['in_motion'].to_numpy()
    # # --------------------------------------------------------------------------
        ex_list.append(df_test_ex)
#        pdb.set_trace()

    # # --------------------------------------------------------------------------
    # phone mean # braucht kein train
    # # --------------------- -----------------------------------------------------
    # test
    df_test_bs_b = df_test_bs.copy()

    if len(phone_selection) == 0:
        df_test_bs = phone_mean_post_processing(df_test_bs, c)
    else:
        df_test_bs = phone_mean_selected(df_test_bs, c, phone_selection)

    df_ex = df_test_bs.loc[df_test_bs['collectionName']
                           == c, :].reset_index(drop=True)
    df_ex_b = df_test_bs_b.loc[df_test_bs['collectionName'] == c, :].reset_index(
        drop=True)

    print(
        f'\033[32m--------------------------------------------------------------\033[0m')
    show_score_proc(df_ex, df_ex_b, df_train_gt, 'phone ')

    data, layout = visualize_tile_append(
        df_test_ex_0, df_ex, data, layout, i)

#    visualize_tile_show(data, layout)
# ===========================================================================
# process summary
# ===========================================================================
print(f'\033[35mclip \033[0m{d_cutoff}')
print(
    f'\033[35mzero \033[32m{detection} {detection_train} \033[0m{dist_crit} {d_crit} {n_replaced}/{n_total} {n_replaced/n_total:.2f}')
# ===========================================================================
visualize_tile_show(data, layout)
score_summary = pd.concat(score_list, axis=0).reset_index(drop=True)
score_train_summary = pd.concat(
    score_train_list, axis=0).reset_index(drop=True)

final_score_test = score_summary['after'].mean()
final_score_train = score_train_summary['after'].mean()

print(f'\033[1;93mFINAL       {final_score_test:.3f}\033[0m')
print(f'\033[1;92mFINAL TRAIN {final_score_train:.3f}\033[0m')

pd.options.display.float_format = '{:,.2f}'.format

print(score_summary[['collectionName',
                     'phoneName', 'after', 'before', 'diff']])

print(f'-----------------------------------------------------')
print(score_train_summary[['collectionName',
                           'phoneName', 'after', 'before', 'diff']])
print(f'-----------------------------------------------------')

# --------------------------------------------------------------------------
# ===========================================================================
# preapre submission
# =========================================================================
# sub = pd.read_csv(SAMPLE_SUBMISSION)
print(f'df_test_bs.shape {df_test_bs.shape}')
print(f'-----------------------------------------------------')
print(f'df_test_bs.isna \n{df_test_bs.isna().sum()}')
print(f'-----------------------------------------------------')
sub = df_test_bs[sub_cols]
sub.to_csv(sub1, index=False)
# -------------------------------------------
# HERE4
# -------------------------------------------
pdb.set_trace()
sub.to_csv('submission_p3-revised.csv', index=False)

print(f'-----------------------------------------------------')
t2 = datetime.now()
print(f'\033[1;93m{t2}\033[0m')
print(f'\033[1;95m{t2-t1}\033[0m')
# ===========================================================================
# do until here
# =========================================================================
# os.system(message)
