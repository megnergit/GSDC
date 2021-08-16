from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.signal import butter, lfilter
from plotly.subplots import make_subplots
import pandas as pd
from pandas.core.algorithms import mode
from scipy.signal.windows import get_window
from pyproj.transformer import TransformerFromCRS
from vincenty import vincenty
import seaborn as sns
import plotly.express as px
import optuna
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pathlib import Path, PurePath
import pyproj
from pyproj import Proj, transform
import pdb
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
import plotly.graph_objs as go
from tqdm import tqdm
from IPython.display import display
import simdkalman
import warnings
from datetime import datetime
from sklearn import preprocessing
# import math
# from math import *

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from cv2 import Rodrigues
from math import sin, cos, atan2, sqrt, acos, atan
from math import *
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.stats import sigmaclip
from scipy.signal import savgol_filter

import folium
from folium import plugins

optuna.logging.set_verbosity(optuna.logging.WARNING)
# =========================================================================
# set params
# =========================================================================


# =========================================================================
# modules
# =========================================================================


def clip_module(df_test_ex, df_train_ex, df_train_gt, d_cutoff, abs_clip, nn):

    df_test_ex_b = df_test_ex.copy()
    df_train_ex_b = df_train_ex.copy()

    df_test_ex = clip_baseline(
        df_test_ex, nn=nn, d_cutoff=d_cutoff, inplace=True, abs_dist=abs_clip)
    df_train_ex = clip_baseline(
        df_train_ex, nn=nn, d_cutoff=d_cutoff, inplace=True, abs_dist=abs_clip)

    show_score_proc_double(
        df_test_ex, df_test_ex_b, df_train_ex, df_train_ex_b, df_train_gt, 'clip  ')

    return df_test_ex, df_train_ex

# --------------------------------------------------------------------------


def tail_module(df_test_ex, df_train_ex, df_train_gt, n_tail=4):

    df_test_ex_b = df_test_ex.copy()
    df_train_ex_b = df_train_ex.copy()

    df_test_ex = cut_tail(df_test_ex, n_tail)
#    df_train_ex = cut_tail(df_train_ex, 0)  # do nto apply to train

    show_score_proc_double(
        df_test_ex, df_test_ex_b, df_train_ex, df_train_ex_b, df_train_gt, 'tail  ')

    return df_test_ex, df_train_ex

# --------------------------------------------------------------------------


def head_module(df_test_ex, df_train_ex, df_train_gt, n_head=4):

    df_test_ex_b = df_test_ex.copy()
    df_train_ex_b = df_train_ex.copy()

    df_test_ex = cut_head(df_test_ex, n_head)
#    df_train_ex = cut_head(df_train_ex, 0)  # do not apply to train
#    print(f"\033[36mafter\n\033[0m{df_test_ex.iloc[:10]['latDeg']}")
#    print(f"\033[36mafter\n\033[0m{df_test_ex.iloc[:10]['lngDeg']}")

    show_score_proc_double(
        df_test_ex, df_test_ex_b, df_train_ex, df_train_ex_b, df_train_gt, 'head  ')

    return df_test_ex, df_train_ex

# --------------------------------------------------------------------------


def cut_tail(df_test_ex, n_tail):
    #    print(df_test_ex.iloc[-(n_tail+1)]['latDeg'])
    #    print(df_test_ex.iloc[-(n_tail+1)]['lngDeg'])
    #    lat_anchor = df_test_ex.iloc[-(n_tail+1)]['latDeg']
    #    lng_anchor = df_test_ex.iloc[-(n_tail+1)]['lngDeg']
    lat_anchor = df_test_ex.iloc[-(n_tail):]['latDeg'].mean()
    lng_anchor = df_test_ex.iloc[-(n_tail):]['lngDeg'].mean()

#    print(f"\033[32mbefore\n\033[0m{df_test_ex.iloc[-n_tail*2:]['latDeg']}")
#    print(f"\033[32mbefore\n\033[0m{df_test_ex.iloc[-n_tail*2:]['lngDeg']}")
    df_test_ex.iloc[-n_tail:]['latDeg'] = lat_anchor
    df_test_ex.iloc[-n_tail:]['lngDeg'] = lng_anchor

#    print(f"\033[36maftere\n\033[0m{df_test_ex.iloc[-n_tail*2:]['latDeg']}")
#    print(f"\033[36maftere\n\033[0m{df_test_ex.iloc[-n_tail*2:]['lngDeg']}")
#    pdb.set_trace()
    return df_test_ex


def cut_head(df_test_ex, n_head):

    #    lat_anchor = df_test_ex.iloc[n_head+1]['latDeg']
    #    lng_anchor = df_test_ex.iloc[n_head+1]['lngDeg']
    lat_anchor = df_test_ex.iloc[:n_head]['latDeg'].mean()
    lng_anchor = df_test_ex.iloc[:n_head]['lngDeg'].mean()
#    print(lat_anchor, lng_anchor)

    # print(f"\033[32mbefore\n\033[0m{df_test_ex.iloc[-n_head*2:]['latDeg']}")
    # print(f"\033[32mbefore\n\033[0m{df_test_ex.iloc[-n_head*2:]['lngDeg']}")
#    print(f"\033[32mbefore\n\033[0m{df_test_ex.iloc[:10]['latDeg']}")
#    print(f"\033[32mbefore\n\033[0m{df_test_ex.iloc[:10]['lngDeg']}")

    df_test_ex.iloc[:n_head]['latDeg'] = lat_anchor
    df_test_ex.iloc[:n_head]['lngDeg'] = lng_anchor

#    print(f"\033[36mafter\n\033[0m{df_test_ex.iloc[:10]['latDeg']}")
#    print(f"\033[36mafter\n\033[0m{df_test_ex.iloc[:10]['lngDeg']}")

    return df_test_ex

# --------------------------------------------------------------------------


def zero_module(df_test_ex, df_train_ex, df_train_gt, acce_test, acce_train,
                rolling_window, dist_crit, d_crit, n_idle):

    df_test_ex_b = df_test_ex.copy()
    df_train_ex_b = df_train_ex.copy()

    # --------------------------------------------------------
    if acce_test.shape[0] == 0:
        detection = 'displ'
        df_test_ex = is_stationary(df_test_ex,
                                   rolling_window=rolling_window, dist_crit=dist_crit)

    else:
        detection = 'accel'
        df_test_ex = is_stationary2(
            df_test_ex, acce_test, d_crit=d_crit)

    # --------------------------------------------------------
    if acce_train.shape[0] == 0:
        detection_train = 'displ'
        df_train_ex = is_stationary(df_train_ex,
                                    rolling_window=rolling_window, dist_crit=dist_crit)

    else:
        detection_train = 'accel'
        df_train_ex = is_stationary2(
            df_train_ex, acce_train, d_crit=d_crit)

    # --------------------------------------------------------
    df_test_ex, n_replaced, n_total = replace_stationary(
        df_test_ex, n_idle=n_idle)

    df_train_ex, n_replaced_train, n_total_train = replace_stationary(
        df_train_ex, n_idle=n_idle)

    show_score_proc_double(df_test_ex, df_test_ex_b,
                           df_train_ex, df_train_ex_b,
                           df_train_gt, 'zero  ')

    return df_test_ex, df_train_ex, detection, detection_train, n_replaced, n_total, n_replaced_train, n_total_train

# --------------------------------------------------------------------------


def kalman_module2(df_test_ex, df_train_ex, df_train_gt, kf_obs_noise, n_head):
    T = 1.0
    state_transition = np.array([[1, 0, T, 0, 0.5 * T ** 2, 0],
                                 [0, 1, 0, T, 0, 0.5 * T ** 2],
                                 [0, 0, 1, 0, T, 0],
                                 [0, 0, 0, 1, 0, T],
                                 [0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 1]])

    process_noise = np.diag([1e-5, 1e-5, 5e-6, 5e-6, 1e-6, 1e-6]
                            ) + np.ones((6, 6)) * 1e-9
    # process_noise = np.diag([1e-5, 1e-5, 5e-6, 5e-6, 1e-3, 1e-3]
    #                         ) + np.ones((6, 6)) * 1e-9

    observation_model = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
#     observation_noise = np.diag([5e-5, 5e-5]) + np.ones((2, 2)) * 1e-9
    observation_noise = np.diag(
        [kf_obs_noise, kf_obs_noise]) + np.ones((2, 2)) * 1e-9

    # --------------------------------------------
    kf = simdkalman.KalmanFilter(
        state_transition=state_transition,
        process_noise=process_noise,
        observation_model=observation_model,
        observation_noise=observation_noise)

    # --------------------------------------------

    df_test_ex_b = df_test_ex.copy()  # before
    df_train_ex_b = df_train_ex.copy()  # before

    # cut head
    df = df_test_ex.iloc[n_head:]

#    df_test_ex = apply_kf_smoothing(df_test_ex, kf_=kf)
    x = apply_kf_smoothing(df, kf_=kf)
#    print(x.head(10))
#    print(x.shape)
    df_test_ex.iloc[n_head:] = x
#    print(df_test_ex.head(9))

#     pdb.set_trace()

    df_train_ex = apply_kf_smoothing(df_train_ex, kf_=kf)

    show_score_proc_double(df_test_ex, df_test_ex_b,
                           df_train_ex, df_train_ex_b,
                           df_train_gt, 'kalman')

    return df_test_ex, df_train_ex
# def kalman_module(df_test_ex, df_train_ex, df_train_gt, kf):

# --------------------------------------------------------------------------


def kalman_module(df_test_ex, df_train_ex, df_train_gt, kf_obs_noise):
    T = 1.0
    state_transition = np.array([[1, 0, T, 0, 0.5 * T ** 2, 0],
                                 [0, 1, 0, T, 0, 0.5 * T ** 2],
                                 [0, 0, 1, 0, T, 0],
                                 [0, 0, 0, 1, 0, T],
                                 [0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 1]])

    process_noise = np.diag([1e-5, 1e-5, 5e-6, 5e-6, 1e-6, 1e-6]
                            ) + np.ones((6, 6)) * 1e-9
    # process_noise = np.diag([1e-5, 1e-5, 5e-6, 5e-6, 1e-3, 1e-3]
    #                         ) + np.ones((6, 6)) * 1e-9

    observation_model = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
#     observation_noise = np.diag([5e-5, 5e-5]) + np.ones((2, 2)) * 1e-9
    observation_noise = np.diag(
        [kf_obs_noise, kf_obs_noise]) + np.ones((2, 2)) * 1e-9

    kf = simdkalman.KalmanFilter(
        state_transition=state_transition,
        process_noise=process_noise,
        observation_model=observation_model,
        observation_noise=observation_noise)

    # --------------------------------------------

    df_test_ex_b = df_test_ex.copy()  # before
    df_train_ex_b = df_train_ex.copy()  # before

    df_test_ex = apply_kf_smoothing(df_test_ex, kf_=kf)
    df_train_ex = apply_kf_smoothing(df_train_ex, kf_=kf)

    show_score_proc_double(df_test_ex, df_test_ex_b,
                           df_train_ex, df_train_ex_b,
                           df_train_gt, 'kalman')

    return df_test_ex, df_train_ex


# --------------------------------------------------------------------------

def course_off_module(df_test_ex, df_train_ex, df_train_gt, off_crit):

    df_test_ex_b = df_test_ex.copy()
    df_train_ex_b = df_train_ex.copy()

    df_test_ex = snap_course_off(
        df_test_ex, df_train_gt, off_crit=off_crit)

    df_traint_ex = snap_course_off(
        df_train_ex, df_train_gt, off_crit=off_crit)

    show_score_proc_double(df_test_ex, df_test_ex_b,
                           df_train_ex, df_train_ex_b,
                           df_train_gt, 'course')

    return df_test_ex, df_train_ex


# --------------------------------------------------------------------------

def gaussian_module(df_test_ex, df_train_ex, df_train_gt):

    df_test_ex_b = df_test_ex.copy()
    df_train_ex_b = df_train_ex.copy()

    df_test_ex = apply_gauss_smoothing(
        df_test_ex, {'sz_1': 0.85, 'sz_2': 5.65, 'sz_crit': 1.5})

    df_train_ex = apply_gauss_smoothing(
        df_train_ex, {'sz_1': 0.85, 'sz_2': 5.65, 'sz_crit': 1.5})

    sb_tr, sa_tr = return_scores_train(
        df_train_ex, df_train_ex_b, df_train_gt)

    if (sa_tr[0] - sb_tr[0]) <= 0:
        df_train_ex = df_train_ex_b

    show_score_proc_double(df_test_ex, df_test_ex_b,
                           df_train_ex, df_train_ex_b,
                           df_train_gt, 'gauss ')

    return df_test_ex, df_train_ex

# --------------------------------------------------------------------------


def position_shift_module(df_test_ex, df_train_ex, df_train_gt, sub_cols):
    #         if do_position_shift and (phone_train == phone):

    df_test_ex_b = df_test_ex.copy()
    df_train_ex_b = df_train_ex.copy()

    df_test_ex = apply_position_shift(
        df_test_ex, df_train_ex, df_train_gt, sub_cols)

#            df_traint_ex = apply_position_shift(
#                df_train_ex, df_train_ex, df_train_gt, sub_cols)

    show_score_proc_double(df_test_ex, df_test_ex_b,
                           df_train_ex, df_train_ex_b,
                           df_train_gt, 'posit ')

    return df_test_ex, df_train_ex
# --------------------------------------------------------------------------


def position_shift_module3(df_test_ex, df_train_ex, df_train_gt):
    #         if do_position_shift and (phone_train == phone):

    df_test_ex_b = df_test_ex.copy()
    df_train_ex_b = df_train_ex.copy()

    df_test_ex = apply_position_shift3(
        df_test_ex, df_train_ex, df_train_gt)

#            df_traint_ex = apply_position_shift(
#                df_train_ex, df_train_ex, df_train_gt, sub_cols)

    show_score_proc_double(df_test_ex, df_test_ex_b,
                           df_train_ex, df_train_ex_b,
                           df_train_gt, 'posit3')

    return df_test_ex, df_train_ex

# --------------------------------------------------------------------------


def spline_module(df_test_ex, df_train_ex, df_train_gt, smooth_factor):

    df_test_ex_b = df_test_ex.copy()
    df_train_ex_b = df_train_ex.copy()

    df_test_ex = apply_spline(df_test_ex, smooth_factor=smooth_factor)
    df_train_ex = apply_spline(
        df_train_ex, smooth_factor=smooth_factor)

    if df_test_ex.isna().sum().sum() != 0:
        print(f'\033[1;91mcould not apply spline\033[0m')
        df_test_ex = df_test_ex_b.copy()

    show_score_proc_double(df_test_ex, df_test_ex_b,
                           df_train_ex, df_train_ex_b,
                           df_train_gt, 'spline')

    return df_test_ex, df_train_ex

# --------------------------------------------------------------------------


def savgol_module(df_test_ex, df_train_ex, df_train_gt, n_window, n_order):

    df_test_ex_b = df_test_ex.copy()
    df_train_ex_b = df_train_ex.copy()

    df_test_ex = apply_savgol(df_test_ex, n_window, n_order)
    df_train_ex = apply_savgol(
        df_train_ex, n_window, n_order)

    if df_test_ex.isna().sum().sum() != 0:
        print(f'\033[1;91mcould not apply savgol\033[0m')
        df_test_ex = df_test_ex_b.copy()

    show_score_proc_double(df_test_ex, df_test_ex_b,
                           df_train_ex, df_train_ex_b,
                           df_train_gt, 'savgol')

    return df_test_ex, df_train_ex

# --------------------------------------------------------------------------


def mean_offsets_module(df_test_ex, df_train_ex, df_train_gt, pairs_all, DLAT_INFO):

    df_test_ex_b = df_test_ex.copy()
    df_train_ex_b = df_train_ex.copy()

    df_test_ex = apply_offset_when_match(
        df_test_ex, pairs_all, DLAT_INFO)

#    df_train_ex = apply_offset_when_match(
#        df_train_ex, pairs_all, DLAT_INFO)

    df_train_ex = apply_offset_train(
        df_train_ex, pairs_all, DLAT_INFO)

    show_score_proc_double(df_test_ex, df_test_ex_b,
                           df_train_ex, df_train_ex_b,
                           df_train_gt, 'offset')

    return df_test_ex, df_train_ex

# =========================================================================
# visualize
# =========================================================================


def visualize_tile_init(n_rows, n_cols, collections):

    ix = list(range(len(collections)))
    s = [str(i) + ': ' + c for i, c in collections]
    subplot_titles = tuple(s)

    specs_list = [[dict(type="mapbox")] * n_cols] * n_rows

    fig = make_subplots(rows=n_rows, cols=n_cols,
                        horizontal_spacing=0.001,
                        vertical_spacing=0.008,
                        subplot_titles=subplot_titles,
                        specs=specs_list
                        )

    layout = go.Layout(hovermode='closest',
                       #                       title=dict(text=title_text,
                       #                                  font=dict(size=20)),
                       margin=dict(l=0, r=0, t=32, b=0),
                       width=1920, height=3200)

    layout = fig.layout.update(layout)

    # ---------------------------------------------
    data = []

    return data, layout


# --------------------------------------------------------------------------

def visualize_tile_append2(df_before, df_after, gt, data, layout, i):

    # ---------------------------------------------
    ma = 'mapbox'
    if i != 0:
        ma = 'mapbox' + str(i+1)

#    print(f'i {i} {ma}')
#    print(f'data {data}')
    # ---------------------------------------------
    zoom = 17
#    center = dict(lat=df_after.loc[0, 'latDeg'],
#                  lon=df_after.loc[0, 'lngDeg'])
    # center = dict(lat=df_after.loc[:, 'latDeg'].mean(),
    #               lon=df_after.loc[:, 'lngDeg'].mean())

    # center = dict(lat=df_after.loc[-1, 'latDeg'],
    #               lon=df_after.loc[-1, 'lngDeg'])

    center = dict(lat=df_after.iloc[-1]['latDeg'],
                  lon=df_after.iloc[-1]['lngDeg'])

    c = df_after['collectionName'].drop_duplicates().values[0]
    p = df_after['phoneName'].drop_duplicates().values[0]

    # ---------------------------------------------

    trace_before = go.Scattermapbox(
        lat=df_before['latDeg'],
        lon=df_before['lngDeg'],
        mode='markers',
        opacity=0.6,
        marker=dict(size=14, color='midnightblue'),
        #        text=df_before[['phoneName', 'millisSinceGpsEpoch']],
        text=df_before[['millisSinceGpsEpoch']],
        hoverlabel=dict(font=dict(size=20)),
        hoverinfo='text',
        subplot=ma)
    #        xaxis=xa, yaxis=ya)

    trace_gt = go.Scattermapbox(
        lat=gt['latDeg'],
        lon=gt['lngDeg'],
        mode='markers',
        opacity=0.6,
        marker=dict(size=14, color='teal'),
        #        text=df_before[['phoneName', 'millisSinceGpsEpoch']],
        text=df_before[['millisSinceGpsEpoch']],
        hoverlabel=dict(font=dict(size=20)),
        hoverinfo='text',
        subplot=ma)
    #        xaxis=xa, yaxis=ya)

#    print(df_after.columns)
    trace_after = go.Scattermapbox(
        lat=df_after['latDeg'],
        lon=df_after['lngDeg'],
        mode='lines',
        #        mode='markers',
        #        marker=dict(size=12, color='tomato'),
        #        opacity=0.8,
        #        text=df_after[['phoneName', 'millisSinceGpsEpoch']],
        text=df_after[['millisSinceGpsEpoch', 'in_motion']],
        hoverlabel=dict(font=dict(size=20)),
        hoverinfo='text',
        subplot=ma)
    #        xaxis=xa, yaxis=ya)

    # ---------------------------------------------

    layout[ma].update(
        dict(style='open-street-map',
             #        dict(style='stamen-terrain',
             center=center,
             zoom=zoom),
    )


#    print(layout)

    # ---------------------------------------------
    #    data = [trace_before, trace_after]
    data.append(trace_before)
    data.append(trace_gt)
    data.append(trace_after)

    # ---------------------------------------------

    return data, layout
# --------------------------------------------------------------------------


def visualize_tile_append(df_before, df_after, data, layout, i):

    # ---------------------------------------------
    ma = 'mapbox'
    if i != 0:
        ma = 'mapbox' + str(i+1)

#    print(f'i {i} {ma}')
#    print(f'data {data}')
    # ---------------------------------------------
    zoom = 17
#    center = dict(lat=df_after.loc[0, 'latDeg'],
#                  lon=df_after.loc[0, 'lngDeg'])
    # center = dict(lat=df_after.loc[:, 'latDeg'].mean(),
    #               lon=df_after.loc[:, 'lngDeg'].mean())

    # center = dict(lat=df_after.loc[-1, 'latDeg'],
    #               lon=df_after.loc[-1, 'lngDeg'])

    center = dict(lat=df_after.iloc[-1]['latDeg'],
                  lon=df_after.iloc[-1]['lngDeg'])

    c = df_after['collectionName'].drop_duplicates().values[0]
    p = df_after['phoneName'].drop_duplicates().values[0]

    # ---------------------------------------------

    trace_before = go.Scattermapbox(
        lat=df_before['latDeg'],
        lon=df_before['lngDeg'],
        mode='markers',
        opacity=0.6,
        marker=dict(size=14, color='midnightblue'),
        #        text=df_before[['phoneName', 'millisSinceGpsEpoch']],
        text=df_before[['millisSinceGpsEpoch']],
        hoverlabel=dict(font=dict(size=20)),
        hoverinfo='text',
        subplot=ma)
    #        xaxis=xa, yaxis=ya)

#    print(df_after.columns)
    trace_after = go.Scattermapbox(
        lat=df_after['latDeg'],
        lon=df_after['lngDeg'],
        mode='lines',
        #        mode='markers',
        #        marker=dict(size=12, color='tomato'),
        #        opacity=0.8,
        #        text=df_after[['phoneName', 'millisSinceGpsEpoch']],
        text=df_after[['millisSinceGpsEpoch', 'in_motion']],
        hoverlabel=dict(font=dict(size=20)),
        hoverinfo='text',
        subplot=ma)
    #        xaxis=xa, yaxis=ya)

    # ---------------------------------------------

    layout[ma].update(
        dict(style='open-street-map',
             #        dict(style='stamen-terrain',
             center=center,
             zoom=zoom),
    )


#    print(layout)

    # ---------------------------------------------
    #    data = [trace_before, trace_after]
    data.append(trace_before)
    data.append(trace_after)

    # ---------------------------------------------

    return data, layout

# --------------------------------------------------------------------------


def visualize_tile_init_one(n_rows, n_cols, title_list):

    subplot_titles = tuple(title_list)

    specs_list = [[dict(type="mapbox")] * n_cols] * n_rows

    fig = make_subplots(rows=n_rows, cols=n_cols,
                        horizontal_spacing=0.001,
                        vertical_spacing=0.008,
                        subplot_titles=subplot_titles,
                        specs=specs_list
                        )

    layout = go.Layout(hovermode='closest',
                       #                       title=dict(text=title_text,
                       #                                  font=dict(size=20)),
                       margin=dict(l=0, r=0, t=32, b=0),
                       width=1920, height=3200)

    layout = fig.layout.update(layout)

    # ---------------------------------------------
    data = []

    return data, layout

# --------------------------------------------------------------------------


def visualize_tile_append_one(df_before, data, layout, i):

    # ---------------------------------------------
    ma = 'mapbox'
    if i != 0:
        ma = 'mapbox' + str(i+1)

#    print(f'i {i} {ma}')
#    print(f'data {data}')
    # ---------------------------------------------
    zoom = 9
#    center = dict(lat=df_after.loc[0, 'latDeg'],
#                  lon=df_after.loc[0, 'lngDeg'])

    center = dict(lat=df_before.loc[:, 'latDeg'].mean(),
                  lon=df_before.loc[:, 'lngDeg'].mean())

    # center = dict(lat=df_after.loc[-1, 'latDeg'],
    #               lon=df_after.loc[-1, 'lngDeg'])

    center = dict(lat=df_before.iloc[-1]['latDeg'],
                  lon=df_before.iloc[-1]['lngDeg'])

    c = df_before['collectionName'].drop_duplicates().values[0]
    p = df_before['phoneName'].drop_duplicates().values[0]

    # ---------------------------------------------

    trace_before = go.Scattermapbox(
        lat=df_before['latDeg'],
        lon=df_before['lngDeg'],
        mode='markers',
        opacity=0.6,
        marker=dict(size=8, color='midnightblue'),
        #        text=df_before[['phoneName', 'millisSinceGpsEpoch']],
        text=df_before[['millisSinceGpsEpoch']],
        hoverlabel=dict(font=dict(size=20)),
        hoverinfo='text',
        subplot=ma)
#     #        xaxis=xa, yaxis=ya)

# #    print(df_after.columns)
#     trace_after = go.Scattermapbox(
#         lat=df_after['latDeg'],
#         lon=df_after['lngDeg'],
#         mode='lines',
#         #        mode='markers',
#         #        marker=dict(size=12, color='tomato'),
#         #        opacity=0.8,
#         #        text=df_after[['phoneName', 'millisSinceGpsEpoch']],
#         text=df_after[['millisSinceGpsEpoch', 'in_motion']],
#         hoverlabel=dict(font=dict(size=20)),
#         hoverinfo='text',
#         subplot=ma)
#     #        xaxis=xa, yaxis=ya)

    # ---------------------------------------------

    layout[ma].update(
        dict(style='open-street-map',
             #        dict(style='stamen-terrain',
             center=center,
             zoom=zoom),
    )


#    print(layout)

    # ---------------------------------------------
    #    data = [trace_before, trace_after]
    data.append(trace_before)
#    data.append(trace_after)

    # ---------------------------------------------

    return data, layout

# ---------------------------------------------


def visualize_tile_show(data, layout):

    fig = go.Figure(data=data, layout=layout)
    fig.show()

#    return data
# --------------------------------------------------------------------------


def visualize_before_after(df_before, df_after):

    zoom = 18
    center = dict(lat=df_after.loc[0, 'latDeg'],
                  lon=df_after.loc[0, 'lngDeg'])

    c = df_after['collectionName'].drop_duplicates().values[0]
    p = df_after['phoneName'].drop_duplicates().values[0]

    title_text = c + '  ' + p

    trace_before = go.Scattermapbox(
        lat=df_before['latDeg'],
        lon=df_before['lngDeg'],
        mode='markers',
        opacity=0.6,
        marker=dict(size=14, color='midnightblue'),
        text=df_before[['phoneName', 'millisSinceGpsEpoch']],
        hoverlabel=dict(font=dict(size=20)),
        hoverinfo='text')

    trace_after = go.Scattermapbox(
        lat=df_after['latDeg'],
        lon=df_after['lngDeg'],
        mode='lines',
        text=df_after[['phoneName', 'millisSinceGpsEpoch']],
        hoverlabel=dict(font=dict(size=20)),
        hoverinfo='text')

    data = [trace_before, trace_after]

    layout = go.Layout(hovermode='closest',
                       mapbox=dict(style='stamen-terrain',
                                   center=center,
                                   zoom=zoom),
                       title=dict(text=title_text,
                                  font=dict(size=20)),
                       width=2048, height=2048)

    fig = go.Figure(data=data, layout=layout)
    fig.show()


# =========================================================================
# phone mean
# =========================================================================
# post processing => all phones

def phone_mean_selected(df_test_bs, c, phone_selection):

    df_bs = df_test_bs.copy()
    df = df_test_bs.copy()
    df = df.loc[df['collectionName'] == c, :]

    phone_list = df['phoneName'].drop_duplicates().to_numpy()
    # -----------------------------------
    phone_data = {}
    phone_data_selected = {}
    corrections = {}
    #    print(phone_list)
    collection = [c]

    phone_list = [[p] for p in phone_list]
#    phone_selection = [[p] for p in phone_selection]
    #    print(f'phone_list {phone_list}')
    #    print(f'collection {collection}')

#    for phone in phone_list:
    # for phone in phone_selection:
    #     cond = np.logical_and(
    #         df['collectionName'] == collection[0], df['phoneName'] == phone[0]).to_list()

    #     phone_data_selected[phone[0]] = df[cond][[
    #         'millisSinceGpsEpoch', 'latDeg', 'lngDeg']].to_numpy()

    for phone in phone_list:
        #    for phone in phone_selection:
        cond = np.logical_and(
            df['collectionName'] == collection[0], df['phoneName'] == phone[0]).to_list()

        phone_data[phone[0]] = df[cond][[
            'millisSinceGpsEpoch', 'latDeg', 'lngDeg']].to_numpy()

#    print(f'cond {cond}')
#    print(f'phone_data {phone_data}')
#    print(f'phone_data {phone_data.keys()}')

#    print(f'\033[31misna 1{phone_data.isna().sum()}\033[0m')
    for current in phone_data:
        #        print(f'current {current}')
        #        pdb.set_trace()
        # just copy the shape
        if current in phone_selection:
            #            print(f'\033[36mcurrent phone is one of phone_selection\033[0m')
            correction = np.ones(phone_data[current].shape, dtype=np.float)
            correction[:, 1:] = phone_data[current][:, 1:]
        else:
            #            print(f'\033[31mcurrent phone is not one of phone_selection\033[0m')
            correction = np.zeros(phone_data[current].shape, dtype=np.float)
            correction[:, 1:] = phone_data[current][:, 1:]
            correction[:, 1:] = 0.0
#            print(correction.shape)

#        print(phone_selection)
#        print(f'current : {current}')
#        print(correction[0:3, 1:])

        # Telephones data don't complitely match by time, so - interpolate.
#        for other in phone_data_selected:
        for other in phone_selection:
            #            print(f'current other: \033[93m {current} {other}\033[0m')
            if other == current:
                continue

            # [:,0] is msge 't' [:,1] is lat
            # interpolation base - others
#            pdb.set_trace()
            # idx is for df
            loc = interp1d(phone_data[other][:, 0],
                           phone_data[other][:, 1:],
                           axis=0,
                           #                           kind='linear',
                           kind='quadratic',
                           copy=False,
                           bounds_error=None,
                           fill_value='extrapolate',
                           assume_sorted=True)

            start_idx = 0
            stop_idx = 0

#            pdb.set_trace()

            # just to find starting and ending index
            # [:,0] is msge : val is msge
            # for idx, val in enumerate(phone_data[current][:, 0]):
            #     if val < phone_data[other][0, 0]:
            #         start_idx = idx
            #     if val < phone_data[other][-1, 0]:
            #         stop_idx = idx + 1

#            print(f'phone_data[current].shape {phone_data[current].shape}')
#            print(phone_data[current][stop_idx-2:stop_idx+2, :])
            # print(
            #     f'current start_idx stop_idx : {current} {start_idx} {stop_idx}')

            # start_idx and stop_idx are for current
            # interpolation here
            # correction is a copy of phone_data[current]

#            if stop_idx - start_idx > 0:
            # count the number
#            print(f'\033[91m{other} interpolation added \033[0m')
#            correction[start_idx:stop_idx, 0] += 1
            correction[:, 0] += 1
            # add data interpolarted at the position of current

# ----------------------------------------
#                correction[start_idx:stop_idx,
#                           1:] += loc(phone_data[current][start_idx: stop_idx, 0])

# ----------------------------------------

            correction[:, 1:] += loc(phone_data[current][:, 0])

# ----------------------------------------


#            print(f'current / other : {current} / \033[1;92m{other}\033[0m')
#            print(f'\033[31m {correction[-3:, :]}\033[0m')
# #            print(f'\033[31misna 1{phone_data.isna().sum()}\033[0m')
#           print(correction[0: 3, 1:])
#        pdb.set_trace()

#        print(correction[correction[:, 0] == 0, :])
#        correction[correction[:, 0] == 0, 0] = 1
#        # lat mean
        correction[:, 1] /= correction[:, 0]
        # lng mean
        correction[:, 2] /= correction[:, 0]

#        pdb.set_trace()
        corrections[current] = correction.copy()

#        print(f'\033[31m {correction[-3:, :]}\033[0m')
#        print(f'------- current finished -------------------')
#        print(f'')

#    pdb.set_trace()

    for phone in phone_list:
        cond = np.logical_and(
            df['collectionName'] == collection[0], df['phoneName'] == phone[0]).to_list()

        df.loc[cond, ['latDeg', 'lngDeg']] = corrections[phone[0]][:, 1:]
#        print(df.loc[cond, ['latDeg', 'lngDeg']])

    df_bs.loc[df_bs['collectionName'] == c, :] = df
#    print(f'isna {df_bs.isna().sum()}')
#    pdb.set_trace()
    return df_bs

# =========================================================================


def phone_mean_post_processing(df_test_bs, c):

    #    ROOT = Path('../input/google-smartphone-decimeter-challenge/')
    #    TRAIN = ROOT/'train'
    #    TEST = ROOT/'test'
    #    TEST_BASELINE = ROOT / 'baseline_locations_test.csv'
    #    df_test_bs = pd.read_csv(TEST_BASELINE)

    df_bs = df_test_bs.copy()
    df = df_test_bs.copy()
    df = df.loc[df['collectionName'] == c, :]

    #    c = df['collectionName'].drop_duplicates().values[0]
    phone_list = df['phoneName'].drop_duplicates().to_numpy()
#    print(f'phone_list {phone_list}')

    #    phone_list = pairs_phones.loc[pairs_phones['collectionName']
    #                                  == c, 'phoneName'].to_numpy()

    # df_test_ex_list = []

    # for p_o in phone_list:
    #     #        p_o_path = TEST/c/p_o/(p_o + '_derived.csv')
    #     #        print(p_o_path.exists())
    #     df_test_routes_o = pd.read_csv(p_o_path)
    #     df_test_ex_o, idx = extract_baseline(df_test_bs, df_test_routes_o)
    #     df_test_ex_list.append(df_test_ex_o)

    #    df = pd.concat(df_test_ex_list, axis=0, ignore_index=True)
    #    return df_test_ex_list

    # -----------------------------------
    phone_data = {}
    corrections = {}
    #    print(phone_list)
    collection = [c]

    phone_list = [[p] for p in phone_list]
    #    print(f'phone_list {phone_list}')
    #    print(f'collection {collection}')

    for phone in phone_list:
        cond = np.logical_and(
            df['collectionName'] == collection[0], df['phoneName'] == phone[0]).to_list()
#        cond = np.logical_and(
#            df['collectionName'] == c, df['phoneName'] == phone[0]).to_list()

        phone_data[phone[0]] = df[cond][[
            'millisSinceGpsEpoch', 'latDeg', 'lngDeg']].to_numpy()

#    print(f'phone_data {phone_data}')
#    print(f'phone_data {phone_data.keys()}')

    for current in phone_data:
        #        print(f'current {current}')
        correction = np.ones(phone_data[current].shape, dtype=np.float)
        correction[:, 1:] = phone_data[current][:, 1:]

        # Telephones data don't complitely match by time, so - interpolate.
        for other in phone_data:
            if other == current:
                continue

            loc = interp1d(phone_data[other][:, 0],
                           phone_data[other][:, 1:],
                           axis=0,
                           kind='linear',
                           copy=False,
                           bounds_error=None,
                           fill_value='extrapolate',
                           assume_sorted=True)

            start_idx = 0
            stop_idx = 0
            for idx, val in enumerate(phone_data[current][:, 0]):
                if val < phone_data[other][0, 0]:
                    start_idx = idx
                if val < phone_data[other][-1, 0]:
                    stop_idx = idx

            if stop_idx - start_idx > 0:
                correction[start_idx:stop_idx, 0] += 1
                correction[start_idx:stop_idx,
                           1:] += loc(phone_data[current][start_idx: stop_idx, 0])

        correction[:, 1] /= correction[:, 0]
        correction[:, 2] /= correction[:, 0]

        corrections[current] = correction.copy()

    for phone in phone_list:
        cond = np.logical_and(
            df['collectionName'] == collection[0], df['phoneName'] == phone[0]).to_list()

        df.loc[cond, ['latDeg', 'lngDeg']] = corrections[phone[0]][:, 1:]

#    print(df.loc[0, 'latDeg'])

    df_bs.loc[df_bs['collectionName'] == c, :] = df

    return df_bs

# =========================================================================


def test_ex_mean_one(df_test_ex, df_test_bs, pairs_phones):

    #    ROOT = Path('../input/google-smartphone-decimeter-challenge/')
    #    TRAIN = ROOT/'train'
    #    TEST = ROOT/'test'
    #    TEST_BASELINE = ROOT / 'baseline_locations_test.csv'
    #    df_test_bs = pd.read_csv(TEST_BASELINE)

    df = df_test_ex.copy()
    c = df_test_ex['collectionName'].drop_duplicates().values[0]
    p = df_test_ex['phoneName'].drop_duplicates().values[0]

    phone_list = pairs_phones.loc[pairs_phones['collectionName']
                                  == c, 'phoneName'].to_numpy()

    # df_test_ex_list = []

    # for p_o in phone_list:
    #     #        p_o_path = TEST/c/p_o/(p_o + '_derived.csv')
    #     #        print(p_o_path.exists())
    #     df_test_routes_o = pd.read_csv(p_o_path)
    #     df_test_ex_o, idx = extract_baseline(df_test_bs, df_test_routes_o)
    #     df_test_ex_list.append(df_test_ex_o)

#    df = pd.concat(df_test_ex_list, axis=0, ignore_index=True)
#    return df_test_ex_list

    df = df_test_bs.loc[df_test_bs['collectionName'] == c, :]
    # -----------------------------------
    phone_data = {}
    corrections = {}
#    print(phone_list)
    collection = [c]

    phone_list = [[p] for p in phone_list]
#    print(f'phone_list {phone_list}')
#    print(f'collection {collection}')

    for phone in phone_list:
        cond = np.logical_and(
            df['collectionName'] == collection[0], df['phoneName'] == phone[0]).to_list()
#        cond = np.logical_and(
#            df['collectionName'] == c, df['phoneName'] == phone[0]).to_list()

        phone_data[phone[0]] = df[cond][[
            'millisSinceGpsEpoch', 'latDeg', 'lngDeg']].to_numpy()

#    print(f'phone_data {phone_data}')
#    print(f'phone_data {phone_data.keys()}')

    for current in phone_data:
        #        print(f'current {current}')
        correction = np.ones(phone_data[current].shape, dtype=np.float)
        correction[:, 1:] = phone_data[current][:, 1:]

        # Telephones data don't complitely match by time, so - interpolate.
        for other in phone_data:
            if other == current:
                continue

            loc = interp1d(phone_data[other][:, 0],
                           phone_data[other][:, 1:],
                           axis=0,
                           kind='linear',
                           copy=False,
                           bounds_error=None,
                           fill_value='extrapolate',
                           assume_sorted=True)

            start_idx = 0
            stop_idx = 0
            for idx, val in enumerate(phone_data[current][:, 0]):
                if val < phone_data[other][0, 0]:
                    start_idx = idx
                if val < phone_data[other][-1, 0]:
                    stop_idx = idx

            if stop_idx - start_idx > 0:
                correction[start_idx:stop_idx, 0] += 1
                correction[start_idx:stop_idx,
                           1:] += loc(phone_data[current][start_idx: stop_idx, 0])

        correction[:, 1] /= correction[:, 0]
        correction[:, 2] /= correction[:, 0]

        corrections[current] = correction.copy()

    for phone in phone_list:
        cond = np.logical_and(
            df['collectionName'] == collection[0], df['phoneName'] == phone[0]).to_list()

        df.loc[cond, ['latDeg', 'lngDeg']] = corrections[phone[0]][:, 1:]

#    print(df.loc[0, 'latDeg'])

    # if all_phones:
    #     df = df.reset_index(drop=True)

    # else:
    df = df.loc[df['phoneName'] == p, :].reset_index(drop=True)

    return df

# # =========================================================================


def mean_with_other_phones(df):
    collections_list = df[['collectionName']].drop_duplicates().to_numpy()
    print(f'collection_list {collections_list}')
    for collection in collections_list:
        phone_list = df[df['collectionName'].to_list() == collection][[
            'phoneName']].drop_duplicates().to_numpy()

        phone_data = {}
        corrections = {}

        print(collection)
        print(phone_list)
        for phone in phone_list:
            print(f'phone {phone}')
            cond = np.logical_and(
                df['collectionName'] == collection[0], df['phoneName'] == phone[0]).to_list()
            phone_data[phone[0]] = df[cond][[
                'millisSinceGpsEpoch', 'latDeg', 'lngDeg']].to_numpy()

        print(collection[0])
        print(phone_data)
        print(phone_data.keys())
        for current in phone_data:
            print(f'current {current}')
            correction = np.ones(phone_data[current].shape, dtype=np.float)
            correction[:, 1:] = phone_data[current][:, 1:]

            # Telephones data don't complitely match by time, so - interpolate.
            for other in phone_data:
                if other == current:
                    continue

                loc = interp1d(phone_data[other][:, 0],
                               phone_data[other][:, 1:],
                               axis=0,
                               kind='linear',
                               copy=False,
                               bounds_error=None,
                               fill_value='extrapolate',
                               assume_sorted=True)

                start_idx = 0
                stop_idx = 0
                for idx, val in enumerate(phone_data[current][:, 0]):
                    if val < phone_data[other][0, 0]:
                        start_idx = idx
                    if val < phone_data[other][-1, 0]:
                        stop_idx = idx

                if stop_idx - start_idx > 0:
                    correction[start_idx:stop_idx, 0] += 1
                    correction[start_idx:stop_idx,
                               1:] += loc(phone_data[current][start_idx: stop_idx, 0])

            correction[:, 1] /= correction[:, 0]
            correction[:, 2] /= correction[:, 0]

            corrections[current] = correction.copy()

        for phone in phone_list:
            cond = np.logical_and(
                df['collectionName'] == collection[0], df['phoneName'] == phone[0]).to_list()

            df.loc[cond, ['latDeg', 'lngDeg']] = corrections[phone[0]][:, 1:]

        break
    return df
# # =========================================================================
# position shift by wojtek
# =========================================================================


# ----------------------------------------------------------------------------


def objective(trial, b_train, g_train):
    a = trial.suggest_uniform('a', -1, 1)
    score, scores = compute_dist2(position_shift2(b_train, a), g_train)

    return score

# ----------------------------------------------------------------------------


def apply_position_shift(df_test_ex, df_train_ex, df_train_gt, sub_cols):

    b_train = df_train_ex[sub_cols]
    g_train = stich_to_phone(df_train_gt)
    g_train = g_train[sub_cols]

    study = optuna.create_study()
    study.optimize(lambda trial: objective(
        trial, b_train, g_train), n_trials=16)
    a_best = study.best_params['a']
    print(f'\033[36mBest offset:\033[0m {a_best:6.3f}')
    # ------------------------------------------------

    df = position_shift2(df_test_ex[sub_cols], a_best)

    df_test_ex['latDeg'] = df['latDeg']
    df_test_ex['lngDeg'] = df['lngDeg']

    return df_test_ex

# d: 'btrain.csv' => df_train_ex (extracted baselin)
# ----------------------------------------------------------------------------


def position_shift2(b_train, a):
    d = b_train.copy()

    msge = 'millisSinceGpsEpoch'
    d['heightAboveWgs84EllipsoidM'] = 63.5

    d['x'], d['y'], d['z'] = zip(
        *d.apply(lambda x: WGS84_to_ECEF(x.latDeg,
                                         x.lngDeg,
                                         x.heightAboveWgs84EllipsoidM),
                 axis=1))

    d.sort_values(['phone', msge], inplace=True)

    for fi in ['x', 'y', 'z']:
        d[fi+'p'] = d[fi].shift().where(d['phone'].eq(d['phone'].shift()))
        d[fi+'diff'] = d[fi]-d[fi+'p']

    d['dist'] = np.sqrt(d['xdiff']**2 + d['ydiff']**2 + d['zdiff']**2)
    for fi in ['x', 'y', 'z']:
        d[fi+'new'] = d[fi+'p'] + d[fi+'diff']*(1-a/d['dist'])

    lng, lat, alt = ECEF_to_WGS84(
        d['xnew'].values, d['ynew'].values, d['znew'].values)

    lng[np.isnan(lng)] = d.loc[np.isnan(lng), 'lngDeg']
    lat[np.isnan(lat)] = d.loc[np.isnan(lat), 'latDeg']

    d['lngDeg'] = lng
    d['latDeg'] = lat

    d.sort_values(['phone', msge], inplace=True)

    return d

# ----------------------------------------------------------------------------


def apply_position_shift3(df_test_ex, b_train, df_train_gt):

    g_train = df_train_gt.copy()

    study = optuna.create_study()

    study.optimize(lambda trial: objective3(
        trial, b_train, g_train), n_trials=64)

    a_best, b_best = study.best_params['a'], study.best_params['b']
    score = study.best_value

    # ------------------------------------------------

    df = position_shift3(df_test_ex, a_best, b_best)

    df_test_ex['latDeg'] = df['latDeg']
    df_test_ex['lngDeg'] = df['lngDeg']

    return df_test_ex

# ----------------------------------------------------------------------------


def objective3(trial, b_train, g_train):
    #    print(zip(b_train['latDeg'], g_train['latDeg']))
    a = trial.suggest_uniform('a', -0.0001, 0.0001)
    b = trial.suggest_uniform('b', -0.0001, 0.0001)

#    pdb.set_trace()

    score, _ = compute_dist3(position_shift3(b_train, a, b), g_train)
#    print(f'{a:8.5f}, {b:8.5f} {score:8.2f}')

    return score

# ----------------------------------------------------------------------------


def position_shift3(b_train, a, b):

    df = b_train.copy()
    df['latDeg'] += a
    df['lngDeg'] += b

    return df

# ----------------------------------------------------------------------------


def compute_dist3(oof, gt):

    #    ix_gt = []
    #    ix_ex = []
    dist = []

    for i_gt, latlng in gt[['latDeg', 'lngDeg']].iterrows():

        lat_gt = latlng['latDeg']
        lng_gt = latlng['lngDeg']

#        x0 = np.array([vincenty((lat_gt, lng_gt), (lat_ex, lng_ex)) for lat_ex,
#                       lng_ex in zip(oof['latDeg'], oof['lngDeg'])])

        n_rep = oof.shape[0]
        lat = np.repeat(lat_gt, n_rep)
        lng = np.repeat(lng_gt, n_rep)

        x0 = calc_haversine(lat, lng, oof['latDeg'], oof['lngDeg'])

#        i_x0 = x0.argmin()
#        lat_ex = oof.iloc[i_x0]['latDeg']
#        lng_ex = oof.iloc[i_x0]['lngDeg']

        dist.append(x0.min())

    dst_df = pd.DataFrame(dist, columns=['dist'])
#    pdb.set_trace()

    d50 = dst_df.quantile(.50).values[0]
    d95 = dst_df.quantile(.95).values[0]
#    print(type(d50))
    score = (d50 + d95) * 0.5

    return score, dst_df

# ----------------------------------------------------------------------------


def compute_dist4(oof, gt):

    # interpolate gt

    # f_interp = interp1d(gt['latDeg'],
    #                     gt['lngDeg'],
    #                     axis=0,
    #                     kind='linear',
    #                     copy=False,
    #                     bounds_error=None,
    #                     fill_value='extrapolate',
    #                     assume_sorted=True)
    f_interp = interp1d(gt['latDeg'],
                        gt['lngDeg'],
                        #                        kind='linear')
                        kind='linear',
                        fill_value='extrapolate')
    #     # copy=False,
    # bounds_error=None,
    # fill_value='extrapolate',
    # assume_sorted=True)

#    pdb.set_trace()
    lng_interp = f_interp(oof['latDeg'])
#    print(lng_interp)
#    pdb.set_trace()
    dst_oof = calc_haversine(oof['latDeg'], oof['lngDeg'],
                             oof['latDeg'], lng_interp)

    dst_df = pd.DataFrame(dst_oof, columns=['dist'])
#    pdb.set_trace()

    d50 = dst_df.quantile(.50).values[0]
    d95 = dst_df.quantile(.95).values[0]
#    print(type(d50))
    score = (d50 + d95) * 0.5

    return score, dst_df
#    return d50, dst_df

# ----------------------------------------------------------------------------


def extract_overlap(df_train_ex, gt, critical_distance=80.0):

    ix_gt = []
    ix_ex = []
    dist = []

    for i_gt, latlng in gt[['latDeg', 'lngDeg']].iterrows():

        lat_gt = latlng['latDeg']
        lng_gt = latlng['lngDeg']

        x0 = np.array([vincenty((lat_gt, lng_gt), (lat_ex, lng_ex)) for lat_ex,
                       lng_ex in zip(df_train_ex['latDeg'], df_train_ex['lngDeg'])]) * 1000.0

        i_x0 = x0.argmin()

#        lat_ex = df_train_ex.iloc[i_x0]['latDeg']
#        lng_ex = df_train_ex.iloc[i_x0]['lngDeg']

#        print(f'i_gt  ix_x0 {i_gt} {i_x0}')
#        print(f'{x0.max():12.3f} {x0.min():12.3f}')
        if x0.min() < critical_distance:

            ix_gt.append(i_gt)
            ix_ex.append(i_x0)
            dist.append(x0.min())

    ex = df_train_ex.iloc[ix_ex]
    ex.reset_index(drop=True, inplace=True)

    gt = gt.iloc[ix_gt]
    gt.reset_index(drop=True, inplace=True)

    return ex, gt

# =========================================================================


def compute_dist2(oof, gt):

    # when one gt is micro seconds and train is nano sec

    if gt.loc[0, 'millisSinceGpsEpoch'] < 1000000000000:
        oof['millisSinceGpsEpoch'] = oof['millisSinceGpsEpoch'] // 1000

    df = oof.merge(gt, on=['phone', 'millisSinceGpsEpoch'])

#    pdb.set_trace()

    dst_oof = calc_haversine(df.latDeg_x, df.lngDeg_x,
                             df.latDeg_y, df.lngDeg_y)
    scores = pd.DataFrame({'phone': df.phone, 'dst': dst_oof})
    scores_grp = scores.groupby('phone')
    d50 = scores_grp.quantile(.50).reset_index()
    d50.columns = ['phone', 'q50']
    d95 = scores_grp.quantile(.95).reset_index()
    d95.columns = ['phone', 'q95']

    score = (scores_grp.quantile(.50).mean() +
             scores_grp.quantile(.95).mean())/2
    scores_50_95 = d50.merge(d95)
    return score, scores_50_95


# =========================================================================
# detect stationary state by accelertation
# =========================================================================

def is_stationary2(df_test_ex, acce_test, d_crit=0.005):

    x_diff = acce_test['x_f'].diff(1)
    x_diff.iloc[0] = 0.0
    x_diff.iloc[-1] = 0.0

    y_diff = acce_test['y_f'].diff(1)
    y_diff.iloc[0] = 0.0
    y_diff.iloc[-1] = 0.0

    idx = x_diff[(abs(x_diff) < d_crit) & (abs(y_diff) < d_crit)].index.values

    df_test_ex['in_motion'] = 'm'
    df_test_ex.loc[idx, 'in_motion'] = 's'
    return df_test_ex


# =========================================================================
# visualize
# =========================================================================
def visualize_stationary2(df_test_ex, acce_test):

    x = acce_test['latDeg'].values
    y = acce_test['lngDeg'].values
    x = (x - x.mean())/x.std()
    y = (y - y.mean())/y.std()

    ts = acce_test.loc[df_test_ex['in_motion'] == 's', msge].values
    xs = acce_test.loc[df_test_ex['in_motion'] == 's', 'latDeg'].values
    ys = acce_test.loc[df_test_ex['in_motion'] == 's', 'lngDeg'].values

    xs = (xs - xs.mean())/xs.std()
    ys = (ys - ys.mean())/ys.std()

    trace0 = go.Scatter(x=acce_test[msge], y=acce_test['x_f'],
                        mode='lines',
                        #                    mode='markers'
                        #                    marker=dict(size=16, color='midnightblue'),
                        text=acce_test.loc[:, [msge, 'latDeg', 'lngDeg']],
                        hoverlabel=dict(font=dict(size=20)),
                        hoverinfo='text')

    trace1 = go.Scatter(x=acce_test[msge], y=acce_test['y_f'],
                        mode='lines',
                        #                    mode='markers'
                        #                    marker=dict(size=16, color='midnightblue'),
                        text=acce_test.loc[:, [msge, 'latDeg', 'lngDeg']],
                        hoverlabel=dict(font=dict(size=20)),
                        hoverinfo='text')

    trace2 = go.Scatter(x=acce_test[msge], y=x)
    trace3 = go.Scatter(x=acce_test[msge], y=y)

    trace4 = go.Scatter(x=ts, y=xs, mode='markers')
    trace5 = go.Scatter(x=ts, y=ys, mode='markers')

    data = [trace0, trace1, trace2, trace3, trace4, trace5]

    layout = go.Layout(hovermode='closest',
                       width=2048, height=2048)

    fig = go.Figure(data=data, layout=layout)
    fig.show()

# =========================================================================
# scratch end
# =========================================================================


def apply_savgol(df_test_ex, n_window=9, n_order=3):

    t = df_test_ex['millisSinceGpsEpoch'].values
    y = df_test_ex['latDeg'].values
    x = df_test_ex['lngDeg'].values

    yhat = savgol_filter(y, 9, 3)
    xhat = savgol_filter(x, 9, 3)

    df_test_ex['latDeg'] = yhat
    df_test_ex['lngDeg'] = xhat

    return df_test_ex

# =========================================================================
# scratch end
# =========================================================================


def apply_spline(df_test_ex, smooth_factor=1e-7):
    t = df_test_ex['millisSinceGpsEpoch'].values
    y = df_test_ex['latDeg'].values
    x = df_test_ex['lngDeg'].values

#    print(np.isnan(x).sum())
#    print(np.isnan(y).sum())

    sply = UnivariateSpline(t, y)
    sply.set_smoothing_factor(smooth_factor)
    yy = sply(t)

    splx = UnivariateSpline(t, x)
    splx.set_smoothing_factor(smooth_factor)
    xx = splx(t)

    df_test_ex['latDeg'] = yy
    df_test_ex['lngDeg'] = xx

    return df_test_ex

# =========================================================================
# scratch end
# =========================================================================


def get_pairs_phones(pairs_all):
    elias_kaggle = Path('/Users/meg/ka/k6/')
    ROOT = Path('../input/google-smartphone-decimeter-challenge/')
    TRAIN = ROOT/'train'
    TEST = ROOT/'test'
    path_test_routes = sorted(list(TEST.glob('*/*/*_derived.csv')))

    c_list = []
    p_list = []

    for p in path_test_routes:
        p_parent = PurePath(p).parent
        phone_name = Path(p_parent).stem
        collection_name = Path(p_parent.parent).stem
        c_list.append(collection_name)
        p_list.append(phone_name)

    pairs_phones = dict(collectionName=c_list, phoneName=p_list)
    pairs_phones = pd.DataFrame(pairs_phones)
    pairs_phones = pairs_phones.sort_values('collectionName')
    pairs_phones.reset_index(drop=True, inplace=True)
    pairs_phones = pairs_phones.merge(
        pairs_all, left_on='collectionName', right_on='test')

    dir_train = pairs_all['train'].unique()

    # 3:4 : 4 samsung full
    c_list = []
    p_list = []
    for d in dir_train:
        p_train = [p.stem for p in list((TRAIN/d).glob('*'))]
        c_list.append(d)
        p_list.append(p_train)

    train_pairs_phones = dict(train=c_list, train_phoneName=p_list)
    train_pairs_phones = pd.DataFrame(train_pairs_phones)

    pairs_phones = pairs_phones.merge(train_pairs_phones, on='train')
#    print(pairs_phones)
#    elias_kaggle = Path('/Users/meg/ka/k6/')
    pairs_phones.to_csv(elias_kaggle/'log'/'pairs_phones.csv', index=False)

    return pairs_phones


# ===========================================================================#
# simple_follium
# ===========================================================================


def simple_folium(df: pd.DataFrame, lat_col: str, lon_col: str, text_cols: list, map_name: str):
    """
    Descrption
    ----------
        Returns a simple Folium HeatMap with Markers
    ----------
    Parameters
    ----------
        df : padnas DataFrame, required
            The DataFrane with the data to map
        lat_col : str, required
            The name of the column with latitude
        lon_col : str, required
            The name of the column with longitude
        test_cols: list, optional
            A list with the names of the columns to print for each marker

    """
    # Preprocess
    # Drop rows that do not have lat/lon
    df = df[df[lat_col].notnull() & df[lon_col].notnull()]

    # Convert lat/lon to (n, 2) nd-array format for heatmap
    # Then send to list
    df_locs = list(df[[lat_col, lon_col]].values)

    # Add the location name to the markers
    text_cols = ["location_name", "location_contact_url_main",
                 "location_address_street"]
    text_feature_list = list(zip(*[df[col] for col in text_cols]))
    text_formated = []
    for text in text_feature_list:
        text = [str(feat) for feat in text]
        text_formated.append("<br>".join(text))
    marker_info = text_formated

    # Set up folium map
    fol_map = folium.Map([41.8781, -87.6298], zoom_start=4)

    # plot heatmap
    heat_map = plugins.HeatMap(df_locs, name=map_name)
    fol_map.add_child(heat_map)

    # plot markers
    markers = plugins.MarkerCluster(
        locations=df_locs, popups=marker_info, name="Testing Site")
    fol_map.add_child(markers)

    # Add Layer Control
    folium.LayerControl().add_to(fol_map)

    return fol_map

# simple_folium(CAC_df, "lat", "lng", ["location_name","location_address_street"], "COVID Testing Sites")
# ===========================================================================
# adaptive gaussian smoothing
# ===========================================================================


def apply_gauss_smoothing(df, params):
    SZ_1 = params['sz_1']
    SZ_2 = params['sz_2']
    SZ_CRIT = params['sz_crit']

    unique_paths = df[['collectionName', 'phoneName']
                      ].drop_duplicates().to_numpy()
    for collection, phone in unique_paths:
        cond = np.logical_and(df['collectionName'] ==
                              collection, df['phoneName'] == phone)
        data = df[cond][['latDeg', 'lngDeg']].to_numpy()

        lat_g1 = gaussian_filter1d(data[:, 0], np.sqrt(SZ_1))
        lon_g1 = gaussian_filter1d(data[:, 1], np.sqrt(SZ_1))
        lat_g2 = gaussian_filter1d(data[:, 0], np.sqrt(SZ_2))
        lon_g2 = gaussian_filter1d(data[:, 1], np.sqrt(SZ_2))

        lat_dif = data[1:, 0] - data[:-1, 0]
        lon_dif = data[1:, 1] - data[:-1, 1]

        lat_crit = np.append(np.abs(gaussian_filter1d(lat_dif, np.sqrt(
            SZ_CRIT)) / (1e-9 + gaussian_filter1d(np.abs(lat_dif), np.sqrt(SZ_CRIT)))), [0])
        lon_crit = np.append(np.abs(gaussian_filter1d(lon_dif, np.sqrt(
            SZ_CRIT)) / (1e-9 + gaussian_filter1d(np.abs(lon_dif), np.sqrt(SZ_CRIT)))), [0])

        df.loc[cond, 'latDeg'] = lat_g1 * lat_crit + lat_g2 * (1.0 - lat_crit)
        df.loc[cond, 'lngDeg'] = lon_g1 * lon_crit + lon_g2 * (1.0 - lon_crit)

    return df


# ===========================================================================
# mean phone offset
# ===========================================================================


def mean_with_other_phones(df):
    collections_list = df[['collectionName']].drop_duplicates().to_numpy()

    for collection in collections_list:
        phone_list = df[df['collectionName'].to_list() == collection][[
            'phoneName']].drop_duplicates().to_numpy()

        phone_data = {}
        corrections = {}
        for phone in phone_list:
            cond = np.logical_and(
                df['collectionName'] == collection[0], df['phoneName'] == phone[0]).to_list()
            phone_data[phone[0]] = df[cond][[
                'millisSinceGpsEpoch', 'latDeg', 'lngDeg']].to_numpy()

        for current in phone_data:
            correction = np.ones(phone_data[current].shape, dtype=np.float)
            correction[:, 1:] = phone_data[current][:, 1:]

            # Telephones data don't complitely match by time, so - interpolate.
            for other in phone_data:
                if other == current:
                    continue

                loc = interp1d(phone_data[other][:, 0],
                               phone_data[other][:, 1:],
                               axis=0,
                               kind='linear',
                               copy=False,
                               bounds_error=None,
                               fill_value='extrapolate',
                               assume_sorted=True)

                start_idx = 0
                stop_idx = 0
                for idx, val in enumerate(phone_data[current][:, 0]):
                    if val < phone_data[other][0, 0]:
                        start_idx = idx
                    if val < phone_data[other][-1, 0]:
                        stop_idx = idx

                if stop_idx - start_idx > 0:
                    correction[start_idx:stop_idx, 0] += 1
                    correction[start_idx:stop_idx,
                               1:] += loc(phone_data[current][start_idx:stop_idx, 0])

            correction[:, 1] /= correction[:, 0]
            correction[:, 2] /= correction[:, 0]

            corrections[current] = correction.copy()

        for phone in phone_list:
            cond = np.logical_and(
                df['collectionName'] == collection[0], df['phoneName'] == phone[0]).to_list()

            df.loc[cond, ['latDeg', 'lngDeg']] = corrections[phone[0]][:, 1:]

    return df

# ===========================================================================
# visualize
# ===========================================================================


def visualize_original_before_after(df_original, df_before, df_after):
    #    df = df_routes_base_true
    #    df['latDeg']
    zoom = 15
    center = dict(lat=df_after.loc[0, 'latDeg'],
                  lon=df_after.loc[0, 'lngDeg'])

    c = df_after['collectionName'].drop_duplicates().values[0]
    p = df_after['phoneName'].drop_duplicates().values[0]

    title_text = c + '  ' + p

    trace_original = go.Scattermapbox(
        lat=df_original['latDeg'],
        lon=df_original['lngDeg'],
        mode='markers',
        marker=dict(size=16, color='midnightblue'),
        text=df_before[['phoneName', 'millisSinceGpsEpoch']],
        hoverinfo='text'
    )

    trace_before = go.Scattermapbox(
        lat=df_before['latDeg'],
        lon=df_before['lngDeg'],
        mode='markers',
        marker=dict(size=12, color='royalblue'),
        text=df_before[['phoneName', 'millisSinceGpsEpoch']],
        hoverinfo='text'
    )

    trace_after = go.Scattermapbox(
        lat=df_after['latDeg'],
        lon=df_after['lngDeg'],
        mode='markers',
        opacity=0.8,
        marker=dict(size=12, color='tomato'),
        text=df_after[['phoneName', 'millisSinceGpsEpoch']],
        hoverinfo='text'
    )

    data = [trace_original, trace_before, trace_after]

    layout = go.Layout(hovermode='closest',
                       mapbox=dict(style='stamen-terrain',
                                   center=center,
                                   zoom=zoom),
                       title=dict(text=title_text,
                                  font=dict(size=20)),
                       width=2048, height=2048)

    fig = go.Figure(data=data, layout=layout)
    fig.show()

# ===========================================================================


def visualize_stationary(df_test_ex):

    zoom = 15
    center = dict(lat=df_test_ex.loc[0, 'latDeg'],
                  lon=df_test_ex.loc[0, 'lngDeg'])

    # lat=df_test_ex.loc[df_test_ex['in_motion'] == 's', ['latDeg']]
    # lon=df_test_ex.loc[df_test_ex['in_motion'] == 's', ['lngDeg']]

    c = df_test_ex['collectionName'].drop_duplicates().values[0]
    p = df_test_ex['phoneName'].drop_duplicates().values[0]

    title_text = c + '  ' + p

    trace1 = go.Scattermapbox(
        lat=df_test_ex['latDeg'],
        lon=df_test_ex['lngDeg'],
        mode='markers',
        marker=dict(size=18, color='midnightblue'),
        text=df_test_ex[['phoneName', 'millisSinceGpsEpoch']],
        hoverlabel=dict(font=dict(size=20)),
        hoverinfo='text')

    trace2 = go.Scattermapbox(
        lat=df_test_ex.loc[df_test_ex['in_motion'] == 's', 'latDeg'],
        lon=df_test_ex.loc[df_test_ex['in_motion'] == 's', 'lngDeg'],
        #        lat = lat, lon =lon,
        #        mode='lines',
        mode='markers',
        marker=dict(size=12, color='tomato'),
        opacity=0.8,
        text=df_test_ex.loc[df_test_ex['in_motion'] ==
                            's', ['phoneName', 'millisSinceGpsEpoch']],
        hoverlabel=dict(font=dict(size=20)),
        hoverinfo='text')

    data = [trace1, trace2]
#    data = [trace1]

    layout = go.Layout(hovermode='closest',
                       mapbox=dict(style='stamen-terrain',
                                   center=center,
                                   zoom=zoom),
                       title=dict(text=title_text,
                                  font=dict(size=20)),
                       width=2048, height=2048)

    fig = go.Figure(data=data, layout=layout)
    fig.show()

# visualize_stationary(df_test_ex)
# lat=df_test_ex.loc[df_test_ex['in_motion'] == 's', ['latDeg']]
# lon=df_test_ex.loc[df_test_ex['in_motion'] == 's', ['lngDeg']]
# lat.shape, lon.shape
# visualize_stationary(df_test_ex)

# ===========================================================================


def visualize_before_after(df_before, df_after):
    #    df = df_routes_base_true
    #    df['latDeg']
    zoom = 15
    center = dict(lat=df_after.loc[0, 'latDeg'],
                  lon=df_after.loc[0, 'lngDeg'])

    c = df_after['collectionName'].drop_duplicates().values[0]
    p = df_after['phoneName'].drop_duplicates().values[0]

    title_text = c + '  ' + p

    trace_before = go.Scattermapbox(
        lat=df_before['latDeg'],
        lon=df_before['lngDeg'],
        mode='markers',
        opacity=0.6,
        marker=dict(size=14, color='midnightblue'),
        text=df_before[['phoneName', 'millisSinceGpsEpoch']],
        hoverlabel=dict(font=dict(size=20)),
        hoverinfo='text')

    trace_after = go.Scattermapbox(
        lat=df_after['latDeg'],
        lon=df_after['lngDeg'],
        mode='lines',
        #        mode='markers',
        #        marker=dict(size=12, color='tomato'),
        #        opacity=0.8,
        text=df_after[['phoneName', 'millisSinceGpsEpoch']],
        hoverlabel=dict(font=dict(size=20)),
        hoverinfo='text')

    data = [trace_before, trace_after]

    layout = go.Layout(hovermode='closest',
                       mapbox=dict(style='stamen-terrain',
                                   center=center,
                                   zoom=zoom),
                       title=dict(text=title_text,
                                  font=dict(size=20)),
                       width=2048, height=2048)

    fig = go.Figure(data=data, layout=layout)
    fig.show()

# =========================================================================
# show score
# =========================================================================


def return_scores(df_test_ex, df_test_ex_org, df_train_gt):
    dist_snap = 48
    df_fake_gt = create_fake_gt(df_test_ex, df_train_gt, dist_snap)
    df_fake_gt_org = create_fake_gt(df_test_ex_org, df_train_gt, dist_snap)
    s_before = score_fake(df_test_ex_org, df_fake_gt_org)
    s_after = score_fake(df_test_ex, df_fake_gt)
    # print(
    #     f'\033[32mBEFORE : {s_before[0]:6.3f} {s_before[1]:6.3f} {s_before[2]:6.3f}\033[0m')
    # print(
    #     f'\033[36mAFTER  : {s_after[0]:6.3f} {s_after[1]:6.3f} {s_after[2]:6.3f}\033[0m')

    return s_before, s_after

# =========================================================================


def return_scores_train(df_train_ex, df_train_ex_org, df_train_gt):
    #    dist_snap = 48
    #    df_fake_gt = create_fake_gt(df_test_ex, df_train_gt, dist_snap)
    #    df_fake_gt_org = create_fake_gt(df_test_ex_org, df_train_gt, dist_snap)
    s_before = score_fake(df_train_ex_org, df_train_gt)
    s_after = score_fake(df_train_ex, df_train_gt)
    # print(
    #     f'\033[32mBEFORE : {s_before[0]:6.3f} {s_before[1]:6.3f} {s_before[2]:6.3f}\033[0m')
    # print(
    #     f'\033[36mAFTER  : {s_after[0]:6.3f} {s_after[1]:6.3f} {s_after[2]:6.3f}\033[0m')

    return s_before, s_after

# =========================================================================


def show_score_proc_double(df_test_ex, df_test_ex_org,
                           df_train_ex, df_train_ex_org, df_train_gt, proc):

    dist_snap = 48

    df_fake_gt = create_fake_gt(df_test_ex, df_train_gt, dist_snap)
    df_fake_gt_org = create_fake_gt(df_test_ex_org, df_train_gt)

    sb = score_fake(df_test_ex_org, df_fake_gt_org)
    sa = score_fake(df_test_ex, df_fake_gt)

    sb_tr = score_fake(df_train_ex_org, df_train_gt)
    sa_tr = score_fake(df_train_ex, df_train_gt)

#    print(f'\033[32mBEFORE : {sb[0]:6.3f} {sb[1]:6.3f} {sb[2]:6.3f}\033[0m')
#    print(f'\033[36mAFTER  : {sa[0]:6.3f} {sa[1]:6.3f} {sa[2]:6.3f}\033[0m')
#    print(f'\033[32mBEFORE : {sb[0]:6.3f} {sb[1]:6.3f} {sb[2]:6.3f}\033[0m')
    diff = sb[0] - sa[0]
    diff_tr = sb_tr[0] - sa_tr[0]
#    print(        f'\033[33m{proc:6s} \033[1;96m{sa[0]:6.3f} {sa[1]:6.3f} {sa[2]:6.3f} {diff:6.3f}\033[0m')
    print(
        f'\033[33m{proc:6s} \033[1;96m{sa[0]:6.3f} {sa[1]:6.3f} {sa[2]:6.3f} {diff:6.3f} \033[0m', end='')

    print(
        f'\033[1;92m{sa_tr[0]:6.3f} {sa_tr[1]:6.3f} {sa_tr[2]:6.3f} {diff_tr:6.3f}\033[0m')
#    return sa, sb
#
# =========================================================================


def show_score_proc(df_test_ex, df_test_ex_org, df_train_gt, proc):
    dist_snap = 48
    df_fake_gt = create_fake_gt(df_test_ex, df_train_gt, dist_snap)
    df_fake_gt_org = create_fake_gt(df_test_ex_org, df_train_gt)
    sb = score_fake(df_test_ex_org, df_fake_gt_org)
    sa = score_fake(df_test_ex, df_fake_gt)

#    print(f'\033[32mBEFORE : {sb[0]:6.3f} {sb[1]:6.3f} {sb[2]:6.3f}\033[0m')
#    print(f'\033[36mAFTER  : {sa[0]:6.3f} {sa[1]:6.3f} {sa[2]:6.3f}\033[0m')
#    print(f'\033[32mBEFORE : {sb[0]:6.3f} {sb[1]:6.3f} {sb[2]:6.3f}\033[0m')
    diff = sb[0] - sa[0]
#    print(        f'\033[33m{proc:6s} \033[1;96m{sa[0]:6.3f} {sa[1]:6.3f} {sa[2]:6.3f} {diff:6.3f}\033[0m')
    print(
        f'\033[33m{proc:6s} \033[1;96m{sa[0]:6.3f} {sa[1]:6.3f} {sa[2]:6.3f} {diff:6.3f}\033[0m')
#    return sa, sb
# =========================================================================


def show_scores(df_test_ex, df_test_ex_org, df_train_gt):
    dist_snap = 48
    df_fake_gt = create_fake_gt(df_test_ex, df_train_gt, dist_snap)
    df_fake_gt_org = create_fake_gt(df_test_ex_org, df_train_gt)
    sb = score_fake(df_test_ex_org, df_fake_gt_org)
    sa = score_fake(df_test_ex, df_fake_gt)

    print(f'\033[32mBEFORE : {sb[0]:6.3f} {sb[1]:6.3f} {sb[2]:6.3f}\033[0m')
    print(f'\033[36mAFTER  : {sa[0]:6.3f} {sa[1]:6.3f} {sa[2]:6.3f}\033[0m')

#    return sa, sb

# =========================================================================
# get mathinng train_gt
# =========================================================================


def score_fake(df_test_ex, df_fake_gt):
    #    print(df_test_ex.head(3))
    #    print(df_fake_gt.head(3))

    pred = list(zip(df_test_ex['latDeg'], df_test_ex['lngDeg']))
    true = list(zip(df_fake_gt['latDeg'], df_fake_gt['lngDeg']))

    x0 = pd.DataFrame([vincenty(p1, p2) for p1, p2 in zip(pred, true)])
    x0 = x0 * 1000.0

#    pdb.set_trace()
    x5 = x0.quantile(.50).values[0]
    x9 = x0.quantile(.95).values[0]
    xm = (x5 + x9) * 0.5

    return xm, x5, x9

# =========================================================================
# get mathinng train_gt
# =========================================================================


def get_matching_gt(df_test_ex, pairs_all):

    ROOT = Path('../Input/Google-Smartphone-Decimeter-Challenge/')
    TRAIN = ROOT/'Train'

    c = df_test_ex['collectionName'].drop_duplicates().values[0]
    p = df_test_ex['phoneName'].drop_duplicates().values[0]

    c_t = pairs_all.loc[pairs_all['test'] == c, 'train'].values[0]
    train_phone_names = [p_t.stem for p_t in sorted(
        list((TRAIN/c_t).glob('*')))]

    if p not in train_phone_names:
        p_t = train_phone_names[0]
    else:
        p_t = p

    df_train_gt = pd.read_csv(TRAIN/c_t/p_t/'ground_truth.csv')

    return df_train_gt
# =========================================================================
# get mathinng train_bs
# =========================================================================


def get_matching_bs(df_test_ex, pairs_all):

    ROOT = Path('../Input/Google-Smartphone-Decimeter-Challenge/')
    TRAIN = ROOT/'Train'
    TRAIN_BASELINE = ROOT / 'baseline_locations_train.csv'
    df_train_baseline = pd.read_csv(TRAIN_BASELINE)

    c = df_test_ex['collectionName'].drop_duplicates().values[0]
    p = df_test_ex['phoneName'].drop_duplicates().values[0]

    c_t = pairs_all.loc[pairs_all['test'] == c, 'train'].values[0]
    train_phone_names = [p_t.stem for p_t in sorted(
        list((TRAIN/c_t).glob('*')))]

    if p not in train_phone_names:
        p_t = train_phone_names[0]
    else:
        p_t = p

    df_train_routes = pd.read_csv(TRAIN/c_t/p_t/(p_t+'_derived.csv'))
    df_train_ex, _ = extract_baseline(df_train_baseline, df_train_routes)

    return df_train_ex


# =========================================================================
# get mathinng train_routes
# =========================================================================


def get_matching_routes(df_test_ex, pairs_all):

    ROOT = Path('../Input/Google-Smartphone-Decimeter-Challenge/')
    TRAIN = ROOT/'Train'

    c = df_test_ex['collectionName'].drop_duplicates().values[0]
    p = df_test_ex['phoneName'].drop_duplicates().values[0]

    c_t = pairs_all.loc[pairs_all['test'] == c, 'train'].values[0]
    train_phone_names = [p_t.stem for p_t in sorted(
        list((TRAIN/c_t).glob('*')))]

    if p not in train_phone_names:
        p_t = train_phone_names[0]
    else:
        p_t = p

    df_train_routes = pd.read_csv(TRAIN/c_t/p_t/(p_t+'_derived.csv'))

    return df_train_routes


# =========================================================================
# phone mean
# =========================================================================


def mean_with_other_phones(df):
    collections_list = df[['collectionName']].drop_duplicates().to_numpy()

    for collection in collections_list:
        phone_list = df[df['collectionName'].to_list() == collection][[
            'phoneName']].drop_duplicates().to_numpy()

        phone_data = {}
        corrections = {}
        for phone in phone_list:
            cond = np.logical_and(
                df['collectionName'] == collection[0], df['phoneName'] == phone[0]).to_list()
            phone_data[phone[0]] = df[cond][[
                'millisSinceGpsEpoch', 'latDeg', 'lngDeg']].to_numpy()

        for current in phone_data:
            correction = np.ones(phone_data[current].shape, dtype=np.float)
            correction[:, 1:] = phone_data[current][:, 1:]

            # Telephones data don't complitely match by time, so - interpolate.
            for other in phone_data:
                if other == current:
                    continue

                loc = interp1d(phone_data[other][:, 0],
                               phone_data[other][:, 1:],
                               axis=0,
                               kind='linear',
                               copy=False,
                               bounds_error=None,
                               fill_value='extrapolate',
                               assume_sorted=True)

                start_idx = 0
                stop_idx = 0
                for idx, val in enumerate(phone_data[current][:, 0]):
                    if val < phone_data[other][0, 0]:
                        start_idx = idx
                    if val < phone_data[other][-1, 0]:
                        stop_idx = idx

                if stop_idx - start_idx > 0:
                    correction[start_idx:stop_idx, 0] += 1
                    correction[start_idx:stop_idx,
                               1:] += loc(phone_data[current][start_idx:stop_idx, 0])

            correction[:, 1] /= correction[:, 0]
            correction[:, 2] /= correction[:, 0]

            corrections[current] = correction.copy()

        for phone in phone_list:
            cond = np.logical_and(
                df['collectionName'] == collection[0], df['phoneName'] == phone[0]).to_list()

            df.loc[cond, ['latDeg', 'lngDeg']] = corrections[phone[0]][:, 1:]

    return df

# =========================================================================
# snap to nearest gt
# =========================================================================


def create_fake_gt(df_test_ex, df_train_gt, dist_snap=48):

    df_fake_gt = df_test_ex.copy()

    for idx_test, d_test in df_test_ex.iterrows():

        latx = d_test['latDeg']
        lngx = d_test['lngDeg']

        n_rep = df_train_gt.shape[0]

        lat = np.repeat(latx, n_rep)
        lng = np.repeat(lngx, n_rep)

        dist = calc_haversine(
            lat, lng, df_train_gt['latDeg'].values, df_train_gt['lngDeg'].values)

        idx = dist.argmin()

        if dist[idx] < dist_snap:

            df_fake_gt.loc[idx_test, 'latDeg'] = df_train_gt.loc[idx, 'latDeg']
            df_fake_gt.loc[idx_test, 'lngDeg'] = df_train_gt.loc[idx, 'lngDeg']

    return df_fake_gt


# =========================================================================
# correct off the course
# =========================================================================

def snap_course_off(df_test_ex, df_train_gt, off_crit=12.5):

    for idx_test, d_test in df_test_ex.iterrows():
        #        d_test =   d_test.T
        #        print(idx_test, d_test)
        #        print(df_train_gt.head(3))

        latx = d_test['latDeg']
        lngx = d_test['lngDeg']

        n_rep = df_train_gt.shape[0]

        lat = np.repeat(latx, n_rep)
        lng = np.repeat(lngx, n_rep)

        dist = calc_haversine(
            lat, lng,
            df_train_gt['latDeg'].values, df_train_gt['lngDeg'].values)

        idx = dist.argmin()
#        print(idx_test, dist[idx])

        if dist[idx] > off_crit:

            #        print(df_test_ex.loc[idx_test, 'latDeg'], df_train_gt.loc[idx, 'latDeg'])

            df_test_ex.loc[idx_test, 'latDeg'] = df_train_gt.loc[idx, 'latDeg']
            df_test_ex.loc[idx_test, 'lngDeg'] = df_train_gt.loc[idx, 'lngDeg']

    return df_test_ex

# =========================================================================
# snap to nearest gt
# =========================================================================


def snap_to_nearest_gt(df_test_bs, pair):

    df_test_ex, df_train_ex, df_train_gt, df_test_routes, df_train_routes, idx_ex = read_pair(
        df_test_bs, pair, first_phone_only=True)

    df_test_ex_before = df_test_ex.copy()
    # def find_nearest_gt(df_test_ex):
    dist_snap = 48
    # df_test_ex['latDeg_nearest'] = df_test_ex['latDeg']
    # df_test_ex['lngDeg_nearest'] = df_test_ex['lngDeg']

    for idx_test, d_test in df_test_ex.iterrows():

        latx = d_test['latDeg']
        lngx = d_test['lngDeg']

        n_rep = df_train_gt.shape[0]

        lat = np.repeat(latx, n_rep)
        lng = np.repeat(lngx, n_rep)

        dist = calc_haversine(
            lat, lng, df_train_gt['latDeg'].values, df_train_gt['lngDeg'].values)

        idx = dist.argmin()
#        print(idx_test, dist[idx])

        if dist[idx] < dist_snap:

            #        print(df_test_ex.loc[idx_test, 'latDeg'], df_train_gt.loc[idx, 'latDeg'])

            df_test_ex.loc[idx_test, 'latDeg'] = df_train_gt.loc[idx, 'latDeg']
            df_test_ex.loc[idx_test, 'lngDeg'] = df_train_gt.loc[idx, 'lngDeg']

    #        print(df_test_ex.loc[idx_test, 'latDeg'],
    #              df_train_gt.loc[idx, 'latDeg'])

#    visualize_match_test_gt(
#        df_test_bs, pair, first_phone_only=True)

    visualize_pair(
        df_test_bs, pair, first_phone_only=True)

    visualize_before_after(df_test_ex_before, df_test_ex)

#    return df_test_ex, idx_ex
    return df_test_ex

# df_test_ex.isna().sum()
# ===========================================================================


def read_pair(df_test_bs, pair, first_phone_only=True):

    print(first_phone_only)

    ROOT = Path('../input/google-smartphone-decimeter-challenge/')
    TRAIN = ROOT/'train'
    TEST = ROOT/'test'
    TRAIN_BASELINE = ROOT / 'baseline_locations_train.csv'

    test_path = list((TEST/pair['test']).glob('*'))

    print(test_path)
    df_baseline = pd.read_csv(TRAIN_BASELINE)

    if first_phone_only:
        test_path = [test_path[0]]

    print(test_path)
    print(type(test_path))

    for p_path in test_path:

        # read test data
        p = p_path.stem
        p_derived = list(p_path.glob('*derived.csv'))[0]

        print(p_path, p_derived)
        df_test_routes = pd.read_csv(p_derived)
        df_test_ex, idx_ex = extract_baseline(df_test_bs, df_test_routes)

        # find matching training gt
        train_phone_names = [p_tr.stem for p_tr in sorted(
            list((TRAIN/pair['train']).glob('*')))]

        if p not in train_phone_names:
            p_t = train_phone_names[0]
        else:
            p_t = p

        p_train_derived = list(
            (TRAIN/pair['train']/p_t).glob('*derived.csv'))[0]

        p_train_gt = list(
            (TRAIN/pair['train']/p_t).glob('*ground_truth.csv'))[0]

        print(p_path, p_derived)
        print(train_phone_names)
        print(p_t)
        print(p_train_derived)
        print(p_train_derived.exists())
        print(p_train_gt.exists())

        df_train_routes = pd.read_csv(p_train_derived)
        df_train_ex, _ = extract_baseline(df_baseline, df_train_routes)
        df_train_gt = pd.read_csv(p_train_gt)

    return df_test_ex, df_train_ex, df_train_gt, df_test_routes, df_train_routes, idx_ex

# x1, x2, x3, x4, x5, x6 = read_pair(pair, first_phone_only=True)

# ===========================================================================


def visualize_pair(df_test_bs, pair, first_phone_only=True):

    ROOT = Path('../input/google-smartphone-decimeter-challenge/')
    TRAIN = ROOT/'train'
    TEST = ROOT/'test'

#    for _, pair in pairs_all.iterrows():

    test_path = list((TEST/pair['test']).glob('*'))

    if first_phone_only:
        test_path = [test_path[0]]

    print(test_path)

    for p_path in test_path:

        p = p_path.stem
        p_derived = list(p_path.glob('*derived.csv'))[0]

        df_test_routes = pd.read_csv(p_derived)
        df_test_ex, _ = extract_baseline(df_test_bs, df_test_routes)

        train_phone_names = [p_tr.stem for p_tr in sorted(
            list((TRAIN/pair['train']).glob('*')))]

        if p not in train_phone_names:
            p_t = train_phone_names[0]
        else:
            p_t = p

        p_train_derived = list(
            (TRAIN/pair['train']/p_t).glob('*derived.csv'))[0]

        p_train_gt = list(
            (TRAIN/pair['train']/p_t).glob('*ground_truth.csv'))[0]

        print(p_path, p_derived)
        print(train_phone_names)
        print(p_t)
        print(p_train_derived)
        print(p_train_derived.exists())
        print(p_train_gt.exists())

        df_train_routes = pd.read_csv(p_train_derived)
#            df_train_ex, _ = extract_baseline(df_baseline, df_train_routes)
        df_train_ex = pd.read_csv(p_train_gt)

        # ===================================================================

        zoom = 16
        center = dict(lat=df_test_ex.loc[0, 'latDeg'],
                      lon=df_test_ex.loc[0, 'lngDeg'])

        trace1 = go.Scattermapbox(
            lat=df_train_ex['latDeg'],
            lon=df_train_ex['lngDeg'],
            #    marker_color=df_test_bs['in_motion'],
            mode='markers',
            #    marker=dict(size=8, color=df_test_bs['in_motion']),
            marker=dict(size=8, color='teal'),
            text=df_test_bs[['phoneName', 'millisSinceGpsEpoch']],
            hoverinfo='text')

        trace2 = go.Scattermapbox(
            lat=df_test_ex['latDeg'],
            lon=df_test_ex['lngDeg'],
            #    marker_color=df_test_bs['in_motion'],
            mode='markers',
            #    marker=dict(size=8, color=df_test_bs['in_motion']),
            marker=dict(size=8, color='darkorange'),
            text=df_test_bs[['phoneName', 'millisSinceGpsEpoch']],
            hoverinfo='text')

        data = [trace1, trace2]

        layout = go.Layout(hovermode='closest',
                           mapbox=dict(style='stamen-terrain',
                                       center=center,
                                       zoom=zoom),
                           width=1024*2, height=1024*2)

        fig = go.Figure(data=data, layout=layout)
        fig.show()

# ===========================================================================


def visualize_match_test_gt(df_test_bs, pairs_all, first_phone_only=True):

    ROOT = Path('../input/google-smartphone-decimeter-challenge/')
    TRAIN = ROOT/'train'
    TEST = ROOT/'test'

    for _, pair in pairs_all.iterrows():

        test_path = list((TEST/pair['test']).glob('*'))

        if first_phone_only:
            test_path = [test_path[0]]

        print(test_path)

        for p_path in test_path:

            p = p_path.stem
            p_derived = list(p_path.glob('*derived.csv'))[0]

            df_test_routes = pd.read_csv(p_derived)
            df_test_ex, _ = extract_baseline(df_test_bs, df_test_routes)

            train_phone_names = [p_tr.stem for p_tr in sorted(
                list((TRAIN/pair['train']).glob('*')))]

            if p not in train_phone_names:
                p_t = train_phone_names[0]
            else:
                p_t = p

            p_train_derived = list(
                (TRAIN/pair['train']/p_t).glob('*derived.csv'))[0]

            p_train_gt = list(
                (TRAIN/pair['train']/p_t).glob('*ground_truth.csv'))[0]

            print(p_path, p_derived)
            print(train_phone_names)
            print(p_t)
            print(p_train_derived)
            print(p_train_derived.exists())
            print(p_train_gt.exists())

            df_train_routes = pd.read_csv(p_train_derived)
#            df_train_ex, _ = extract_baseline(df_baseline, df_train_routes)
            df_train_ex = pd.read_csv(p_train_gt)

            # ===================================================================

            zoom = 16
            center = dict(lat=df_test_ex.loc[0, 'latDeg'],
                          lon=df_test_ex.loc[0, 'lngDeg'])

            trace1 = go.Scattermapbox(
                lat=df_train_ex['latDeg'],
                lon=df_train_ex['lngDeg'],
                #    marker_color=df_test_bs['in_motion'],
                mode='markers',
                #    marker=dict(size=8, color=df_test_bs['in_motion']),
                marker=dict(size=8, color='teal'),
                text=df_test_bs[['phoneName', 'millisSinceGpsEpoch']],
                hoverinfo='text')

            trace2 = go.Scattermapbox(
                lat=df_test_ex['latDeg'],
                lon=df_test_ex['lngDeg'],
                #    marker_color=df_test_bs['in_motion'],
                mode='markers',
                #    marker=dict(size=8, color=df_test_bs['in_motion']),
                marker=dict(size=8, color='darkorange'),
                text=df_test_bs[['phoneName', 'millisSinceGpsEpoch']],
                hoverinfo='text')

            data = [trace1, trace2]

            layout = go.Layout(hovermode='closest',
                               mapbox=dict(style='stamen-terrain',
                                           center=center,
                                           zoom=zoom),
                               width=1024*2, height=1024*2)

            fig = go.Figure(data=data, layout=layout)
            fig.show()

# --------------------------------------------------------------------------


# =========================================================================
# replace coordinate with no motion by mean
# =========================================================================


def kalman_clip(df_test_bs, path_test_routes, kf):

    for p in path_test_routes:

        #    df_test_bs = df_test_baseline
        df_test_routes = pd.read_csv(p)

        # cut short for easy handling
        df_test_ex, idx = extract_baseline(df_test_bs, df_test_routes)

    #    break
        # --------------------------------------------------------------------------
        # 1.
#        df_test_ex = clip_baseline(
#            df_test_ex, nn=7, sig_cutoff=2.0, inplace=True)

        df_test_ex = apply_kf_smoothing(df_test_ex, kf_=kf)

    #    break
        # 2.
#        df_test_ex = clip_baseline(
#            df_test_ex, nn=5, sig_cutoff=2.0, inplace=True)
#        df_test_ex = apply_kf_smoothing(df_test_ex, kf_=kf)

        df_before = df_test_bs.iloc[idx]
        df_test_bs.loc[idx, 'latDeg'] = df_test_ex['latDeg'].to_numpy()
        df_test_bs.loc[idx, 'lngDeg'] = df_test_ex['lngDeg'].to_numpy()

        # --------------------------------------------------------------------------
        # score
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # visualize
        # --------------------------------------------------------------------------

        visualize_before_after(df_before, df_test_bs.iloc[idx])
        # --------------------------------------------------------------------------

    return df_test_bs


# =========================================================================
# replace coordinate with no motion by mean
# =========================================================================

def replace_stationary(df_test_ex, n_idle=8):

    df_test_ex['in_motion'].shift(periods=1, axis=0)
    x = df_test_ex['in_motion'].shift(periods=1, axis=0)

    ix = df_test_ex[df_test_ex['in_motion'] != x].index.values
    if len(ix) != 1:
        ix = np.append(ix, df_test_ex.shape[0])

#    print(ix)
#    pdb.set_trace()

    n_total = df_test_ex.shape[0]
    n_replaced = 0

    for i, _ in enumerate(ix[:-1]):

        i_start = ix[i]
        i_end = ix[i+1] - 1
    #    print(i_start, i_end)

        if (df_test_ex.loc[i_start, 'in_motion'] == 's') & ((i_end - i_start) > n_idle):
            #            print(df_test_ex.loc[i_start:i_end, 'in_motion'])
            #            print('='*30)

            latx = df_test_ex.loc[i_start:i_end, 'latDeg'].values
            latx, _, _ = sigmaclip(latx, 2.5, 2.5)
            lat_mean = np.mean(latx)

            lngx = df_test_ex.loc[i_start:i_end, 'lngDeg'].values
            lngx, _, _ = sigmaclip(lngx, 2.5, 2.5)
            lng_mean = np.mean(lngx)
#            print(lat_mean, lng_mean)
#            pdb.set_trace()

#            lat_mean = df_test_ex.loc[i_start:i_end, 'latDeg'].mean()
#            lng_mean = df_test_ex.loc[i_start:i_end, 'lngDeg'].mean()

            df_test_ex.loc[i_start:i_end, 'latDeg'] = lat_mean
            df_test_ex.loc[i_start:i_end, 'lngDeg'] = lng_mean

            n_replaced += (i_end - i_start)

#    print(f'\033[1;95m {n_replaced}/{n_total}\033[0m')
#    pdb.set_trace()
    return df_test_ex, n_replaced, n_total

# =========================================================================
# apply offset
# =========================================================================
# print(train_phone_names)


def apply_offset_when_match(df_test_ex, pairs_all, DLAT_INFO):

    ROOT = Path('../input/google-smartphone-decimeter-challenge/')
    TRAIN = ROOT/'train'
#    TEST = ROOT/'test'

    d_info = pd.read_csv(DLAT_INFO)
    d_info = split_date_vehicle(d_info)
    d_info.rename({'phone': 'phoneName'}, axis=1, inplace=True)
    cols = ['dlat_x3s', 'dlng_x3s', 'dlat_x3m', 'dlng_x3m']

    phone_mean = d_info.groupby(['phoneName']).mean().loc[:, cols]


#    phone_mean.loc[p, :]
    df_test_ex = is_stationary(df_test_ex, rolling_window=16, dist_crit=8.0)

    c = df_test_ex['collectionName'].drop_duplicates().values[0]
    p = df_test_ex['phoneName'].drop_duplicates().values[0]

    c_t = pairs_all.loc[pairs_all['test'] == c, 'train'].values[0]
    train_phone_names = [p_t.stem for p_t in sorted(
        list((TRAIN/c_t).glob('*')))]

    if p not in train_phone_names:
        return df_test_ex
        # dx_info = phone_mean.loc[p, :]
        # dx_info = dx_info.to_frame().T
        # p_t = 'mean'

    else:
        dx_info = d_info.loc[(d_info['collection'] == c_t)
                             & (d_info['phoneName'] == p), cols]
        p_t = p

#    print(f'test  {c} {p}')
#    print(f'train {c_t} {p_t}')

#    print(dx_info)
#     print(df_test_ex.head(3))

    idx_s = df_test_ex[df_test_ex['in_motion'] == 's'].index
    idx_m = df_test_ex[df_test_ex['in_motion'] == 'm'].index

    df_test_ex.loc[idx_s, 'latDeg'] = df_test_ex.loc[idx_s,
                                                     'latDeg'] - dx_info['dlat_x3s'].values

    df_test_ex.loc[idx_s, 'lngDeg'] = df_test_ex.loc[idx_s,
                                                     'lngDeg'] - dx_info['dlng_x3s'].values

    df_test_ex.loc[idx_m, 'latDeg'] = df_test_ex.loc[idx_m,
                                                     'latDeg'] - dx_info['dlat_x3m'].values

    df_test_ex.loc[idx_m, 'lngDeg'] = df_test_ex.loc[idx_m,
                                                     'lngDeg'] - dx_info['dlng_x3m'].values

    return df_test_ex

# =========================================================================


def apply_offset_train(df_train_ex, pairs_all, DLAT_INFO):

    ROOT = Path('../input/google-smartphone-decimeter-challenge/')
    TRAIN = ROOT/'train'
#    TEST = ROOT/'test'

    d_info = pd.read_csv(DLAT_INFO)
    d_info = split_date_vehicle(d_info)
    d_info.rename({'phone': 'phoneName'}, axis=1, inplace=True)
    cols = ['dlat_x3s', 'dlng_x3s', 'dlat_x3m', 'dlng_x3m']

    phone_mean = d_info.groupby(['phoneName']).mean().loc[:, cols]

#    phone_mean.loc[p, :]
    df_train_ex = is_stationary(df_train_ex, rolling_window=16, dist_crit=8.0)

    c = df_train_ex['collectionName'].drop_duplicates().values[0]
    p = df_train_ex['phoneName'].drop_duplicates().values[0]

    dx_info = d_info.loc[(d_info['collection'] == c) &
                         (d_info['phoneName'] == p), cols]

#     print(df_train_ex.head(3))
    idx_s = df_train_ex[df_train_ex['in_motion'] == 's'].index
    idx_m = df_train_ex[df_train_ex['in_motion'] == 'm'].index

    df_train_ex.loc[idx_s, 'latDeg'] = df_train_ex.loc[idx_s,
                                                       'latDeg'] - dx_info['dlat_x3s'].values

    df_train_ex.loc[idx_s, 'lngDeg'] = df_train_ex.loc[idx_s,
                                                       'lngDeg'] - dx_info['dlng_x3s'].values

    df_train_ex.loc[idx_m, 'latDeg'] = df_train_ex.loc[idx_m,
                                                       'latDeg'] - dx_info['dlat_x3m'].values

    df_train_ex.loc[idx_m, 'lngDeg'] = df_train_ex.loc[idx_m,
                                                       'lngDeg'] - dx_info['dlng_x3m'].values

    return df_train_ex

# =========================================================================


def apply_offset(df_test_ex, pairs_all, DLAT_INFO):

    ROOT = Path('../input/google-smartphone-decimeter-challenge/')
    TRAIN = ROOT/'train'
#    TEST = ROOT/'test'

    d_info = pd.read_csv(DLAT_INFO)
    d_info = split_date_vehicle(d_info)
    d_info.rename({'phone': 'phoneName'}, axis=1, inplace=True)
    cols = ['dlat_x3s', 'dlng_x3s', 'dlat_x3m', 'dlng_x3m']

    phone_mean = d_info.groupby(['phoneName']).mean().loc[:, cols]

#    phone_mean.loc[p, :]
#    df_test_bs = is_stationary(df_test_bs, rolling_window=16, dist_crit=8.0)

    c = df_test_ex['collectionName'].drop_duplicates().values[0]
    p = df_test_ex['phoneName'].drop_duplicates().values[0]

    c_t = pairs_all.loc[pairs_all['test'] == c, 'train'].values[0]
    train_phone_names = [p_t.stem for p_t in sorted(
        list((TRAIN/c_t).glob('*')))]

    if p not in train_phone_names:
        dx_info = phone_mean.loc[p, :]
        dx_info = dx_info.to_frame().T
        p_t = 'mean'

    else:
        dx_info = d_info.loc[(d_info['collection'] == c_t)
                             & (d_info['phoneName'] == p), cols]
        p_t = p

    print(f'test  {c} {p}')
    print(f'train {c_t} {p_t}')
#    print(dx_info)

#     print(df_test_ex.head(3))
    idx_s = df_test_ex[df_test_ex['in_motion'] == 's'].index
    idx_m = df_test_ex[df_test_ex['in_motion'] == 'm'].index

    df_test_ex.loc[idx_s, 'latDeg'] = df_test_ex.loc[idx_s,
                                                     'latDeg'] - dx_info['dlat_x3s'].values

    df_test_ex.loc[idx_s, 'lngDeg'] = df_test_ex.loc[idx_s,
                                                     'lngDeg'] - dx_info['dlng_x3s'].values

    df_test_ex.loc[idx_m, 'latDeg'] = df_test_ex.loc[idx_m,
                                                     'latDeg'] - dx_info['dlat_x3m'].values

    df_test_ex.loc[idx_m, 'lngDeg'] = df_test_ex.loc[idx_m,
                                                     'lngDeg'] - dx_info['dlng_x3m'].values

    return df_test_ex

# =========================================================================


def apply_offset_all(df_test_bs, pairs_all, DLAT_INFO):

    ROOT = Path('../input/google-smartphone-decimeter-challenge/')
#    TRAIN = ROOT/'train'
    TEST = ROOT/'test'

    d_info = pd.read_csv(DLAT_INFO)
    d_info = split_date_vehicle(d_info)
    d_info.rename({'phone': 'phoneName'}, axis=1, inplace=True)
    cols = ['dlat_x3s', 'dlng_x3s', 'dlat_x3m', 'dlng_x3m']

    phone_mean = d_info.groupby(['phoneName']).mean().loc[:, cols]

#    phone_mean.loc[p, :]
#    df_test_bs = is_stationary(df_test_bs, rolling_window=16, dist_crit=8.0)

    for _, pair in pairs_all.iterrows():
        c = pair['test']
        c_t = pair['train']

        test_phone_names = [p.stem for p in sorted(
            list((TEST/pair['test']).glob('*')))]

        for p in test_phone_names:
            dx_info = d_info.loc[(d_info['collection'] == c_t)
                                 & (d_info['phoneName'] == p), cols]

            if dx_info.shape[0] == 0:
                dx_info = phone_mean.loc[p, :]
                dx_info = dx_info.to_frame().T
    #            print(dx_info)

    #        type(dx_info)
    #        print(c, p, dx[cols])
            phone = '_'.join([c, p])

            idx_s = df_test_bs[(df_test_bs['phone'] == phone) &
                               (df_test_bs['in_motion'] == 's')].index
            idx_m = df_test_bs[(df_test_bs['phone'] == phone) &
                               (df_test_bs['in_motion'] == 'm')].index

#           df_test_bs_before = df_test_bs.copy()
            # center = dict(lat=df_test_bs.loc[idx_s[0], 'latDeg'],
            #               lon=df_test_bs.loc[idx_s[0], 'lngDeg'])
            # idx = idx_s.append(idx_m)

            df_test_bs.loc[idx_s, 'latDeg'] = df_test_bs.loc[idx_s,
                                                             'latDeg'] - dx_info['dlat_x3s'].values

            df_test_bs.loc[idx_s, 'lngDeg'] = df_test_bs.loc[idx_s,
                                                             'lngDeg'] - dx_info['dlng_x3s'].values

            df_test_bs.loc[idx_m, 'latDeg'] = df_test_bs.loc[idx_m,
                                                             'latDeg'] - dx_info['dlat_x3m'].values

            df_test_bs.loc[idx_m, 'lngDeg'] = df_test_bs.loc[idx_m,
                                                             'lngDeg'] - dx_info['dlng_x3m'].values

    return df_test_bs

# =========================================================================
# to calculate stationary / in_motion
# =========================================================================


def is_stationary(df_test_bs, rolling_window=16, dist_crit=8.0):

    #    rolling_window = 16
    #    dist_crit = 8.0

    #    df_test_bs = pd.read_csv(TEST_BASELINE)
    x_latix = df_test_bs.loc[:, ['latDeg', 'lngDeg']]
    x_latix.set_index('latDeg', inplace=True)
#    x_latix
    dist_std = x_latix.copy()
    dist_std['dist_std'] = x_latix.rolling(
        rolling_window).apply(rolling_dist_std, raw=False)
    dist_std.reset_index(inplace=True)
    dist_std['dist_std'].fillna(value=0.0, inplace=True)

    # # --------------------------------------------------------------------------

    df_test_bs['in_motion'] = 'm'
    df_test_bs.loc[dist_std['dist_std'] < dist_crit, 'in_motion'] = 's'

    return df_test_bs

# # # ===========================================================================


def visualize_match_test_gt(df_test_bs, pairs_all, first_phone_only=True):

    ROOT = Path('../input/google-smartphone-decimeter-challenge/')
    TRAIN = ROOT/'train'
    TEST = ROOT/'test'

    for _, pair in pairs_all.iterrows():

        test_path = list((TEST/pair['test']).glob('*'))

        if first_phone_only:
            test_path = [test_path[0]]

        print(test_path)

        for p_path in test_path:

            p = p_path.stem
            p_derived = list(p_path.glob('*derived.csv'))[0]

            df_test_routes = pd.read_csv(p_derived)
            df_test_ex, _ = extract_baseline(df_test_bs, df_test_routes)

            train_phone_names = [p_tr.stem for p_tr in sorted(
                list((TRAIN/pair['train']).glob('*')))]

            if p not in train_phone_names:
                p_t = train_phone_names[0]
            else:
                p_t = p

            p_train_derived = list(
                (TRAIN/pair['train']/p_t).glob('*derived.csv'))[0]

            p_train_gt = list(
                (TRAIN/pair['train']/p_t).glob('*ground_truth.csv'))[0]

            print(p_path, p_derived)
            print(train_phone_names)
            print(p_t)
            print(p_train_derived)
            print(p_train_derived.exists())
            print(p_train_gt.exists())

            df_train_routes = pd.read_csv(p_train_derived)
#            df_train_ex, _ = extract_baseline(df_baseline, df_train_routes)
            df_train_ex = pd.read_csv(p_train_gt)

            # ===================================================================

            zoom = 16
            center = dict(lat=df_test_ex.loc[0, 'latDeg'],
                          lon=df_test_ex.loc[0, 'lngDeg'])

            trace1 = go.Scattermapbox(
                lat=df_train_ex['latDeg'],
                lon=df_train_ex['lngDeg'],
                #    marker_color=df_test_bs['in_motion'],
                mode='markers',
                #    marker=dict(size=8, color=df_test_bs['in_motion']),
                marker=dict(size=8, color='teal'),
                text=df_test_bs[['phoneName', 'millisSinceGpsEpoch']],
                hoverinfo='text')

            trace2 = go.Scattermapbox(
                lat=df_test_ex['latDeg'],
                lon=df_test_ex['lngDeg'],
                #    marker_color=df_test_bs['in_motion'],
                mode='markers',
                #    marker=dict(size=8, color=df_test_bs['in_motion']),
                marker=dict(size=8, color='darkorange'),
                text=df_test_bs[['phoneName', 'millisSinceGpsEpoch']],
                hoverinfo='text')

            data = [trace1, trace2]

            layout = go.Layout(hovermode='closest',
                               mapbox=dict(style='stamen-terrain',
                                           center=center,
                                           zoom=zoom),
                               width=1024*2, height=1024*2)

            fig = go.Figure(data=data, layout=layout)
            fig.show()


# # # ===========================================================================


# def visualize_stationary(df_test_bs):

#     zoom = 17
#     center = dict(lat=df_test_bs.loc[0, 'latDeg'],
#                   lon=df_test_bs.loc[0, 'lngDeg'])

#     trace1 = go.Scattermapbox(
#         lat=df_test_bs.loc[df_test_bs['in_motion'] == 'm', 'latDeg'],
#         lon=df_test_bs.loc[df_test_bs['in_motion'] == 'm', 'lngDeg'],
#         #    marker_color=df_test_bs['in_motion'],
#         mode='markers',
#         #    marker=dict(size=8, color=df_test_bs['in_motion']),
#         marker=dict(size=8, color='darkorange'),
#         text=df_test_bs[['phoneName', 'millisSinceGpsEpoch']],
#         hoverinfo='text')

#     trace2 = go.Scattermapbox(
#         lat=df_test_bs.loc[df_test_bs['in_motion'] == 's', 'latDeg'],
#         lon=df_test_bs.loc[df_test_bs['in_motion'] == 's', 'lngDeg'],
#         #    marker_color=df_test_bs['in_motion'],
#         mode='markers',
#         #    marker=dict(size=8, color=df_test_bs['in_motion']),
#         marker=dict(size=8, color='teal'),
#         text=df_test_bs[['phoneName', 'millisSinceGpsEpoch']],
#         hoverinfo='text')

#     data = [trace1, trace2]

#     layout = go.Layout(hovermode='closest',
#                        mapbox=dict(style='stamen-terrain',
#                                    center=center,
#                                    zoom=zoom),
#                        width=1024*2, height=1024*2)

#     fig = go.Figure(data=data, layout=layout)
#     fig.show()

# # # ===========================================================================


def rolling_dist_std(x):
    '''
     x: ndarray => pd.Series (with latDeg as index
    '''
#    print(f'x.shape {x.shape}')
#    print(f'type(x) {type(x)}')

    x_lat = x.index.to_numpy()
    x_lng = x.values

#    print(f'lat {x_lat}')
#    print(f'lng {x_lng}')

#    x_lat_roll = np.roll(x_lat, shift=1, axis=0)
#    x_lng_roll = np.roll(x_lng, shift=1, axis=0)

#    x_lat_roll = x_lat_roll[1:]
#    x_lng_roll = x_lng_roll[1:]

    n_repeat = x_lng.shape[0] - 1

    x_lat_start = np.repeat(x_lat[0], n_repeat)
    x_lng_start = np.repeat(x_lng[0], n_repeat)

    x_lat = x_lat[1:]
    x_lng = x_lng[1:]

#    dist = calc_haversine(x_lat, x_lng, x_lat_roll, x_lng_roll)
    dist = calc_haversine(x_lat_start, x_lng_start, x_lat, x_lng)
#    print(f'dist {type(dist)} {dist.shape}')

    return dist.std()

# =========================================================================
# alvin imu
# =========================================================================


def imu_correct(df_test_bs, pairs, window_size):

    ROOT = Path('../input/google-smartphone-decimeter-challenge/')
    TRAIN = ROOT/'train'
    TEST = ROOT/'test'
    TRAIN_BASELINE = ROOT / 'baseline_locations_train.csv'
    SAMPLE_SUBMISSION = ROOT / 'sample_submission.csv'

    df_baseline = pd.read_csv(TRAIN_BASELINE)
    sub_sample = pd.read_csv(SAMPLE_SUBMISSION)

    params = {
        'metric': 'mse',
        'objective': 'regression',
        'seed': 2021,
        'boosting_type': 'gbdt',
        #    'early_stopping_rounds': 10,
        'early_stopping_rounds': 32,
        'subsample': 0.7,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'reg_lambda': 10
    }
    folds = 5
    # --------------------------------------------------------------------------
    score_record_list = []
    for i, pair in pairs.iterrows():

        df_test_bs_before = pd.read_csv(TEST_BASELINE)
        print('')
        print(f'\033[33m===================================\033[0m')
        print(f'\033[33mTest \033[0m {i} \n{pair}')

        test_phone_names = [p.stem for p in sorted(
            list((TEST/pair['test']).glob('*')))]
        train_phone_names = [p.stem for p in sorted(
            list((TRAIN/pair['train']).glob('*')))]

        print(test_phone_names)
        print(train_phone_names)

        for p in test_phone_names:

            if p not in train_phone_names:
                p_t = train_phone_names[0]
            else:
                p_t = p

            c_t = pair['train']
            print(f'\033[36mtrain \033[0m{c_t} {p_t}')
            df_all_train = prepare_imu_data(
                ROOT, 'train', c_t, p_t, df_baseline, sub_sample)

            if df_all_train.shape[0] == 0:
                print(f'\033[31mSkipping {c_t} {p_t}\033[0m')
                continue

            lat_lng_df_train, df_all_train = get_xyz(df_all_train, 'train')
            df_train = prepare_df_train(df_all_train,  window_size)  # 所有轴的数据

        # --------------------------------------------------------------------------

            print(f'\033[36m=\033[0m'*20)
            print(f'\033[32mpreparing df_test {i} {p}\033[0m')
            c = pair['test']
            print(f'\033[33m{c} {p}\033[0m')

            df_all_test = prepare_imu_data(
                ROOT, 'test', c, p, df_test_bs, sub_sample)

            if df_all_test.shape[0] == 0:
                print(f'\033[31mSkipping {c} {p}\033[0m')
                continue

            lat_lng_df_test, df_all_test = get_xyz(df_all_test, 'test')

            df_test = prepare_df_test(df_all_test,  window_size)

            df_train_x, df_test_x, pred_valid_x, pred_test_x = training(
                df_train, df_test, 'X', window_size, folds, params)

            df_train_y, df_test_y, pred_valid_y, pred_test_y = training(
                df_train, df_test, 'Y', window_size, folds, params)

            df_train_z, df_test_z, pred_valid_z, pred_test_z = training(
                df_train, df_test, 'Z', window_size, folds, params)

            # --------------------------------------------------------------------------
            val_compare_df = pd.DataFrame({'Xgt': df_train_x['Xgt'].values, 'Xpred': pred_valid_x,
                                           'Ygt': df_train_y['Ygt'].values, 'Ypred': pred_valid_y,
                                           'Zgt': df_train_z['Zgt'].values, 'Zpred': pred_valid_z})

            lng_gt, lat_gt, _ = ECEF_to_WGS84(
                val_compare_df['Xgt'].values, val_compare_df['Ygt'].values, val_compare_df['Zgt'].values)

            lng_pred, lat_pred, _ = ECEF_to_WGS84(
                val_compare_df['Xpred'].values, val_compare_df['Ypred'].values, val_compare_df['Zpred'].values)

            lng_test_pred, lat_test_pred, _ = ECEF_to_WGS84(
                pred_test_x, pred_test_y, pred_test_z)

            phone = '_'.join([c, p])
            print(f'\033[33mphone {phone}\033[0m')

            # idx = df_test_bs[df_test_bs['phone']
            #                  == phone].index.values[window_size:]
            # print(idx)
            idx = df_test_bs[df_test_bs['phone']
                             == phone].index[window_size:]

    #        print(idx)
            df_test_bs.loc[idx, 'latDeg'] = lat_test_pred
            df_test_bs.loc[idx, 'lngDeg'] = lng_test_pred

            # -------------------------------------------
            df_train_before = lat_lng_df_train[['latDeg_bl', 'lngDeg_bl']]

            df_train_before.rename(
                {'latDeg_bl': 'latDeg', 'lngDeg_bl': 'lngDeg'}, inplace=True, axis=1)

            df_train_after = pd.DataFrame(
                dict(latDeg=lat_pred, lngDeg=lng_pred))

            df_train_gt = pd.DataFrame(
                dict(latDeg=lat_gt, lngDeg=lng_gt))
            # -------------------------------------------
            score_before = score_imu(df_train_before, df_train_gt)
            score_after = score_imu(df_train_after, df_train_gt)

            score_before_test = score_imu(
                df_test_bs_before.iloc[idx], df_train_gt)
            score_after_test = score_imu(df_test_bs.iloc[idx], df_train_gt)

            print(
                f'\033[32mBEFORE {score_before}\033[0m')
            print(
                f'\033[36mAFTER  {score_after}\033[0m')
            print(
                f'\033[32mBEFORE (TEST) {score_before_test}\033[0m')
            print(
                f'\033[36mAFTER  (TEST) {score_after_test}\033[0m')

            # s_rec = dict(collectionName=[c], phoneName=[p],
            #              before=[score_before[0]], after=[score_after[0]])
            # s_rec = pd.DataFrame(s_rec)
            # score_record_list.append(s_rec)

            # -------------------------------------------
            #        visualize_imu(df_test_before, df_test_after, df_train_gt)

        center = dict(lat=df_test_bs_before.loc[idx[0], 'latDeg'],
                      lon=df_test_bs_before.loc[idx[0], 'lngDeg'])

        visualize_before_after(df_test_bs_before.iloc[idx],
                               df_test_bs.iloc[idx], center)

    return df_test_bs


# =========================================================================
# road type
# =========================================================================

def road_matching(TEST, TRAIN, TEST_BASELINE, TRAIN_BASELINE):

    elias_kaggle = Path('/Users/meg/ka/k6/')

    df_baseline = pd.read_csv(TRAIN_BASELINE)
    df_test_baseline = pd.read_csv(TEST_BASELINE)
    # --------------------------------------------------------------------------
    # sub_sample.rename(columns={msge: msge_sample}, inplace=True)
    df_test_baseline = pd.read_csv(TEST_BASELINE)
#    path_test_routes = sorted(list(TEST.glob('*/*/*_derived.csv')))

    collections, test_collections = road_type(df_baseline, df_test_baseline)
#    test_collections
    dir_test_routes = sorted(list(TEST.glob('*')))
    dir_train_routes = sorted(list(TRAIN.glob('*')))

    # for (p, r), cpr in test_collections.groupby(['phoneName', 'roadType']):
    # #    print(p, r)
    #     print(cpr)

    d_list = []
    for dir in dir_test_routes:
        dir_path = list(dir.glob('*/*_derived.csv'))

        for p in dir_path:
            df_test_routes = pd.read_csv(p)
            df_test_bs, _ = extract_baseline(df_test_baseline, df_test_routes)
            dfx_test = merge_routes_base(df_test_routes, df_test_bs)
            x = dfx_test.groupby(['latDeg', 'lngDeg']).mean()
            x = x.reset_index(level=['latDeg', 'lngDeg'], col_level=1)
            lat_mean = x['latDeg'].mean()
            lng_mean = x['lngDeg'].mean()
            collection = dfx_test['collectionName'].drop_duplicates()
            d = dict(collectionName=collection, lat=lat_mean, lng=lng_mean)
            d = pd.DataFrame(d)
            d_list.append(d)

    road_test = pd.concat(d_list, axis=0, ignore_index=True)
    road_test = road_test.groupby('collectionName').mean()
    road_test = road_test.reset_index(level=['collectionName'], col_level=1)

    # -------------------------
    d_list = []
    for dir in dir_train_routes:
        dir_path = list(dir.glob('*/*_derived.csv'))

        for p in dir_path:
            df_train_routes = pd.read_csv(p)
            df_train_bs, _ = extract_baseline(df_baseline, df_train_routes)
            dfx_train = merge_routes_base(df_train_routes, df_train_bs)
            x = dfx_train.groupby(['latDeg', 'lngDeg']).mean()
            x = x.reset_index(level=['latDeg', 'lngDeg'], col_level=1)
            lat_mean = x['latDeg'].mean()
            lng_mean = x['lngDeg'].mean()
            collection = dfx_train['collectionName'].drop_duplicates()
            d = dict(collectionName=collection, lat=lat_mean, lng=lng_mean)
            d = pd.DataFrame(d)
            d_list.append(d)

    road_train = pd.concat(d_list, axis=0, ignore_index=True)
    road_train = road_train.groupby('collectionName').mean()
    road_train = road_train.reset_index(level=['collectionName'], col_level=1)

#    road_test
#    road_train

    # ------------------------------
    pair_list = []
#    pair_list
    for c, r_lat, r_lng in zip(road_test['collectionName'], road_test['lat'], road_test['lng']):
        #    print(c, r_lat, r_lng)
        x = road_train['lat'] - r_lat
        y = road_train['lng'] - r_lng
        dist = x * x + y * y
        c_train = road_train.loc[dist.argmin(), 'collectionName']
    #    print(c)
    #    print(c_train)
        d = dict(test=[c], train=[c_train])
        d = pd.DataFrame(d)
    #    print(d)
        pair_list.append(d)

#    pair_list

    pair = pd.concat(pair_list, axis=0, ignore_index=True)
    pair.reset_index(drop=True, inplace=True)
#    pair

    tc = test_collections.rename({'collectionName': 'test'}, axis=1)
    tc.drop('phoneName', axis=1, inplace=True)
    tc = tc.drop_duplicates().reset_index(drop=True)
    pair = pair.merge(tc, on='test', how='inner')
#    pair

    train_c = collections.rename({'collectionName': 'train'}, axis=1)
    train_c.drop('phoneName', axis=1, inplace=True)
    train_c = train_c.drop_duplicates().reset_index(drop=True)
    pair = pair.merge(train_c, on='train', how='inner')

    pair.to_csv(elias_kaggle/'log'/'pairs_all.csv')
    return pair


'''

     test                train roadType_x roadType_y

1   2020-05-28-US-MTV-1  2020-05-21-US-MTV-2   autobahn   autobahn good
2   2020-06-10-US-MTV-2  2020-05-21-US-MTV-2   autobahn   autobahn good
3   2020-08-03-US-MTV-2  2020-05-21-US-MTV-2   autobahn   autobahn good
4   2020-05-28-US-MTV-2  2020-05-14-US-MTV-2   autobahn   autobahn good
5   2020-06-04-US-MTV-2  2020-06-04-US-MTV-1   autobahn   autobahn good
6   2020-06-10-US-MTV-1  2020-06-04-US-MTV-1   autobahn   autobahn good

7   2020-08-13-US-MTV-1  2020-06-05-US-MTV-1   autobahn   autobahn good
8   2021-03-16-US-MTV-2  2021-01-04-US-RWC-1   autobahn   autobahn good

9   2021-03-16-US-RWC-2  2021-04-28-US-MTV-1       tree       tree good
10  2021-04-28-US-MTV-2  2021-04-28-US-MTV-1       tree       tree good
11  2021-04-29-US-MTV-2  2021-04-28-US-MTV-1       tree       tree good

12  2021-03-25-US-PAO-1  2020-07-17-US-MTV-1       tree   autobahn  ng 2021-04-15-MTV-1
13  2021-04-02-US-SJC-1  2021-04-22-US-SJC-1       tree     canyon  ng
14  2021-04-08-US-MTV-1  2021-04-29-US-MTV-1       tree       tree  ng

2020-09-04-US-SF-1 /
2021-04-15-MTV-1 /
2020-0904--US-SF-2 /
2020-08-06-US-MTV-2 /
2020-08-03-US-MTV-1 /
2020-07-17-US-MTV-2 /
2020-07-17-US--MTV-1 /
2020-06-11-US-MTV-1 /
2020-06-05-US-MTV-1 / 2
020-06-04-US-MTV-1 /
2020-05-29-US-MTV-2 /
2020-05-29-US-MTV-1 /
2020-05-14-US-MTV-2

15  2021-04-21-US-MTV-1  2021-04-29-US-MTV-1       tree       tree good

16  2021-04-22-US-SJC-2  2021-04-28-US-SJC-1     canyon     canyon  good
17  2021-04-26-US-SVL-2  2021-04-26-US-SVL-1       tree       tree  good
18  2021-04-29-US-SJC-3  2021-04-29-US-SJC-2     canyon     canyon  good

'''

# --------------------------------------------------------------------------


def road_type(df_baseline, df_test_baseline):

    train_autobahn = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    train_tree = [22, 23, 25, 26, 28]
    train_canyon = [24, 27, 29]

    test_autobahn = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    test_tree = [10, 11, 12, 13, 14, 16, 17, 18]
    test_canyon = [15, 19]

    # --------------------------------------------------------------------------

#    path_routes = list(TRAIN.glob('*/*/*_derived.csv'))
#    path_routes_collections = list(TRAIN.glob('*'))
    collections = df_baseline[[
        'collectionName', 'phoneName']].drop_duplicates()
    collections_type_list = list(
        collections['collectionName'].drop_duplicates().values)
 #   collections_type_list = [p.stem for p in path_routes_collections]
    road_type_list = []

    for i, c in enumerate(collections_type_list):
        s = ''
        if i+1 in train_autobahn:
            s = 'autobahn'
        if i+1 in train_tree:
            s = 'tree'
        if i+1 in train_canyon:
            s = 'canyon'

        road_type_list.append(s)

    collections_type_df = pd.DataFrame(
        dict(collectionName=collections_type_list))
    road_type_df = pd.DataFrame(dict(roadType=road_type_list))

    road_df = pd.concat([collections_type_df, road_type_df], axis=1)
#    road_df

    train_collections = collections.merge(road_df, on='collectionName',
                                          how='left', suffixes=('', ''))
    # --------------------------------------------------------------------------
#    path_test_routes = list(TEST.glob('*/*/*_derived.csv'))

    # print(df_test_baseline.head(3))
    # print(df_test_baseline.columns)

    # pdb.set_trace()

    test_collections = df_test_baseline[[
        'collectionName', 'phoneName']].drop_duplicates()

#    pdb.set_trace()

#    path_test_routes_collections = list(TEST.glob('*'))

#    test_collections_type_list = [p.stem for p in path_test_routes_collections]
    test_collections_type_list = list(
        test_collections['collectionName'].drop_duplicates().values)

#    pdb.set_trace()

    test_road_type_list = []

    for i, c in enumerate(test_collections_type_list):
        s = ''
        if i+1 in test_autobahn:
            s = 'autobahn'
        if i+1 in test_tree:
            s = 'tree'
        if i+1 in test_canyon:
            s = 'canyon'

        test_road_type_list.append(s)

    test_collections_type_df = pd.DataFrame(
        dict(collectionName=test_collections_type_list))
    test_road_type_df = pd.DataFrame(dict(roadType=test_road_type_list))

    test_road_df = pd.concat(
        [test_collections_type_df, test_road_type_df], axis=1)

#    print(test_road_df.head(3))
#    print(test_collections.head(3))
#    pdb.set_trace()
    test_collections = test_collections.merge(test_road_df, on=['collectionName'],
                                              how='left', suffixes=('', ''))

    return train_collections, test_collections

# --------------------------------------------------------------------------


def road_type_path(TRAIN, TEST):

    train_autobahn = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    train_tree = [22, 23, 25, 26, 28]
    train_canyon = [24, 27, 29]

    test_autobahn = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    test_tree = [10, 11, 12, 13, 14, 16, 17, 18]
    test_canyon = [15, 19]

    # --------------------------------------------------------------------------

#    path_routes = list(TRAIN.glob('*/*/*_derived.csv'))
    path_routes_collections = list(TRAIN.glob('*'))

    collections_type_list = [p.stem for p in path_routes_collections]
    road_type_list = []

    for i, c in enumerate(collections_type_list):
        s = ''
        if i+1 in train_autobahn:
            s = 'autobahn'
        if i+1 in train_tree:
            s = 'tree'
        if i+1 in train_canyon:
            s = 'canyon'

        road_type_list.append(s)

    collections_type_df = pd.DataFrame(
        dict(collectionName=collections_type_list))
    road_type_df = pd.DataFrame(dict(roadType=road_type_list))

    road_df = pd.concat([collections_type_df, road_type_df],
                        axis=1, ignore_index=True)
    road_df

    # --------------------------------------------------------------------------

#    path_test_routes = list(TEST.glob('*/*/*_derived.csv'))
    path_test_routes_collections = list(TEST.glob('*'))

    test_collections_type_list = [p.stem for p in path_test_routes_collections]
    test_road_type_list = []

    for i, c in enumerate(test_collections_type_list):
        s = ''
        if i+1 in test_autobahn:
            s = 'autobahn'
        if i+1 in test_tree:
            s = 'tree'
        if i+1 in test_canyon:
            s = 'canyon'

        test_road_type_list.append(s)

    test_collections_type_df = pd.DataFrame(
        dict(collectionName=test_collections_type_list))
    test_road_type_df = pd.DataFrame(dict(roadType=test_road_type_list))

    test_road_df = pd.concat(
        [test_collections_type_df, test_road_type_df], axis=1, ignore_index=True)

    return road_df, test_road_df


# =========================================================================
# IMU by alvin
# =========================================================================

# pitch:y
# yaw:z
# roll:x
# =========================================================================


def an2v(y_delta, z_delta, x_delta):
    '''
    Euler Angles ->Rotation Matrix -> Rotation Vector

    Input：
        1. y_delta          (float): the angle with rotateing around y-axis.
        2. z_delta         (float): the angle with rotateing around z-axis.
        3. x_delta         (float): the angle with rotateing around x-axis.
    Output：
        rx/ry/rz             (float): the rotation vector with rotateing


    '''
    # yaw: z
    Rz_Matrix = np.matrix([
        [cos(z_delta), -sin(z_delta), 0],
        [sin(z_delta), cos(z_delta), 0],
        [0, 0, 1]
    ])

    # pitch: y
    Ry_Matrix = np.matrix([
        [cos(y_delta), 0, sin(y_delta)],
        [0, 1, 0],
        [-sin(y_delta), 0, cos(y_delta)]
    ])

    # roll: x
    Rx_Matrix = np.matrix([
        [1, 0, 0],
        [0, cos(x_delta), -sin(x_delta)],
        [0, sin(x_delta), cos(x_delta)]
    ])

    R = Rz_Matrix * Ry_Matrix * Rx_Matrix

    theta = acos(((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2)
    multi = 1 / (2 * sin(theta))

    rx = multi * (R[2, 1] - R[1, 2]) * theta
    ry = multi * (R[0, 2] - R[2, 0]) * theta
    rz = multi * (R[1, 0] - R[0, 1]) * theta

    return rx, ry, rz

# --------------------------------------------------------------------------


def v2a(rotation_v):
    '''
    Rotation Vector -> Rotation Matrix -> Euler Angles

    Input：
        rx/ry/rz             (float): the rotation vector with rotateing around x/y/z-axis.
    Output：
        1. y_delta          (float): the angle with rotateing around y-axis.
        2. z_delta         (float): the angle with rotateing around z-axis.
        3. x_delta         (float): the angle with rotateing around x-axis.
    '''
    # Rotation Vector -> Rotation Matrix
    R = Rodrigues(rotation_v)[0]

    sq = sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)

    if not (sq < 1e-6):
        x_delta = atan2(R[2, 1], R[2, 2])
        y_delta = atan2(-R[2, 0], sq)
        z_delta = atan2(R[1, 0], R[0, 0])
    else:
        x_delta = atan2(-R[1, 2], R[1, 1])
        y_delta = atan2(-R[2, 0], sq)
        z_delta = 0

    return y_delta, z_delta, x_delta

# --------------------------------------------------------------------------


def UTC2GpsEpoch(df):
    '''UTC to GpsEpoch

    utcTimeMillis         : UTC epoch (1970/1/1)
    millisSinceGpsEpoch   : GPS epoch(1980/1/6 midnight 12:00 UTC)

    Ref: https://www.kaggle.com/c/google-smartphone-decimeter-challenge/discussion/239187
    '''
    dt_offset = pd.to_datetime('1980-01-06 00:00:00')
    dt_offset_in_ms = int(dt_offset.value / 1e6)
    df['millisSinceGpsEpoch'] = df['utcTimeMillis'] - dt_offset_in_ms + 18000
    return df

# --------------------------------------------------------------------------


def prepare_imu_data(data_dir, dataset_name, cname, pname, bl_df, sample_df):
    '''Prepare IMU Dataset (For Train: IMU+GT+BL; For Test: IMU+BL)
    Input：
        1. data_dir: data_dir
        2. dataset_name: dataset name（'train'/'test'）
        3. cname: CollectionName
        4. pname: phoneName
        5. bl_df: baseline's dataframe
    Output：df_all
    '''
    gnss_df = gnss_log_to_dataframes(
        str(data_dir / dataset_name / cname / pname / f'{pname}_GnssLog.txt'))
    # print('sub-dataset shape：')
    # print('Raw:', gnss_df['Raw'].shape)
    # print('Status:', gnss_df['Status'].shape)
    # print('UncalAccel:', gnss_df['UncalAccel'].shape)
    # print('UncalGyro:', gnss_df['UncalGyro'].shape)
    # print('UncalMag:', gnss_df['UncalMag'].shape)
    # print('OrientationDeg:', gnss_df['OrientationDeg'].shape)
    # print('Fix:', gnss_df['Fix'].shape)

#    print(gnss_df['OrientationDeg'].shape[0])
    if gnss_df['OrientationDeg'].shape[0] == 0:
        print(f'\033[33mSkipping {cname} {pname}\033[0m')
        return pd.DataFrame()

#    print(gnss_df['OrientationDeg'].shape[0])
    if gnss_df['UncalGyro']['DriftXRadPerSec'].dtype == 'object':
        print(f'\033[33mXradPerSec incomplete\033[0m')
        return pd.DataFrame()

    # merge sub-datasets
    # accel + gyro
    imu_df = pd.merge_asof(gnss_df['UncalAccel'].sort_values('utcTimeMillis'),
                           gnss_df['UncalGyro'].drop(
        'elapsedRealtimeNanos', axis=1).sort_values('utcTimeMillis'),
        on='utcTimeMillis',
        direction='nearest')
    # (accel + gyro) + mag
    imu_df = pd.merge_asof(imu_df.sort_values('utcTimeMillis'),
                           gnss_df['UncalMag'].drop(
        'elapsedRealtimeNanos', axis=1).sort_values('utcTimeMillis'),
        on='utcTimeMillis',
        direction='nearest')
    # ((accel + gyro) + mag) + OrientationDeg
    imu_df = pd.merge_asof(imu_df.sort_values('utcTimeMillis'),
                           gnss_df['OrientationDeg'].drop(
        'elapsedRealtimeNanos', axis=1).sort_values('utcTimeMillis'),
        on='utcTimeMillis',
        direction='nearest')

    # UTC->GpsEpoch
    imu_df = UTC2GpsEpoch(imu_df)

    # print IMU time
    dt_offset = pd.to_datetime('1980-01-06 00:00:00')
    dt_offset_in_ms = int(dt_offset.value / 1e6)
    tmp_datetime = pd.to_datetime(
        imu_df['millisSinceGpsEpoch'] + dt_offset_in_ms, unit='ms')
    print(f"imu_df time scope: {tmp_datetime.min()} - {tmp_datetime.max()}")

    if dataset_name == 'train':
        # read GT dataset
        gt_path = data_dir / dataset_name / cname / pname / 'ground_truth.csv'
        gt_df = pd.read_csv(gt_path, usecols=[
            'collectionName', 'phoneName', 'millisSinceGpsEpoch', 'latDeg', 'lngDeg'])

        # print GT time
        tmp_datetime = pd.to_datetime(
            gt_df['millisSinceGpsEpoch'] + dt_offset_in_ms, unit='ms')
        print(f"gt_df time scope: {tmp_datetime.min()} - {tmp_datetime.max()}")

        # merge GT dataset
        imu_df = pd.merge_asof(gt_df.sort_values('millisSinceGpsEpoch'),
                               imu_df.drop(['elapsedRealtimeNanos'], axis=1).sort_values(
            'millisSinceGpsEpoch'),
            on='millisSinceGpsEpoch',
            direction='nearest')
    elif dataset_name == 'test':
        # merge smaple_df
        imu_df = pd.merge_asof(sample_df.sort_values('millisSinceGpsEpoch'),
                               imu_df.drop(['elapsedRealtimeNanos'], axis=1).sort_values(
            'millisSinceGpsEpoch'),
            on='millisSinceGpsEpoch',
            direction='nearest')

    # OrientationDeg -> Rotation Vector
    rxs = []
    rys = []
    rzs = []
#    print(f'\033[33mhere 2\033[0m')
    for i in range(len(imu_df)):
        y_delta = imu_df['rollDeg'].iloc[i]
        z_delta = imu_df['yawDeg'].iloc[i]
        x_delta = imu_df['pitchDeg'].iloc[i]
#        print(f'\033[33m{i} here 3\033[0m')
        rx, ry, rz = an2v(y_delta, z_delta, x_delta)
        rxs.append(rx)
        rys.append(ry)
        rzs.append(rz)

    imu_df['ahrsX'] = rxs
    imu_df['ahrsY'] = rys
    imu_df['ahrsZ'] = rzs

    for axis in ['X', 'Y', 'Z']:

        imu_df['Accel{}Mps2'.format(axis)] = imu_df['UncalAccel{}Mps2'.format(
            axis)] - imu_df['Bias{}Mps2'.format(axis)]
#        print(f'\033[33mhere 6\033[0m')

        imu_df['Gyro{}RadPerSec'.format(axis)] = imu_df['UncalGyro{}RadPerSec'.format(
            axis)] - imu_df['Drift{}RadPerSec'.format(axis)]

#        print(f'\033[33mhere 7\033[0m')
        imu_df['Mag{}MicroT'.format(axis)] = imu_df['UncalMag{}MicroT'.format(
            axis)] - imu_df['Bias{}MicroT'.format(axis)]

#        print(f'\033[33mhere 8\033[0m')
        # clearn bias features
        imu_df.drop(['Bias{}Mps2'.format(axis), 'Drift{}RadPerSec'.format(
            axis), 'Bias{}MicroT'.format(axis)], axis=1, inplace=True)

#    print(f'\033[33mhere 9\033[0m')

    if dataset_name == 'train':
        # merge Baseline dataset：imu_df + bl_df = (GT + IMU) + Baseline
        df_all = pd.merge(imu_df.rename(columns={'latDeg': 'latDeg_gt', 'lngDeg': 'lngDeg_gt'}),
                          bl_df.drop(['phone'], axis=1).rename(
            columns={'latDeg': 'latDeg_bl', 'lngDeg': 'lngDeg_bl'}),
            on=['collectionName', 'phoneName', 'millisSinceGpsEpoch'])

    elif dataset_name == 'test':
        # print(f'imu_df {imu_df.shape}')
        # print(f'bl_df {bl_df.shape}')
        # print(
        #     f"{bl_df[(bl_df['collectionName'] == cname) & (bl_df['phoneName'] == pname)].shape}")

        phone = '_'.join([cname, pname])
        imu_df = imu_df[imu_df['phone'] == phone]

        df_all = pd.merge(imu_df,
                          bl_df[(bl_df['collectionName'] == cname) & (bl_df['phoneName'] == pname)].drop(
                              ['phone'], axis=1).rename(
                              columns={'latDeg': 'latDeg_bl', 'lngDeg': 'lngDeg_bl'}),
                          on=['millisSinceGpsEpoch'])
        df_all.drop(['phone'], axis=1, inplace=True)
#        print(f'df_allf {df_all.shape}')
#     pdb.set_trace()

    return df_all


# --------------------------------------------------------------------------

def add_stat_feats(data, tgt_axis, window_size):
    for f in ['yawZDeg', 'rollYDeg', 'pitchXDeg']:
        if f.find(tgt_axis) >= 0:
            ori_feat = f
            break

    cont_feats = ['heightAboveWgs84EllipsoidM', 'ahrs{}'.format(tgt_axis),
                  'Accel{}Mps2'.format(tgt_axis), 'Gyro{}RadPerSec'.format(
        tgt_axis), 'Mag{}MicroT'.format(tgt_axis),
        '{}bl'.format(tgt_axis)] + [ori_feat]

    for f in cont_feats:
        data[f + '_' + str(window_size) + '_mean'] = data[[f +
                                                           f'_{i}' for i in range(1, window_size)]].mean(axis=1)
        data[f + '_' + str(window_size) + '_std'] = data[[f +
                                                          f'_{i}' for i in range(1, window_size)]].std(axis=1)
        data[f + '_' + str(window_size) + '_max'] = data[[f +
                                                          f'_{i}' for i in range(1, window_size)]].max(axis=1)
        data[f + '_' + str(window_size) + '_min'] = data[[f +
                                                          f'_{i}' for i in range(1, window_size)]].min(axis=1)
        data[f + '_' + str(window_size) + '_median'] = data[[f +
                                                             f'_{i}' for i in range(1, window_size)]].median(axis=1)
    return data

# --------------------------------------------------------------------------


def training(df_train, df_test, tgt_axis, window_size, folds, params):
    '''For the given axis target to train the model. Also, it has validation and prediciton.'''
    df_train = remove_other_axis_feats(df_train, tgt_axis)
    df_train = add_stat_feats(df_train, tgt_axis, window_size)
    df_test = remove_other_axis_feats(df_test, tgt_axis)
    df_test = add_stat_feats(df_test, tgt_axis, window_size)

    feature_names = [f for f in list(df_train) if f not in [
        'Xgt', 'Ygt', 'Zgt']]
    target = '{}gt'.format(tgt_axis)

    kfold = KFold(n_splits=folds, shuffle=True, random_state=params['seed'])

    pred_valid = np.zeros((len(df_train),))
    pred_test = np.zeros((len(df_test),))
    scores = []
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df_train, df_train[target])):
        X_train = df_train.iloc[trn_idx][feature_names]
        Y_train = df_train.iloc[trn_idx][target]
        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][target]

        model = lgb.LGBMRegressor(**params)
        lgb_model = model.fit(X_train,
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              verbose=0,
                              eval_metric=params['metric'],
                              early_stopping_rounds=params['early_stopping_rounds'])

        pred_valid[val_idx] = lgb_model.predict(
            X_val, num_iteration=lgb_model.best_iteration_)
        pred_test += lgb_model.predict(df_test[feature_names],
                                       num_iteration=lgb_model.best_iteration_)

        scores.append(lgb_model.best_score_['valid']['l2'])

    pred_test = pred_test / kfold.n_splits

    # if verbose_flag == True:
    #     print("Each Fold's MSE：{}, Average MSE：{:.4f}".format(
    #         [np.round(v, 2) for v in scores], np.mean(scores)))
    #     print("-"*60)

#    if verbose_flag == True:
    print("Each Fold's MSE：{}, Average MSE：{:.4f}".format(
        [np.round(v, 2) for v in scores], np.mean(scores)))
    print("-"*60)

    return df_train, df_test, pred_valid, pred_test


# --------------------------------------------------------------------------


def get_xyz(df_all, dataset_name):
    # baseline: lat/lngDeg -> x/y/z
    df_all['Xbl'], df_all['Ybl'], df_all['Zbl'] = zip(
        *df_all.apply(lambda x: WGS84_to_ECEF(x.latDeg_bl, x.lngDeg_bl, x.heightAboveWgs84EllipsoidM), axis=1))

    if dataset_name == 'train':
        # gt: lat/lngDeg -> x/y/z
        df_all['Xgt'], df_all['Ygt'], df_all['Zgt'] = zip(
            *df_all.apply(lambda x: WGS84_to_ECEF(x.latDeg_gt, x.lngDeg_gt, x.heightAboveWgs84EllipsoidM), axis=1))
        # copy lat/lngDeg
        lat_lng_df = df_all[['latDeg_gt',
                             'lngDeg_gt', 'latDeg_bl', 'lngDeg_bl']]
        df_all.drop(['latDeg_gt', 'lngDeg_gt', 'latDeg_bl',
                     'lngDeg_bl'], axis=1, inplace=True)
    elif dataset_name == 'test':
        # copy lat/lngDeg
        lat_lng_df = df_all[['latDeg_bl', 'lngDeg_bl']]
        df_all.drop(['latDeg_bl', 'lngDeg_bl', 'latDeg',
                     'lngDeg', ], axis=1, inplace=True)

    return lat_lng_df, df_all

# --------------------------------------------------------------------------


def prepare_df_train(df_all_train, window_size):
    '''
    prepare training dataset with all aixses
    window_size:

    '''
    tgt_df = df_all_train.copy()
    total_len = len(tgt_df)
    moving_times = total_len - window_size

    tgt_df.rename(columns={'yawDeg': 'yawZDeg', 'rollDeg': 'rollYDeg',
                           'pitchDeg': 'pitchXDeg'}, inplace=True)

    feature_cols = [f for f in list(tgt_df) if f not in ['Xgt', 'Ygt', 'Zgt']]

    # Historical Feature names
    hist_feats = []
    for time_flag in range(1, window_size + 1):
        for fn in feature_cols:
            hist_feats.append(fn + '_' + str(time_flag))

    # Window Sliding
    # t1 t2 t3 t4 t5 -> t6
    # t2 t3 t4 t5 t6 -> t7

    # Add historical data
    df_train = pd.DataFrame()
    features = []
    xs = []
    ys = []
    zs = []

    for start_idx in range(moving_times):
        feature_list = list()
        x_list = list()
        y_list = list()
        z_list = list()

        for window_idx in range(window_size):
            feature_list.extend(
                tgt_df[feature_cols].iloc[start_idx + window_idx, :].to_list())
        x_list.append(tgt_df['Xgt'].iloc[start_idx + window_size])
        y_list.append(tgt_df['Ygt'].iloc[start_idx + window_size])
        z_list.append(tgt_df['Zgt'].iloc[start_idx + window_size])

        features.append(feature_list)
        xs.extend(x_list)
        ys.extend(y_list)
        zs.extend(z_list)

    df_train = pd.DataFrame(features, columns=hist_feats)
    df_train['Xgt'] = xs
    df_train['Ygt'] = ys
    df_train['Zgt'] = zs

    # clean single-value feature: collectionName_[1-5]\phoneName_[1-5]
    tmp_feats = []
    for fn in list(df_train):
        if (fn.startswith('collectionName_') == False) and (fn.startswith('phoneName_') == False):
            tmp_feats.append(fn)
    df_train = df_train[tmp_feats]

    # clean time feature
    tmp_drop_feats = []
    for f in list(df_train):
        if (f.startswith('millisSinceGpsEpoch') == True) or (f.startswith('timeSinceFirstFixSeconds') == True) or (f.startswith('utcTimeMillis') == True):
            tmp_drop_feats.append(f)
    df_train.drop(tmp_drop_feats, axis=1, inplace=True)

    return df_train

# --------------------------------------------------------------------------
#


def prepare_df_test(df_all_test, window_size):
    '''prepare testing dataset with all aixses'''
    tgt_df = df_all_test.copy()
    total_len = len(tgt_df)
    moving_times = total_len - window_size

    tgt_df.rename(columns={'yawDeg': 'yawZDeg', 'rollDeg': 'rollYDeg',
                           'pitchDeg': 'pitchXDeg'}, inplace=True)

    feature_cols = [f for f in list(tgt_df) if f not in ['Xgt', 'Ygt', 'Zgt']]

    hist_feats = []
    for time_flag in range(1, window_size + 1):
        for fn in feature_cols:
            hist_feats.append(fn + '_' + str(time_flag))

    # t1 t2 t3 t4 t5 -> t6
    # t2 t3 t4 t5 t6 -> t7
    df_test = pd.DataFrame()
    features = []

    for start_idx in range(moving_times):
        feature_list = list()

        for window_idx in range(window_size):
            feature_list.extend(
                tgt_df[feature_cols].iloc[start_idx + window_idx, :].to_list())
        features.append(feature_list)

    df_test = pd.DataFrame(features, columns=hist_feats)

    tmp_feats = []
    for fn in list(df_test):
        if (fn.startswith('collectionName_') == False) and (fn.startswith('phoneName_') == False):
            tmp_feats.append(fn)
    df_test = df_test[tmp_feats]

    tmp_drop_feats = []
    for f in list(df_test):
        if (f.startswith('millisSinceGpsEpoch') == True) or (f.startswith('timeSinceFirstFixSeconds') == True) or (f.startswith('utcTimeMillis') == True) or (f.startswith('elapsedRealtimeNanos') == True):
            tmp_drop_feats.append(f)
    df_test.drop(tmp_drop_feats, axis=1, inplace=True)

    return df_test
#
#
# --------------------------------------------------------------------------


def remove_other_axis_feats(df_all, tgt_axis):
    '''unrelated-aixs features and uncalibrated features'''
    # Clean unrelated-aixs features
    all_imu_feats = ['UncalAccelXMps2', 'UncalAccelYMps2', 'UncalAccelZMps2',
                     'UncalGyroXRadPerSec', 'UncalGyroYRadPerSec', 'UncalGyroZRadPerSec',
                     'UncalMagXMicroT', 'UncalMagYMicroT', 'UncalMagZMicroT',
                     'ahrsX', 'ahrsY', 'ahrsZ',
                     'AccelXMps2', 'AccelYMps2', 'AccelZMps2',
                     'GyroXRadPerSec', 'GyroZRadPerSec', 'GyroYRadPerSec',
                     'MagXMicroT', 'MagYMicroT', 'MagZMicroT',
                     'yawZDeg', 'rollYDeg', 'pitchXDeg',
                     'Xbl', 'Ybl', 'Zbl']
    tgt_imu_feats = []
    for axis in ['X', 'Y', 'Z']:
        if axis != tgt_axis:
            for f in all_imu_feats:
                if f.find(axis) >= 0:
                    tgt_imu_feats.append(f)

    tmp_drop_feats = []
    for f in list(df_all):
        if f.split('_')[0] in tgt_imu_feats:
            tmp_drop_feats.append(f)

    tgt_df = df_all.drop(tmp_drop_feats, axis=1)

    # Clean uncalibrated features
    uncal_feats = [f for f in list(tgt_df) if f.startswith('Uncal') == True]
    tgt_df = tgt_df.drop(uncal_feats, axis=1)

    return tgt_df
# #  #
# =========================================================================
# end fo alvin
# =========================================================================


#  #
# =========================================================================
# magnetic sensor by museas
# =========================================================================
def gnss_log_to_dataframes(path):
    #    print(f'\033[36mGNSS log exists? \033[0m{Path(path).exists()}')
    #    print('Loading ' + path, flush=True)
    gnss_section_names = {'Raw', 'UncalAccel', 'UncalGyro',
                          'UncalMag', 'Fix', 'Status', 'OrientationDeg'}
    with open(path) as f_open:
        datalines = f_open.readlines()

    datas = {k: [] for k in gnss_section_names}
    gnss_map = {k: [] for k in gnss_section_names}
    for dataline in datalines:
        is_header = dataline.startswith('#')
        dataline = dataline.strip('#').strip().split(',')
        # skip over notes, version numbers, etc
        if is_header and dataline[0] in gnss_section_names:
            try:
                gnss_map[dataline[0]] = dataline[1:]
            except:
                pass
        elif not is_header:
            try:
                datas[dataline[0]].append(dataline[1:])
            except:
                pass
    results = dict()
    for k, v in datas.items():
        results[k] = pd.DataFrame(v, columns=gnss_map[k])
    # pandas doesn't properly infer types from these lists by default
    for k, df in results.items():
        for col in df.columns:
            if col == 'CodeType':
                continue
            try:
                results[k][col] = pd.to_numeric(results[k][col])
            except:
                pass
    return results

# --------------------------------------------------------------------------
# lowpass filter


def butter_lowpass(cutoff=2.5, fs=50.0, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff=2.5, fs=50.0, order=3):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# order = 3
# fs = 50.0
# cutoff = 2.5

# --------------------------------------------------------------------------
# # Offset correction
# refarence https://github.com/J-ROCKET-BOY/SS-Fitting

def SS_fit(data):

    x = data[:, [0]]
    y = data[:, [1]]
    z = data[:, [2]]

    data_len = len(x)

    x2 = np.power(x, 2)
    y2 = np.power(y, 2)
    z2 = np.power(z, 2)

    r1 = -x*(x2+y2+z2)
    r2 = -y*(x2+y2+z2)
    r3 = -z*(x2+y2+z2)
    r4 = -(x2+y2+z2)

    left = np.array([[np.sum(x2), np.sum(x*y), np.sum(x*z), np.sum(x)],
                     [np.sum(x*y), np.sum(y2), np.sum(y*z), np.sum(y)],
                     [np.sum(x*z), np.sum(y*z), np.sum(z2), np.sum(z)],
                     [np.sum(x), np.sum(y), np.sum(z), data_len]])

    right = np.array([np.sum(r1),
                      np.sum(r2),
                      np.sum(r3),
                      np.sum(r4)])

    si = np.dot(np.linalg.inv(left), right)

    x0 = (-1/2) * si[0]
    y0 = (-1/2) * si[1]
    z0 = (-1/2) * si[2]

    return np.array([x0, y0, z0])

# --------------------------------------------------------------------------


def vincenty_inverse(lat1, lon1, lat2, lon2):

    # Not advanced
    if isclose(lat1, lat2) and isclose(lon1, lon2):
        return False

    # WGS84
    a = 6_378_137.0
    f = 1 / 298.257223563
    b = (1 - f) * a

    lat_1 = atan((1 - f) * tan(radians(lat1)))
    lat_2 = atan((1 - f) * tan(radians(lat2)))

    lon_diff = radians(lon2) - radians(lon1)
    lam = lon_diff

    for i in range(1000):
        sin_lam = sin(lam)
        cos_lam = cos(lam)

        sin_sig = sqrt((cos(lat_2) * sin_lam) ** 2
                       + (cos(lat_1) * sin(lat_2) - sin(lat_1) * cos(lat_2) * cos_lam) ** 2)

        cos_sig = sin(lat_1) * sin(lat_2) + cos(lat_1) * cos(lat_2) * cos_lam

        sigma = atan2(sin_sig, cos_sig)

        sin_alf = cos(lat_1) * cos(lat_2) * sin_lam / sin_sig

        cos_2alf = 1 - sin_alf ** 2
        cos_2sm = cos_sig - 2 * sin(lat_1) * sin(lat_2) / cos_2alf

        C = f / 16 * cos_2alf * (4 + f * (4 - 3 * cos_2alf))
        lam_p = lam

        lam = lon_diff + (1 - C) * f * sin_alf * (sigma + C * sin_sig * (cos_2sm + C * cos_sig
                                                                         * (-1 + 2 * cos_2sm ** 2)))

        if abs(lam - lam_p) <= 1e-12:
            break
    else:
        return None

    alf = atan2(cos(lat_2) * sin_lam,
                cos(lat_1) * sin(lat_2) - sin(lat_1) * cos(lat_2) * cos_lam)

    if alf < 0:
        alf = alf + pi * 2

    return degrees(alf)
# --------------------------------------------------------------------------


def calc3(row):
    deg = - degrees(atan2(-1*row['calc2'], row['calc1']))
    if deg < 0:
        deg += 360
    return deg


# --------------------------------------------------------------------------
# for n, logg_path in enumerate(logg_paths):
# def get_prev(logg_path):

def get_acce_test(sample, df_test_ex):

    #    path = logg_path
    #    gt_path = path.split("Pixel4_")[0]+"ground_truth.csv"

    #    gt_df = pd.read_csv(gt_path)
    #    sample = gnss_log_to_dataframes(str(path))
    # sample is a dictionary

    order = 3
    fs = 50.0
    cutoff = 2.5

    acce_df = sample["UncalAccel"]
    mag_df = sample["UncalMag"]

    if acce_df.shape[0] == 0:
        return acce_df, mag_df

    msge = "millisSinceGpsEpoch"
    acce_df[msge] = acce_df["utcTimeMillis"] - 315964800000
    mag_df[msge] = mag_df["utcTimeMillis"] - 315964800000

    #     acce filtering and smooting
#    print(f'\033[33macce_shape 2 {acce_df.shape}')
#    print(f'\033[33mmag_shape 2 {mag_df.shape}\033[0m')

    acce_df["global_x"] = acce_df["UncalAccelZMps2"]
    acce_df["global_y"] = acce_df["UncalAccelXMps2"]
    acce_df["global_z"] = acce_df["UncalAccelYMps2"]

    acce_df["x_f"] = butter_lowpass_filter(
        acce_df["global_x"], cutoff, fs, order)
    acce_df["y_f"] = butter_lowpass_filter(
        acce_df["global_y"], cutoff, fs, order)
    acce_df["z_f"] = butter_lowpass_filter(
        acce_df["global_z"], cutoff, fs, order)

    #     acce filtering and smooting
#    print(f'\033[33macce_shape 3 {acce_df.shape}')
#    print(f'\033[33mmag_shape 3 {mag_df.shape}\033[0m')

    smooth_range = 1000

    acce_df["x_f"] = acce_df["x_f"].rolling(
        smooth_range, center=True, min_periods=1).mean()
    acce_df["y_f"] = acce_df["y_f"].rolling(
        smooth_range, center=True, min_periods=1).mean()
    acce_df["z_f"] = acce_df["z_f"].rolling(
        smooth_range, center=True, min_periods=1).mean()

    #     magn filtering and smooting , offset correction

    mag_df["global_mx"] = mag_df["UncalMagZMicroT"]
    mag_df["global_my"] = mag_df["UncalMagYMicroT"]
    mag_df["global_mz"] = mag_df["UncalMagXMicroT"]

    smooth_range = 1000

    mag_df["global_mx"] = mag_df["global_mx"].rolling(
        smooth_range, min_periods=1).mean()
    mag_df["global_my"] = mag_df["global_mz"].rolling(
        smooth_range, min_periods=1).mean()
    mag_df["global_mz"] = mag_df["global_my"].rolling(
        smooth_range, min_periods=1).mean()

#    print(f'acce_shape {acce_df.shape}')
#    print(f'mag_shape {mag_df.shape}')

    offset = SS_fit(np.array(mag_df[["global_mx", "global_my", "global_mz"]]))
    mag_df["global_mx"] = (mag_df["global_mx"] - offset[0])*-1
    mag_df["global_my"] = mag_df["global_my"] - offset[1]
    mag_df["global_mz"] = mag_df["global_mz"] - offset[2]

    # # no smoothing
    # mag_df["global_mx"] = mag_df["UncalMagZMicroT"]
    # mag_df["global_my"] = mag_df["UncalMagYMicroT"]
    # mag_df["global_mz"] = mag_df["UncalMagXMicroT"]

    #     merge the value of the one with the closest time
    acce_df[msge] = acce_df[msge]//1000 + 10
    mag_df[msge] = mag_df[msge]//1000 + 10

    df_test_ex[msge] = df_test_ex[msge]//1000

#    print(f'shape of acce_df 1 {acce_df.shape}')
#    print(f'shape of mag_df 1 {mag_df.shape}')

    acce_df = pd.merge_asof(acce_df.sort_values(msge),
                            mag_df[["global_mx", "global_my",
                                    "global_mz", msge]].sort_values(msge),
                            on=msge, direction='nearest')

#    print(f'shape of acce_df 2 {acce_df.shape}')
#    print(f'shape of mag_df 2 {mag_df.shape}')

    acce_df = pd.merge_asof(df_test_ex[[msge, "latDeg", "lngDeg"]].sort_values(msge),
                            acce_df[[msge, "x_f", "y_f", "z_f", "global_mx",
                                     "global_my", "global_mz"]].sort_values(msge),
                            on=msge, direction='nearest')

#    print(f'shape of acce_df 3 {acce_df.shape}')
    return acce_df, mag_df

    # # no smoothing  ************************
    # acce_df["x_f"] = acce_df["UncalAccelZMps2"]
    # acce_df["y_z"] = acce_df["UncalAccelXMps2"]
    # acce_df["z_z"] = acce_df["UncalAccelYMps2"]
    # # no smoothing  ************************

# --------------------------------------------------------------------------
# for n, logg_path in enumerate(logg_paths):
# def get_prev(logg_path):


def get_acce_train(sample, gt_df):

    #    path = logg_path
    #    gt_path = path.split("Pixel4_")[0]+"ground_truth.csv"

    #    gt_df = pd.read_csv(gt_path)
    #    sample = gnss_log_to_dataframes(str(path))
    # sample is a dictionary

    order = 3
    fs = 50.0
    cutoff = 2.5

    acce_df = sample["UncalAccel"]
    mag_df = sample["UncalMag"]

    if acce_df.shape[0] == 0:
        return acce_df, mag_df

    msge = "millisSinceGpsEpoch"
    acce_df[msge] = acce_df["utcTimeMillis"] - 315964800000
    mag_df[msge] = mag_df["utcTimeMillis"] - 315964800000

    #     acce filtering and smooting

    acce_df["global_x"] = acce_df["UncalAccelZMps2"]
    acce_df["global_y"] = acce_df["UncalAccelXMps2"]
    acce_df["global_z"] = acce_df["UncalAccelYMps2"]

    acce_df["x_f"] = butter_lowpass_filter(
        acce_df["global_x"], cutoff, fs, order)
    acce_df["y_f"] = butter_lowpass_filter(
        acce_df["global_y"], cutoff, fs, order)
    acce_df["z_f"] = butter_lowpass_filter(
        acce_df["global_z"], cutoff, fs, order)

    smooth_range = 1000

    acce_df["x_f"] = acce_df["x_f"].rolling(
        smooth_range, center=True, min_periods=1).mean()
    acce_df["y_f"] = acce_df["y_f"].rolling(
        smooth_range, center=True, min_periods=1).mean()
    acce_df["z_f"] = acce_df["z_f"].rolling(
        smooth_range, center=True, min_periods=1).mean()

#    print(f'shape of acce_df 0 {acce_df.shape}')
    # # no smoothing  ************************
    # acce_df["x_f"] = acce_df["UncalAccelZMps2"]
    # acce_df["y_z"] = acce_df["UncalAccelXMps2"]
    # acce_df["z_z"] = acce_df["UncalAccelYMps2"]
    # # no smoothing  ************************

    #     magn filtering and smooting , offset correction

    mag_df["global_mx"] = mag_df["UncalMagZMicroT"]
    mag_df["global_my"] = mag_df["UncalMagYMicroT"]
    mag_df["global_mz"] = mag_df["UncalMagXMicroT"]

    smooth_range = 1000

    mag_df["global_mx"] = mag_df["global_mx"].rolling(
        smooth_range, min_periods=1).mean()
    mag_df["global_my"] = mag_df["global_mz"].rolling(
        smooth_range, min_periods=1).mean()
    mag_df["global_mz"] = mag_df["global_my"].rolling(
        smooth_range, min_periods=1).mean()

    offset = SS_fit(np.array(mag_df[["global_mx", "global_my", "global_mz"]]))
    mag_df["global_mx"] = (mag_df["global_mx"] - offset[0])*-1
    mag_df["global_my"] = mag_df["global_my"] - offset[1]
    mag_df["global_mz"] = mag_df["global_mz"] - offset[2]

    # # no smoothing
    # mag_df["global_mx"] = mag_df["UncalMagZMicroT"]
    # mag_df["global_my"] = mag_df["UncalMagYMicroT"]
    # mag_df["global_mz"] = mag_df["UncalMagXMicroT"]

    #     merge the value of the one with the closest time
    acce_df[msge] = acce_df[msge]//1000 + 10
    mag_df[msge] = mag_df[msge]//1000 + 10
    gt_df[msge] = gt_df[msge]//1000
#    print(f'shape of acce_df 1 {acce_df.shape}')
#    print(f'shape of mag_df 1 {mag_df.shape}')

    acce_df = pd.merge_asof(acce_df.sort_values(msge),
                            mag_df[["global_mx", "global_my",
                                    "global_mz", msge]].sort_values(msge),
                            on=msge, direction='nearest')

#    print(f'shape of acce_df 2 {acce_df.shape}')
#    print(f'shape of mag_df 2 {mag_df.shape}')
    acce_df = pd.merge_asof(gt_df[[msge, "latDeg", "lngDeg", 'speedMps', 'courseDegree']].sort_values(msge),
                            acce_df[[msge, "x_f", "y_f", "z_f", "global_mx",
                                     "global_my", "global_mz"]].sort_values(msge),
                            on=msge, direction='nearest')

#    print(f'shape of acce_df 3 {acce_df.shape}')
    # =====================================================
    #     as a sensor value when stopped

    start_mean_range = 10

    x_start_mean = acce_df[:start_mean_range]["x_f"].mean()
    y_start_mean = acce_df[:start_mean_range]["y_f"].mean()
    z_start_mean = acce_df[:start_mean_range]["z_f"].mean()

    #     roll and picth, device tilt

    r = atan(y_start_mean/z_start_mean)
    p = atan(x_start_mean/(y_start_mean**2 + z_start_mean**2)**0.5)

    #     calculation　degrees

    acce_df["calc1"] = acce_df["global_mx"] * \
        cos(p) + acce_df["global_my"]*sin(r) * \
        sin(p) + acce_df["global_mz"]*sin(p)*cos(r)

    acce_df["calc2"] = acce_df["global_mz"] * \
        sin(r) - acce_df["global_my"]*cos(r)

    acce_df["calc_deg"] = acce_df.apply(calc3, axis=1)

    #     degrees with lat and lng by ground truth

    len_df = len(acce_df)
    acce_df["deg"] = 0

    gt_lat_prev = 0
    gt_lng_prev = 0

    for i in range(1, len_df):
        if i > 1:
            res = vincenty_inverse(
                gt_lat_prev, gt_lng_prev, acce_df["latDeg"].loc[i], acce_df["lngDeg"].loc[i])
            if res:
                acce_df["deg"].loc[i] = res
            else:
                if i > 0:
                    acce_df["deg"].loc[i] = acce_df["deg"].loc[i-1]
                else:
                    acce_df["deg"].loc[i] = 0

        gt_lat_prev = acce_df["latDeg"].loc[i]
        gt_lng_prev = acce_df["lngDeg"].loc[i]

#     print('finished.')

    return acce_df, mag_df

# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# =========================================================================
# visualization
# =========================================================================


def visualize_traffic(df, center, zoom=9):
    fig = px.scatter_mapbox(df,

                            # Here, plotly gets, (x,y) coordinates
                            lat="latDeg",
                            lon="lngDeg",

                            # Here, plotly detects color of series
                            color="phoneName",
                            labels="phoneName",

                            zoom=zoom,
                            center=center,
                            height=1024,
                            width=1024)
    fig.update_layout(mapbox_style='stamen-terrain')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="GPS trafic")
    fig.show()

# --------------------------------------------------------------------------
# compare baseline and ground truth


def visualize_comp(df, center, zoom=9):
    #    df = df_routes_base_true
    #    df['latDeg']

    trace_base = go.Scattermapbox(
        lat=df['latDeg'],
        lon=df['lngDeg'],
        mode='markers',
        opacity=0.8,
        #        marker=dict(size=12, color='orange'),
        marker=dict(size=12, color='brown'),
        text=df[['phoneName', 'millisSinceGpsEpoch']],
        hoverinfo='text'
    )

    trace_true = go.Scattermapbox(
        lat=df['latDeg_true'],
        lon=df['lngDeg_true'],
        mode='markers',
        marker=dict(size=8, color='blue'),
        text=df[['phoneName', 'millisSinceGpsEpoch']],
        hoverinfo='text'
    )

#    data = [trace_true, trace_base]
    data = [trace_true, trace_base]

    layout = go.Layout(hovermode='closest',
                       mapbox=dict(style='stamen-terrain',
                                   center=center,
                                   zoom=zoom),
                       width=1024*2, height=1024*2)

    fig = go.Figure(data=data, layout=layout)

    fig.show()

# --------------------------------------------------------------------------


# def visualize_before_after(df_before, df_after):
#     #    df = df_routes_base_true
#     #    df['latDeg']
#     zoom = 16
#     center = dict(lat=df_after.loc[0, 'latDeg'],
#                   lon=df_after.loc[0, 'lngDeg'])

#     trace_before = go.Scattermapbox(
#         lat=df_before['latDeg'],
#         lon=df_before['lngDeg'],
#         mode='markers',
#         marker=dict(size=8, color='darkblue'),
#         text=df_before[['phoneName', 'millisSinceGpsEpoch']],
#         hoverinfo='text'
#     )

#     trace_after = go.Scattermapbox(
#         lat=df_after['latDeg'],
#         lon=df_after['lngDeg'],
#         mode='markers',
#         marker=dict(size=8, color='red'),
#         text=df_after[['phoneName', 'millisSinceGpsEpoch']],
#         hoverinfo='text'
#     )

#     data = [trace_before, trace_after]

#     layout = go.Layout(hovermode='closest',
#                        mapbox=dict(style='stamen-terrain',
#                                    center=center,
#                                    zoom=zoom),
#                        width=2048, height=2048)

#     fig = go.Figure(data=data, layout=layout)
#     fig.show()


# --------------------------------------------------------------------------


def visualize_comp3(df, center, zoom=9):
    #    df = df_routes_base_true
    #    df['latDeg']

    trace_clipped = go.Scattermapbox(
        lat=df['latDeg_clipped'],
        lon=df['lngDeg_clipped'],
        mode='markers',
        marker=dict(size=8, color='red'),
        text=df[['phoneName', 'millisSinceGpsEpoch']],
        hoverinfo='text'
    )

    trace_true = go.Scattermapbox(
        lat=df['latDeg_true'],
        lon=df['lngDeg_true'],
        mode='markers',
        marker=dict(size=8, color='blue'),
        text=df[['phoneName', 'millisSinceGpsEpoch']],
        hoverinfo='text'
    )

    trace_base = go.Scattermapbox(
        lat=df['latDeg'],
        lon=df['lngDeg'],
        mode='markers',
        opacity=0.8,
        marker=dict(size=20, color='orange'),
        text=df[['phoneName', 'millisSinceGpsEpoch']],
        hoverinfo='text'
    )

#    data = [trace_true, trace_base]
#    data = [trace_base, trace_clipped]
    data = [trace_base, trace_clipped, trace_true]

    layout = go.Layout(hovermode='closest',
                       mapbox=dict(style='stamen-terrain',
                                   center=center,
                                   zoom=zoom),
                       width=1024*2, height=1024*2)

    fig = go.Figure(data=data, layout=layout)

    fig.show()

# =========================================================================
#  by carl mcbride ellis
# --------------------------------------------------------------------------
#
# df['x'], df['y'], df['z'] = zip(*df.apply(lambda x: WGS84_to_ECEF(x.latDeg, x.lngDeg, x.heightAboveWgs84EllipsoidM), axis=1))


def WGS84_to_ECEF(lat, lon, alt):
    # convert to radians
    rad_lat = lat * (np.pi / 180.0)
    rad_lon = lon * (np.pi / 180.0)
    a = 6378137.0
    # f is the flattening factor
    finv = 298.257223563
    f = 1 / finv
    # e is the eccentricity
    e2 = 1 - (1 - f) * (1 - f)
    # N is the radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(rad_lat) * np.sin(rad_lat))
    x = (N + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (N + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = (N * (1 - e2) + alt) * np.sin(rad_lat)
    return x, y, z


# --------------------------------------------------------------------------
transformer = pyproj.Transformer.from_crs(
    {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
    {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},)


def ECEF_to_WGS84(x, y, z):
    lon, lat, alt = transformer.transform(x, y, z, radians=False)
    return lon, lat, alt


# =========================================================================
# Kalman filter by Trinh
# --------------------------------------------------------------------------
def make_shifted_matrix(vec):
    matrix = []
    size = len(vec)
    for i in range(size):
        row = [0] * i + vec[:size-i]
        matrix.append(row)
    return np.array(matrix)


def make_state_vector(T, size):
    vector = [1, 0]
    step = 2
    for i in range(size - 2):
        if i % 2 == 0:
            vector.append(T)
            T *= T / step
            step += 1
        else:
            vector.append(0)
    return vector


def make_noise_vector(noise, size):
    noise_vector = []
    for i in range(size):
        if i > 0 and i % 2 == 0:
            noise *= 0.5
        noise_vector.append(noise)
    return noise_vector


def make_kalman_filter(T, size, noise, obs_noise):
    vec = make_state_vector(T, size)
    state_transition = make_shifted_matrix(vec)
    process_noise = np.diag(make_noise_vector(
        noise, size)) + np.ones(size) * 1e-9
    observation_model = np.array(
        [[1] + [0] * (size - 1), [0, 1] + [0] * (size - 2)])
    observation_noise = np.diag([obs_noise] * 2) + np.ones(2) * 1e-9
    kf = simdkalman.KalmanFilter(
        state_transition=state_transition,
        process_noise=process_noise,
        observation_model=observation_model,
        observation_noise=observation_noise)
    return kf

# =========================================================================
# Kalman filter by Marcin Bodych
# =========================================================================


def apply_kf_smoothing(df, kf_):
    unique_paths = df[['collectionName', 'phoneName']
                      ].drop_duplicates().to_numpy()
#    for collection, phone in tqdm(unique_paths):

    for collection, phone in unique_paths:
        cond = np.logical_and(df['collectionName'] ==
                              collection, df['phoneName'] == phone)
        data = df[cond][['latDeg', 'lngDeg']].to_numpy()
        data = data.reshape(1, len(data), 2)
        smoothed = kf_.smooth(data)
        df.loc[cond, 'latDeg'] = smoothed.states.mean[0, :, 0]
        df.loc[cond, 'lngDeg'] = smoothed.states.mean[0, :, 1]

    return df

# =========================================================================


def rbt(nline, cols):
    print(df_routes_base_true.loc[:, cols].head(nline))
#    print(df_routes_base_true.loc[10:20, cols].head(nline))
#    print(df_routes_base_true.loc[20:30, cols].head(nline))
#    print(df_routes_base_true.loc[30:, cols].head(nline))


def rbtx(df):
    nline = 3
    # show all columns
    print('----------------')
    print(df.iloc[:, :10].head(nline))
    print('----------------')
    print(df.iloc[:, 10:20].head(nline))
    print('----------------')
    print(df.iloc[:, 20:28].head(nline))
    print('----------------')
    print(df.iloc[:, 28:38].head(nline))
    print('----------------')
    print(df.iloc[:, 38:].head(nline))


def rbt_all(nline):
    # show all columns
    print('----------------')
    print(df_routes_base_true.iloc[:, :10].head(nline))
    print('----------------')
    print(df_routes_base_true.iloc[:, 10:20].head(nline))
    print('----------------')
    print(df_routes_base_true.iloc[:, 20:30].head(nline))
    print('----------------')
    print(df_routes_base_true.iloc[:, 30:40].head(nline))
    print('----------------')
    print(df_routes_base_true.iloc[:, 40:].head(nline))

# =========================================================================


def error_no_correction(df):

    msge = 'millisSinceGpsEpoch'
    df = df.groupby(msge).mean()

    true_ref = list(zip(df['latDeg_true'], df['lngDeg_true']))
    measured = list(zip(df['latDeg'], df['lngDeg']))

#    x0 = [vincenty(p1, p2) for p1, p2 in zip(measured, true_ref)]
#    x0 = np.array(x0, dtype=np.float64) * 1000.0
    x0 = pd.DataFrame([vincenty(p1, p2) for p1, p2 in zip(measured, true_ref)])
    x0 = x0 * 1000.0

#    return np.nanmean(x0)
    return (x0.quantile(.50) + x0.quantile(.95)) * 0.5, x0.quantile(.50), x0.quantile(.95)


# -------------------------


def error_raw_correction(df):
    msge = 'millisSinceGpsEpoch'
    df = df.groupby(msge).mean()

    dlat_mean = df['dlat'].mean()
    dlng_mean = df['dlng'].mean()

    df['latDeg_corrected'] = df['latDeg'] - dlat_mean
    df['lngDeg_corrected'] = df['lngDeg'] - dlng_mean

    true_ref = list(zip(df['latDeg_true'], df['lngDeg_true']))
    corrected = list(zip(df['latDeg_corrected'], df['lngDeg_corrected']))

    x1 = pd.DataFrame([vincenty(p1, p2)
                       for p1, p2 in zip(corrected, true_ref)])
    x1 = x1 * 1000.0
#    x1 = [vincenty(p1, p2) for p1, p2 in zip(corrected, true_ref)]
#    x1 = np.array(x1, dtype=np.float64) * 1000.0

#    return np.nanmean(x1), dlat_mean, dlng_mean
    return (x1.quantile(.50) + x1.quantile(.95)) * 0.5, dlat_mean, dlng_mean

# -------------------------


def error_conditional_correction(df):
    msge = 'millisSinceGpsEpoch'
    df = df.groupby(msge).mean()

    df_sta = df[df['speedMps'] == 0.0]  # stationary
    df_mot = df[df['speedMps'] != 0.0]  # in motion

    idx_sta = df['speedMps'] == 0.0  # stationary
    idx_mot = df['speedMps'] != 0.0  # in motion

    dlat_sta, dlng_sta = (df_sta['dlat'].mean(), df_sta['dlng'].mean())
    dlat_mot, dlng_mot = (df_mot['dlat'].mean(), df_mot['dlng'].mean())

    df.loc[idx_sta, 'latDeg_cond'] = df['latDeg'] - dlat_sta
    df.loc[idx_sta, 'lngDeg_cond'] = df['lngDeg'] - dlng_sta

    df.loc[idx_mot, 'latDeg_cond'] = df['latDeg'] - dlat_mot
    df.loc[idx_mot, 'lngDeg_cond'] = df['lngDeg'] - dlng_mot

    true_ref = list(zip(df['latDeg_true'], df['lngDeg_true']))
    corrected = list(zip(df['latDeg_cond'], df['lngDeg_cond']))

    x3 = pd.DataFrame([vincenty(p1, p2)
                       for p1, p2 in zip(corrected, true_ref)])
    x3 = x3 * 1000.0

#    return np.nanmean(x3), dlat_sta, dlng_sta, dlat_mot, dlng_mot
    return (x3.quantile(.50) + x3.quantile(.95)) * 0.5, dlat_sta, dlng_sta, dlat_mot, dlng_mot

# ===============================


def score_imu(df_pred, df_gt):

    pred = list(zip(df_pred['latDeg'], df_pred['lngDeg']))
    true = list(zip(df_gt['latDeg'], df_gt['lngDeg']))

    len_pred = len(pred)
    len_true = len(true)

    if len_pred < len_true:
        true = true[-len_pred:]

    else:
        pred = pred[-len_true:]

#    print(len(pred), len(true))

    x0 = pd.DataFrame([vincenty(p1, p2) for p1, p2 in zip(pred, true)])
    x0 = x0 * 1000.0

    x5 = x0.quantile(.50).values[0]
    x9 = x0.quantile(.95).values[0]
    xm = (x5 + x9) * 0.5

    return xm, x5, x9

# -------------------------
#


def score_no_correction(df, clipped=False):

    msge = 'millisSinceGpsEpoch'
    df = df.groupby(msge).mean()

    true_ref = list(zip(df['latDeg_true'], df['lngDeg_true']))

    if clipped:
        measured = list(zip(df['latDeg_clipped'], df['lngDeg_clipped']))
    else:
        measured = list(zip(df['latDeg'], df['lngDeg']))

#    x0 = [vincenty(p1, p2) for p1, p2 in zip(measured, true_ref)]
#    x0 = np.array(x0, dtype=np.float64) * 1000.0
    x0 = pd.DataFrame([vincenty(p1, p2) for p1, p2 in zip(measured, true_ref)])
    x0 = x0 * 1000.0

    x5 = x0.quantile(.50).values[0]
    x9 = x0.quantile(.95).values[0]
    xm = (x5 + x9) * 0.5

#    return np.nanmean(x0)
#    return (x0.quantile(.50) + x0.quantile(.95)) * 0.5, x0.quantile(.50), x0.quantile(.95)
    return xm, x5, x9

# -------------------------


def score_raw_correction(df, clipped=False):
    msge = 'millisSinceGpsEpoch'
    df = df.groupby(msge).mean()

    dlat_mean = df['dlat'].mean()
    dlng_mean = df['dlng'].mean()

    if clipped:

        df['latDeg_corrected'] = df['latDeg_clipped'] - dlat_mean
        df['lngDeg_corrected'] = df['lngDeg_clipped'] - dlng_mean

    else:
        df['latDeg_corrected'] = df['latDeg'] - dlat_mean
        df['lngDeg_corrected'] = df['lngDeg'] - dlng_mean

    true_ref = list(zip(df['latDeg_true'], df['lngDeg_true']))
    corrected = list(zip(df['latDeg_corrected'], df['lngDeg_corrected']))

    x1 = pd.DataFrame([vincenty(p1, p2)
                       for p1, p2 in zip(corrected, true_ref)])
    x1 = x1 * 1000.0

    x5 = x1.quantile(.50).values[0]
    x9 = x1.quantile(.95).values[0]
    xm = (x5 + x9) * 0.5

#    x1 = [vincenty(p1, p2) for p1, p2 in zip(corrected, true_ref)]
#    x1 = np.array(x1, dtype=np.float64) * 1000.0

#    return np.nanmean(x1), dlat_mean, dlng_mean
#    return (x1.quantile(.50) + x1.quantile(.95)) * 0.5, x1.quantile(.50), x1.quantile(.95)
    return xm, x5, x9
# -------------------------


def score_conditional_correction(df, clipped=False):
    msge = 'millisSinceGpsEpoch'
    df = df.groupby(msge).mean()

    df_sta = df[df['speedMps'] == 0.0]  # stationary
    df_mot = df[df['speedMps'] != 0.0]  # in motion

    idx_sta = df['speedMps'] == 0.0  # stationary
    idx_mot = df['speedMps'] != 0.0  # in motion

    dlat_sta, dlng_sta = (df_sta['dlat'].mean(), df_sta['dlng'].mean())
    dlat_mot, dlng_mot = (df_mot['dlat'].mean(), df_mot['dlng'].mean())

    if clipped:
        df.loc[idx_sta, 'latDeg_cond'] = df['latDeg_clipped'] - dlat_sta
        df.loc[idx_sta, 'lngDeg_cond'] = df['lngDeg_clipped'] - dlng_sta

        df.loc[idx_mot, 'latDeg_cond'] = df['latDeg_clipped'] - dlat_mot
        df.loc[idx_mot, 'lngDeg_cond'] = df['lngDeg_clipped'] - dlng_mot

    else:
        df.loc[idx_sta, 'latDeg_cond'] = df['latDeg'] - dlat_sta
        df.loc[idx_sta, 'lngDeg_cond'] = df['lngDeg'] - dlng_sta

        df.loc[idx_mot, 'latDeg_cond'] = df['latDeg'] - dlat_mot
        df.loc[idx_mot, 'lngDeg_cond'] = df['lngDeg'] - dlng_mot

    true_ref = list(zip(df['latDeg_true'], df['lngDeg_true']))
    corrected = list(zip(df['latDeg_cond'], df['lngDeg_cond']))

    x3 = pd.DataFrame([vincenty(p1, p2)
                       for p1, p2 in zip(corrected, true_ref)])
    x3 = x3 * 1000.0

    x5 = x3.quantile(.50).values[0]
    x9 = x3.quantile(.95).values[0]
    xm = (x5 + x9) * 0.5


#    return np.nanmean(x3), dlat_sta, dlng_sta, dlat_mot, dlng_mot
#    return (x3.quantile(.50) + x3.quantile(.95)) * 0.5, x3.quantile(.50), x3.quantile(.95)
    return xm, x5, x9
# =========================


def extract_baseline(df_base, df_routes):

    # print(df_base.shape)
    # print(df_routes.shape)
    # print(df_routes.head(3))
    # print(df_base.head(3))

    collection = df_routes['collectionName'].unique()[0]
    phone = df_routes['phoneName'].unique()[0]

    msge_min = df_routes['millisSinceGpsEpoch'].min() - 8000
    msge_max = df_routes['millisSinceGpsEpoch'].max() + 8000

    b_idx = (df_base['collectionName'] == collection) & (
        df_base['phoneName'] == phone) & (
        df_base['millisSinceGpsEpoch'] >= msge_min) & (
        df_base['millisSinceGpsEpoch'] <= msge_max)

#    print(b_idx.sum())

    df_ex = df_base[b_idx]
#    print(df_ex.head(3))

    df_ex.reset_index(inplace=True, drop=True)

    idx = b_idx[b_idx].index.values

    return df_ex, idx
# -------------------------


def outlier_removal(df, sig_cutoff):
    # from df basline
    diff_lat = df['latDeg'] - df['latDeg_nn']
    diff_lng = df['lngDeg'] - df['lngDeg_nn']

#    print(diff_lat.std(), diff_lng.std())
#    sig_cutoff = 2.0<
    idx_lat = abs(diff_lat) > (diff_lat.std() * sig_cutoff)
    idx_lng = abs(diff_lng) > (diff_lng.std() * sig_cutoff)

#    (idx_lng).sum()
#    (idx_lat).sum()
    idx_rep = idx_lat + idx_lng

#    df.loc[idx_rep, 'latDeg_clipped'] = df.loc[idx_rep, 'latDeg_nn']
#    df.loc[idx_rep, 'lngDeg_clipped'] = df.loc[idx_rep, 'lngDeg_nn']

    # overwrite
#    df.iloc[idx_rep]['latDeg'] = df.iloc[idx_rep]['latDeg_nn']
#    df.iloc[idx_rep]['lngDeg'] = df.iloc[idx_rep]['lngDeg_nn']
#    pdb.set_trace()

    df.loc[idx_rep, 'latDeg'] = df.loc[idx_rep, 'latDeg_nn']
    df.loc[idx_rep, 'lngDeg'] = df.loc[idx_rep, 'lngDeg_nn']

    return df

# --------------------------------------------------------------------------


def nn_clipped(df, sig_cutoff):
    # replace_with_nn + outlider removal

    # from df basline
    diff_lat = df['latDeg'] - df['latDeg_nn']
    diff_lng = df['lngDeg'] - df['lngDeg_nn']

#    print(diff_lat.std(), diff_lng.std())
#    sig_cutoff = 2.0
    idx_lat = abs(diff_lat) > (diff_lat.std() * sig_cutoff)
    idx_lng = abs(diff_lng) > (diff_lng.std() * sig_cutoff)

#    (idx_lng).sum()
#    (idx_lat).sum()
    idx_rep = idx_lat + idx_lng

    df['latDeg_clipped'] = df['latDeg']
    df['lngDeg_clipped'] = df['lngDeg']

    df.loc[idx_rep, 'latDeg_clipped'] = df.loc[idx_rep, 'latDeg_nn']
    df.loc[idx_rep, 'lngDeg_clipped'] = df.loc[idx_rep, 'lngDeg_nn']

    # overwrite
#    df.iloc[idx_rep]['latDeg'] = df.iloc[idx_rep]['latDeg_nn']
#    df.iloc[idx_rep]['lngDeg'] = df.iloc[idx_rep]['lngDeg_nn']
#    pdb.set_trace()

#    df.loc[idx_rep, 'latDeg'] = df.loc[idx_rep, 'latDeg_nn']
#    df.loc[idx_rep, 'lngDeg'] = df.loc[idx_rep, 'lngDeg_nn']

    return df


# --------------------------------------------------------------------------
    # latx = df_test_bs.loc[i_start:i_end, 'latDeg'].values
    # latx, _, _ = sigmaclip(latx, 2.5, 2.5)
    # lat_mean = np.mean(latx)


def clip_baseline(df, nn, d_cutoff, inplace=True, abs_dist=False):

    lat_nn = []
    lng_nn = []

    sig = 1.5
    for m in df['millisSinceGpsEpoch']:
        x = np.abs(df['millisSinceGpsEpoch'] - m)
        idx = np.argsort(x)[1:nn]

        latx = df.iloc[idx]['latDeg'].values
        latx, _, _ = sigmaclip(latx, sig, sig)

        lngx = df.iloc[idx]['lngDeg'].values
        lngx, _, _ = sigmaclip(lngx, sig, sig)

        lat_nn.append(np.mean(latx))
        lng_nn.append(np.mean(lngx))

    # -----------------------
    # add new columns
    df['latDeg_nn'] = lat_nn
    df['lngDeg_nn'] = lng_nn

    # from df basline
    diff_lat = df['latDeg'] - df['latDeg_nn']
    diff_lng = df['lngDeg'] - df['lngDeg_nn']

    if abs_dist:

        idx_lat = abs(diff_lat) > d_cutoff
        idx_lng = abs(diff_lng) > d_cutoff

    else:
        idx_lat = abs(diff_lat) > (diff_lat.std() * d_cutoff)
        idx_lng = abs(diff_lng) > (diff_lng.std() * d_cutoff)

    idx_rep = idx_lat + idx_lng

    if inplace == False:

        df['latDeg_clipped'] = df['latDeg']
        df['lngDeg_clipped'] = df['lngDeg']

        # add new columns
        df.loc[idx_rep, 'latDeg_clipped'] = df.loc[idx_rep, 'latDeg_nn']
        df.loc[idx_rep, 'lngDeg_clipped'] = df.loc[idx_rep, 'lngDeg_nn']

    else:
        # override
        df.loc[idx_rep, 'latDeg'] = df.loc[idx_rep, 'latDeg_nn']
        df.loc[idx_rep, 'lngDeg'] = df.loc[idx_rep, 'lngDeg_nn']

    return df

# --------------------------------------------------------------------------
# merge using msge, but ignoring lastone digit


def merge_msge(df1, df2, suffixes=('', '_base')):
    msge = 'millisSinceGpsEpoch'
    df1['msge'] = df1[msge].astype(str).str[:-2].astype(np.int64)
    df2['msge'] = df2[msge].astype(str).str[:-2].astype(np.int64)

    df = df1.merge(df2, how='left', on=[
        #    df = df1.merge(df2, how='inner', on=[
        'collectionName', 'phoneName', 'msge'], suffixes=suffixes)
    df.drop('msge', axis=1, inplace=True, errors='ignore')
#    df['msge']
#    print(df.columns)

    return df


def merge_base_gt(df1, df2, suffixes=('', '_true')):
    msge = 'millisSinceGpsEpoch'
    df1['msge'] = df1[msge].astype(str).str[:-2].astype(np.int64)
    df2['msge'] = df2[msge].astype(str).str[:-2].astype(np.int64)

#    df = df1.merge(df2, how='left', on=[
    df = df1.merge(df2, how='inner', on=[
        'collectionName', 'phoneName', 'msge'], suffixes=suffixes)
    df.drop('msge', axis=1, inplace=True, errors='ignore')
#    df['msge']
#    print(df.columns)

    return df


def merge_msge_true(df1, df2, suffixes=('', '_true')):
    msge = 'millisSinceGpsEpoch'
    df1['msge'] = df1[msge].astype(str).str[:-2].astype(np.int64)
    df2['msge'] = df2[msge].astype(str).str[:-2].astype(np.int64)

    df = df1.merge(df2, on=['msge', 'collectionName',
                            'phoneName'], how='left', suffixes=suffixes)

    df.drop('msge', axis=1, inplace=True, errors='ignore')
#    df['msge']
#    print(df.columns)

    return df

# --------------------------------------------------------------------------
#


def merge_routes_base(df1, df2, suffixes=('', '_base')):
    msge = 'millisSinceGpsEpoch'
    df1['msge'] = df1[msge].astype(str).str[:-2].astype(np.int64)
    df2['msge'] = df2[msge].astype(str).str[:-2].astype(np.int64)

    #    df = df1.merge(df2, how='inner', on=[
    df = df1.merge(df2, how='left', on=[
        'collectionName', 'phoneName', 'msge'], suffixes=suffixes)
    df.drop('msge', axis=1, inplace=True, errors='ignore')
    df.dropna(inplace=True)
#    df['msge']
#    print(df.columns)

    return df


def merge_routes_base_true(df1, df2, suffixes=('', '_true')):
    msge = 'millisSinceGpsEpoch'
    df1['msge'] = df1[msge].astype(str).str[:-2].astype(np.int64)
    df2['msge'] = df2[msge].astype(str).str[:-2].astype(np.int64)

    df = df1.merge(df2, on=['msge', 'collectionName',
                            'phoneName'], how='left', suffixes=suffixes)

    df.drop('msge', axis=1, inplace=True, errors='ignore')
#    df['msge']
#    print(df.columns)

    return df


# df = pd.DataFrame(dict(A=[100, 101, 102, 112, 115, 119]))
# df.info()
# df['A'] = df['A'].astype(str).str[:-1].astype(np.int64)


# --------------------------------------------------------------------------

def add_dlat_dlng(df):
    df['dlat'] = [lat2 - lat1 for lat1,
                  lat2 in zip(df['latDeg_true'], df['latDeg'])]
    df['dlng'] = [lng2 - lng1 for lng1,
                  lng2 in zip(df['lngDeg_true'], df['lngDeg'])]
    return df


def add_dist(df):

    measured = list(zip(df['latDeg'], df['lngDeg']))
    true_ref = list(zip(df['latDeg_true'], df['lngDeg_true']))

    df['dist'] = [vincenty(p1, p2) for p1, p2 in zip(measured, true_ref)]
    df['dist'] *= 1000

    return df


def add_angle(df):
    df['ang'] = [offset_angle(lat1, lng1, lat2, lng2) for
                 lat1, lng1, lat2, lng2 in zip(df['latDeg_true'],
                                               df['lngDeg_true'],
                                               df['latDeg'],
                                               df['lngDeg'])]
    return df


def offset_angle(lat1, lng1, lat2, lng2):
    #    print(lat1, lng1, lat2, lng2, np.cos(lng1))
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    d_lng = (lng2 - lng1) * np.cos(lng1)
    d_lat = (lat2 - lat1)
#    print(lat1, lng1, lat2, lng2, np.cos(lng1))
#    print(d_lat, d_lng)
    ang = np.arctan(d_lat/d_lng) * 180.0 / np.pi
    return ang


def split_date_vehicle(df):  # todo: le as parameter
    if 'collectionName' in df.columns:
        x = df['collectionName'].str.split('-US-', expand=True)
    elif 'collection' in df.columns:
        x = df['collection'].str.split('-US-', expand=True)
    else:
        return 1

    df['date'] = pd.to_datetime(x[0])
    df['vehicle'] = x[1]
#    le = preprocessing.LabelEncoder()
#    df['vehicle_code'] = le.fit_transform(df['vehicle'])

    return df

# ----------------


def split_phone(df):  # todo: le as parameter
    # if 'collectionName' in df.columns:
    #     x = df['collectionName'].str.split('-US-', expand=True)
    if 'phone' in df.columns:
        x = df['phone'].str.split('-US-', expand=True)
    else:
        return 1

    df['date'] = pd.to_datetime(x[0])
    y = x[1].str.split('_', expand=True)

    df['vehicle'] = y[0]
    df['phoneName'] = y[1]

#    le = preprocessing.LabelEncoder()
#    df['vehicle_code'] = le.fit_transform(df['vehicle'])

    return df

# ----------------


def stich_to_phone(df):  # todo: le as parameter

    df['phone'] = df['collectionName'] + '_' + df['phoneName']

    return df

# ===========================================================================


def calc_haversine(lat1, lon1, lat2, lon2):
    RADIUS = 6_367_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    dist = 2 * RADIUS * np.arcsin(a**0.5)
    return dist
# --------------------------------------------------------------------------


def compute_dist(fname, fname2='./src/gt.csv'):
    oof = pd.read_csv(fname)
    gt = pd.read_csv(fname2)
    df = oof.merge(gt, on=['phone', 'millisSinceGpsEpoch'])
    dst_oof = calc_haversine(df.latDeg_x, df.lngDeg_x,
                             df.latDeg_y, df.lngDeg_y)
    scores = pd.DataFrame({'phone': df.phone, 'dst': dst_oof})
    scores_grp = scores.groupby('phone')
    d50 = scores_grp.quantile(.50).reset_index()
    d50.columns = ['phone', 'q50']
    d95 = scores_grp.quantile(.95).reset_index()
    d95.columns = ['phone', 'q95']
    return (scores_grp.quantile(.50).mean() + scores_grp.quantile(.95).mean())/2, d50.merge(d95)


# --------------------------------------------------------------------------


def gsdc_score(df):
    dst_oof = calc_haversine(df['latDeg'], df['lngDeg'],
                             df['latDeg_true'], df['lngDeg_true'])
    scores = pd.DataFrame({'phone': df.phone, 'dst': dst_oof})
    scores_grp = scores.groupby('phone')
    d50 = scores_grp.quantile(.50).reset_index()
    d50.columns = ['phone', 'q50']
    d95 = scores_grp.quantile(.95).reset_index()
    d95.columns = ['phone', 'q95']
    scores = d50.merge(d95)
    s50 = scores_grp.quantile(.50).mean()
    s95 = scores_grp.quantile(.95).mean()

    score = (s50 + s95)/2
    return score, s50, s95, scores

# --------------------------------------------------------------------------
# make a list of error and amount of corrections


def cal_errors(df_routes, df_baseline, df_gt, clipped=False, nn=3, sig_cutoff=2.0):
    # process one route
    # -------------------------
    dfx = merge_routes_base(df_routes, df_baseline)
    dfx = merge_routes_base_true(dfx, df_gt)

    dfx = add_dlat_dlng(dfx)
    dfx = add_dist(dfx)
    dfx = add_angle(dfx)
    dfx = split_date_vehicle(dfx)

# -------------------
    x0 = error_no_correction(dfx, clipped)
    x1, dlat_x1, dlng_x1 = error_raw_correction(dfx, clipped)
    x3, dlat_x3s, dlng_x3s, dlat_x3m, dlng_x3m = error_conditional_correction(
        dfx, clipped)

    # --------------------------------------------------
    #  with outlier correction
    # --------------------------------------------------
    # inplace = True
    # dfc = merge_routes_base(df_routes, df_baseline)
    # dfc = clip_baseline(dfc, nn, sig_cutoff, inplace)
    # dfc = merge_routes_base_true(dfc, df_gt)
    # dfc = add_dlat_dlng(dfc)
    # dfc = add_dist(dfc)
    # dfc = add_angle(dfc)
    # dfc = split_date_vehicle(dfc)

    # c0 = error_no_correction(dfc)
    # c1, dlat_c1, dlng_c1 = error_raw_correction(dfc)
    # c3, dlat_c3s, dlng_c3s, dlat_c3m, dlng_c3m = error_conditional_correction(
    #     dfc)

    # --------------------------------------------------
    collectionName = dfx['collectionName'].unique()[0]
    phoneName = dfx['phoneName'].unique()[0]

    x_log = dict(
        #        i_p=[i_p],
        collectionName=[collectionName],
        phoneName=[phoneName],
        x0=[x0],
        x1=[x1],
        #        x2=[x2],
        x3=[x3],

        # c0=[c0],
        # c1=[c1],
        # #        c2=[c2],
        # c3=[c3],

        dlat_x1=[dlat_x1],
        #        dlat_x2=[dlat_x2],
        dlat_x3s=[dlat_x3s],
        dlat_x3m=[dlat_x3m],

        dlng_x1=[dlng_x1],
        #       dlng_x2=[dlng_x2],
        dlng_x3s=[dlng_x3s],
        dlng_x3m=[dlng_x3m],

        # dlat_c1=[dlat_c1],
        # #        dlat_c2=[dlat_c2],
        # dlat_c3s=[dlat_c3s],
        # dlat_c3m=[dlat_c3m],

        # dlng_c1=[dlng_c1],
        # #       dlng_c2=[dlng_c2],
        # dlng_c3s=[dlng_c3s],
        # dlng_c3m=[dlng_c3m]

    )

    df_out1 = pd.DataFrame(x_log)
    return df_out1

# --------------------------------------------------------------------------


def cal_scores(df_routes, df_baseline, df_gt, clipped=False):
    # process one route
    # -------------------------
    dfx = merge_routes_base(df_routes, df_baseline)
    dfx = merge_routes_base_true(dfx, df_gt)

    dfx = add_dlat_dlng(dfx)
    dfx = add_dist(dfx)
    dfx = add_angle(dfx)
    dfx = split_date_vehicle(dfx)

    # -------------------
    x0m, x05, x09 = score_no_correction(dfx, clipped)
    x1m, x15, x19 = score_raw_correction(dfx, clipped)
    x3m, x35, x39 = score_conditional_correction(dfx, clipped)

#    pdb.set_trace()
    # --------------------------------------------------
    #  with outlier correction
    # # --------------------------------------------------
    # dfc = merge_routes_base(df_routes, df_baseline)
    # dfc = clip_baseline(dfc, nn, sig_cutoff, inplace=True)
    # dfc = merge_routes_base_true(dfc, df_gt)
    # dfc = add_dlat_dlng(dfc)
    # dfc = add_dist(dfc)
    # dfc = add_angle(dfc)
    # dfc = split_date_vehicle(dfc)

    # c0m, c05, c09 = score_no_correction(dfc, clipped=True)
    # c1m, c15, c19 = score_raw_correction(dfc, clipped=True)
    # c3m, c35, c39 = score_conditional_correction(dfc, clipped=True)

#     pdb.set_trace()
    # -------------------
    collectionName = dfx['collectionName'].unique()[0]
    phoneName = dfx['phoneName'].unique()[0]

    x_log = dict(
        #        i_p=[i_p],
        collectionName=[collectionName],
        phoneName=[phoneName],
        x0m=[x0m],
        x05=[x05],
        x09=[x09],

        x1m=[x1m],
        x15=[x15],
        x19=[x19],

        x3m=[x3m],
        x35=[x35],
        x39=[x39])

    # c0m=[c0m],
    # c05=[c05],
    # c09=[c09],

    # c1m=[c1m],
    # c15=[c15],
    # c19=[c19],

    # c3m=[c3m],
    # c35=[c35],
    # c39=[c39])

    df_out1 = pd.DataFrame(x_log)

    return df_out1

#     if i_p == 0:
#         df_out = df_out1
#     else:
#         df_out = pd.concat([df_out, df_out1], axis=0)

#     print(f'{i_p:2} {collection:19} {phone:14} {x0:6.3f} {x1:6.3f} {x3:6.3f} {c0:6.3f} {c1:6.3f}  {c3:6.3f}')
# # #    print(f'{i_p:2} {collection:19} {phone:14} {dlat_x1:6.2e} {dlat_x3s:6.2e} {dlat_c1:6.2e} {dlat_c3s:6.2e}')

# df_out.to_csv(log1, index=False)
# # --------------------------------------------------------------------------
# make a list of error and amount of corrections
# --------------------------------------------------------------------------
