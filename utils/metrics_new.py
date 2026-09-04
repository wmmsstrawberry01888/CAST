import numpy as np
import math
from dtaidistance import dtw
from scipy import stats
from sklearn.metrics import r2_score
import torch


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def SMAPE(true, pred):
    """
    Symmetric Mean Absolute Percentage Error
    Note: result is multiplied by 100
    + EPSILON  # small constant to prevent division by zero
    """
    return 100 * np.mean((2.0 * np.abs(true-pred)+1e-5) / ((np.abs(true) + np.abs(pred)+1e-5) ))

def list_checker_nan(data):
    for i in range(len(data)):
        if math.isnan(data[i]):
            data[i] = 0.0
    return data

def Calc_dtw(true, pred, n_test):
    true = stats.zscore (true, axis=1).astype(np.float64)
    pred = stats.zscore (pred, axis=1).astype(np.float64)
    dtw_set = np.zeros((n_test,))

    for ii in range(n_test):
        for jj in range(true.shape[1]):
            tem_t = list_checker_nan(true[ii, jj, :])
            tem_p = list_checker_nan(pred[ii, jj, :])
            dtw_set[ii] += dtw.distance_fast(tem_t, tem_p)
        dtw_set[ii] /= true.shape[1]

    return np.mean(dtw_set)


def R2_score(pred, true):
    pred_flat = np.reshape(pred, (pred.shape[0], -1))
    true_flat = np.reshape(true, (true.shape[0], -1))

    # Create mask: find positions where both are finite
    mask = np.isfinite(pred_flat) & np.isfinite(true_flat)

    # Keep valid values
    pred_valid = pred_flat[mask]
    true_valid = true_flat[mask]

    if len(pred_valid) == 0 or len(true_valid) == 0:
        print("Warning: No valid values for R2 score.")
        return np.nan  # or return 0.0, depending on need

    return r2_score(true_valid, pred_valid)

def max_absolute_error(true, pred):
    true_tensor=torch.tensor(true)
    pred_tensor=torch.tensor(pred)
    absolute_errors = torch.abs(true_tensor - pred_tensor)
    max_ae = torch.max(absolute_errors)
    return max_ae


def extreme_value_error(true, pred):
    true_tensor=torch.tensor(true)
    pred_tensor=torch.tensor(pred)
    y_true_max = torch.max(true_tensor)
    y_true_min = torch.min(true_tensor)

    y_pred_max = torch.max(pred_tensor)
    y_pred_min = torch.min(pred_tensor)

    max_error = torch.abs(y_true_max - y_pred_max)
    min_error = torch.abs(y_true_min - y_pred_min)

    return max_error, min_error
    #return max_error.item(),min_error.item()

def metric(pred, true, n_test):
    pred = np.array(pred)
    true = np.array(true)

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    smape = SMAPE(true, pred)
    dtw_set = 0.0
    r2 = R2_score(pred, true)
    cre = 0.0
    Maxae = max_absolute_error(true, pred)
    max_error, min_error = extreme_value_error(true, pred)
    ade = ADE(pred, true)
    fde = FDE(pred, true)
    return mae, mse, rmse, smape, dtw_set, r2, cre, Maxae, max_error, min_error, ade, fde

def distance(lon1, lat1, lon2, lat2):  # Result in kilometers
    radius = 6371  # km

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) \
        * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2

    # Clip a to [0,1] to avoid numerical errors
    a = np.clip(a, 0.0, 1.0)

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = radius * c

    return np.mean(d)


def ADE(pred, true):
    lon_index = 0
    lat_index = 1
    return distance(pred[:, :, lon_index].reshape(-1), pred[:, :, lat_index].reshape(-1), true[:, :, lon_index].reshape(-1), true[:, :, lat_index].reshape(-1))


def FDE(pred, true, index=-1):
    lon_index = 0
    lat_index = 1
    return distance(pred[:, index, lon_index], pred[:, index, lat_index], true[:, index, lon_index], true[:, index, lat_index])







