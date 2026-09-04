"""
Model evaluation — test-loop inference and metrics computation.
"""

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from engine.trainer import _baseline_forward
from utils.metrics_new import metric as compute_metrics


class Evaluator:
    """Run evaluation on a test dataloader and collect metrics."""

    def __init__(self, model: torch.nn.Module, config, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, air_data,
                 model_name: str, is_cast: bool) -> Dict[str, float]:
        """Run the test loop and return metrics computed over the full test set.

        Predictions and ground truth from every batch are accumulated and
        concatenated, and all metrics are computed exactly once on the complete
        (denormalized, raw-scale) test set. This is the single evaluation
        pipeline applied uniformly to CAST and every baseline. Because MSE,
        RMSE, MaxAE, MaxError and MinError are computed on the whole test set
        rather than averaged across batches, RMSE equals sqrt(MSE) exactly and
        the tail-focused extremes are true global extrema.
        """
        self.model.eval()
        T_h = self.config.model.T_h
        V = self.config.model.V
        all_true: List[np.ndarray] = []
        all_pred: List[np.ndarray] = []

        for batch in dataloader:
            batch_future, batch_history, pos_w, pos_d, edge = batch
            batch_future = torch.nan_to_num(batch_future, nan=0.0)
            batch_history = torch.nan_to_num(batch_history, nan=0.0)

            if is_cast:
                pred, eval_future, a = _cast_predict(
                    self.model, batch_future, batch_history, pos_w, pos_d,
                    T_h, self.device,
                )
                # eval_future / pred are (a, T_h, V, F). reverse_normalization
                # consumes the 4D array and we flatten the V*F node-feature axes
                # into a single channel axis, exactly as the Trainer does, so the
                # metric channel slice [:, :, 3:] drops the same date channels.
                b = eval_future.shape[1]
                final_future_np = air_data.reverse_normalization(
                    eval_future.detach().cpu().numpy()
                ).reshape(a, b, -1)
                final_pred_np = air_data.reverse_normalization(
                    pred.detach().cpu().numpy()
                ).reshape(a, b, -1)
            else:
                pred, eval_future, a = _baseline_predict(
                    self.model, batch_future, batch_history,
                    T_h, model_name, V, self.device,
                )
                final_future_np = air_data.reverse_normalization(
                    eval_future.detach().cpu().numpy().reshape(a, T_h, V, -1)
                ).reshape(a, T_h, -1)
                final_pred_np = air_data.reverse_normalization(
                    pred.detach().cpu().numpy().reshape(a, T_h, V, -1)
                ).reshape(a, T_h, -1)

            all_true.append(final_future_np[:, :, 3:])
            all_pred.append(final_pred_np[:, :, 3:])

        true_all = np.concatenate(all_true, axis=0)
        pred_all = np.concatenate(all_pred, axis=0)
        avg = np.array(compute_metrics(true_all, pred_all, true_all.shape[0]))
        return {
            'mae': avg[0], 'mse': avg[1], 'rmse': avg[2],
            'smape': avg[3], 'dtw': avg[4], 'r2': avg[5],
            'cre': avg[6], 'max_ae': avg[7],
            'max_error': avg[8], 'min_error': avg[9],
            'ade': avg[10], 'fde': avg[11],
        }


def _cast_predict(model, batch_future, batch_history, pos_w, pos_d,
                  T_h: int, device):
    batch_future = batch_future.to(device)
    batch_history = batch_history.to(device)
    pos_w = pos_w.to(device)
    pos_d = pos_d.to(device)
    pred = model(batch_history, pos_w, pos_d)
    pred = pred[:, -T_h:]
    eval_future = batch_future[:, :T_h]
    eval_pred = pred[:, :T_h]
    a = eval_future.shape[0]
    return eval_pred, eval_future, a


def _baseline_predict(model, batch_future, batch_history,
                      T_h: int, model_name: str, V: int, device):
    x, y, z, k = batch_future.shape
    batch_future = batch_future.view(x, y, -1)
    batch_history = batch_history.view(x, y, -1)
    batch_future = batch_future.to(device)
    batch_history = batch_history.to(device)
    pred = _baseline_forward(model, batch_history, batch_future, model_name)
    eval_future = batch_future[:, :T_h]
    eval_pred = pred[:, :T_h]
    a = eval_future.shape[0]
    return eval_pred, eval_future, a
