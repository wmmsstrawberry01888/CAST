"""
Training engine with validation, early stopping, checkpointing, and TensorBoard logging.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics_new import metric as compute_metrics


class Trainer:
    """Unified training loop for all CAST and baseline models.

    Parameters
    ----------
    model : nn.Module
    config : easydict or object with train.* attributes
    device : torch.device
    save_dir : str
        Directory for model checkpoints.
    log_dir : str
        Directory for TensorBoard logs.
    """

    def __init__(self, model: nn.Module, config, device: torch.device,
                 save_dir: str = "./output/model",
                 log_dir: str = "./logs"):
        self.model = model
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=log_dir)
        self.early_stopping = EarlyStopping(patience=config.train.early_stop_patience)
        self.scaler = GradScaler() if getattr(config.train, 'use_amp', False) else None

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )
        self.loss_fn = nn.MSELoss().to(device)

        # State
        self.current_epoch: int = 0
        self.best_val_loss: float = float('inf')
        self.best_epoch: int = 0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        # Number of optimizer steps skipped due to non-finite gradients.
        self.skipped_steps: int = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader: DataLoader, air_data,
                    model_name: str, is_cast: bool) -> Dict[str, float]:
        """Run one training epoch. Returns average metrics."""
        self.model.train()
        T_h = self.config.model.T_h
        total_loss = 0.0
        all_true: List[np.ndarray] = []
        all_pred: List[np.ndarray] = []

        # Add progress bar for batches
        pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                   desc=f"Epoch {self.current_epoch}", unit="batch",
                   ncols=100, leave=False, colour='blue')

        for i, batch in pbar:
            self.optimizer.zero_grad()

            batch_future, batch_history, pos_w, pos_d, edge = batch
            batch_future = torch.nan_to_num(batch_future, nan=0.0)
            batch_history = torch.nan_to_num(batch_history, nan=0.0)

            if self.scaler is not None:
                with autocast():
                    loss, true_np, pred_np = self._forward_step(
                        batch_future, batch_history, pos_w, pos_d,
                        T_h, air_data, model_name, is_cast,
                    )
                self.scaler.scale(loss).backward()
                if getattr(self.config.train, 'grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                            self.config.train.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, true_np, pred_np = self._forward_step(
                    batch_future, batch_history, pos_w, pos_d,
                    T_h, air_data, model_name, is_cast,
                )
                loss.backward()

                # Skip the optimizer step when the backward pass produced a
                # non-finite gradient. On datasets with very large / near-
                # constant raw features (e.g. C919 flight logs), the GAT +
                # diffusion path can stochastically overflow to inf. Feeding
                # that through clip_grad_norm_ is catastrophic: the clip
                # coefficient becomes max_norm/inf == 0, and 0 * inf == nan,
                # which then poisons EVERY parameter permanently. Detecting and
                # skipping the bad step keeps training stable instead of
                # collapsing the whole run to nan (mirrors what AMP's
                # GradScaler does on inf/nan gradients).
                finite_grad = all(
                    torch.isfinite(p.grad).all()
                    for p in self.model.parameters() if p.grad is not None
                )
                if not finite_grad:
                    self.skipped_steps += 1
                    self.optimizer.zero_grad()
                    continue

                if getattr(self.config.train, 'grad_clip', 0) > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                            self.config.train.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            all_true.append(true_np)
            all_pred.append(pred_np)

            # Update progress bar with current loss
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        pbar.close()
        # Average over the batches that actually contributed a loss (skipped
        # non-finite-gradient steps are excluded from both numerator and
        # denominator). Guard against an all-skipped epoch.
        num_valid = len(all_true)
        avg_loss = total_loss / num_valid if num_valid > 0 else float('nan')
        metrics = self._compute_epoch_metrics(all_true, all_pred)
        self.train_losses.append(avg_loss)

        return {'loss': avg_loss, **metrics}

    # ------------------------------------------------------------------
    # Validation loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader, air_data,
                       model_name: str, is_cast: bool) -> Dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        T_h = self.config.model.T_h
        total_loss = 0.0
        all_true: List[np.ndarray] = []
        all_pred: List[np.ndarray] = []

        for batch in dataloader:
            batch_future, batch_history, pos_w, pos_d, edge = batch
            batch_future = torch.nan_to_num(batch_future, nan=0.0)
            batch_history = torch.nan_to_num(batch_history, nan=0.0)

            loss, true_np, pred_np = self._forward_step(
                batch_future, batch_history, pos_w, pos_d,
                T_h, air_data, model_name, is_cast,
            )
            total_loss += loss.item()
            all_true.append(true_np)
            all_pred.append(pred_np)

        avg_loss = total_loss / len(dataloader)
        metrics = self._compute_epoch_metrics(all_true, all_pred)
        self.val_losses.append(avg_loss)

        return {'loss': avg_loss, **metrics}

    # ------------------------------------------------------------------
    # Full training
    # ------------------------------------------------------------------

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader],
            air_data, model_name: str, is_cast: bool,
            num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """Run full training with validation and early stopping.

        Returns a dict with training history.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader or None
            Validation dataloader. If None, trains without validation.
        air_data : DataPreprocess
            For reverse normalization during metric computation.
        model_name : str
        is_cast : bool
        num_epochs : int or None
            Override config.train.epochs.

        Returns
        -------
        dict with keys: best_epoch, best_val_loss, train_losses, val_losses
        """
        epochs = num_epochs or self.config.train.epochs

        # Create progress bar for epochs
        epoch_pbar = tqdm(range(epochs), desc="Training Progress",
                         unit="epoch", ncols=120, colour='cyan')

        for epoch in epoch_pbar:
            self.current_epoch = epoch
            t0 = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, air_data, model_name, is_cast)
            elapsed = time.time() - t0

            # Log
            self._log_metrics(train_metrics, 'train', epoch)
            self.writer.add_scalar('time/epoch_sec', elapsed, epoch)

            # Validate
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader, air_data, model_name, is_cast)
                self._log_metrics(val_metrics, 'val', epoch)
                val_loss = val_metrics['loss']

                # Early stopping
                self.early_stopping(val_loss, self.model)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self._save_checkpoint('best')
                if self.early_stopping.early_stop:
                    epoch_pbar.write(f"Early stopping at epoch {epoch}")
                    break
            else:
                val_loss = float('inf')
                self._save_checkpoint(f'epoch_{epoch}')

            # LR scheduling
            if getattr(self.config.train, 'lr_scheduler', 'none') != 'none':
                adjust_learning_rate(self.optimizer, epoch + 1, self.config.train)

            # Update progress bar with metrics
            epoch_pbar.set_postfix({
                'loss': f"{train_metrics['loss']:.4f}",
                'mae': f"{train_metrics['mae']:.4f}",
                'rmse': f"{train_metrics['rmse']:.4f}",
                'time': f"{elapsed:.1f}s"
            })

        epoch_pbar.close()
        self.writer.close()

        # Restore the best-validation weights so downstream test evaluation uses
        # the checkpoint selected by validation loss rather than the last epoch.
        # EarlyStopping keeps a deepcopy of the best state_dict; fall back to the
        # on-disk 'best' checkpoint when validation was used.
        if val_loader is not None and self.early_stopping.check_point is not None:
            self.model.load_state_dict(self.early_stopping.check_point)

        print(f"\nTraining completed. Best epoch: {self.best_epoch}, "
              f"Best val loss: {self.best_val_loss:.4f}")

        return {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_epoch_metrics(self, all_true: List[np.ndarray],
                               all_pred: List[np.ndarray]) -> Dict[str, float]:
        """Compute metrics once over the full concatenated epoch.

        Metrics are computed on the whole (denormalized) set rather than
        averaged across batches, so RMSE == sqrt(MSE) exactly and the tail
        extremes are true global extrema. This matches the Evaluator pipeline
        used to produce the reported table numbers.
        """
        if not all_true:
            # Entire epoch was skipped (all steps had non-finite gradients).
            nan = float('nan')
            return {'mae': nan, 'mse': nan, 'rmse': nan,
                    'smape': nan, 'r2': nan, 'max_ae': nan, 'max_error': nan}
        true_all = np.concatenate(all_true, axis=0)
        pred_all = np.concatenate(all_pred, axis=0)
        m = np.array(compute_metrics(true_all, pred_all, true_all.shape[0]))
        return {
            'mae': m[0], 'mse': m[1], 'rmse': m[2],
            'smape': m[3], 'r2': m[5],
            'max_ae': m[7], 'max_error': m[8],
        }

    def _forward_step(self, batch_future, batch_history, pos_w, pos_d,
                      T_h: int, air_data, model_name: str, is_cast: bool):
        """Single forward/backward step.

        Returns ``(loss, true_np, pred_np)`` where ``true_np`` and ``pred_np``
        are the denormalized, raw-scale ground truth and prediction arrays for
        this batch (channels 3: onward). Metrics are not computed here: callers
        accumulate the raw arrays and compute every metric once over the full
        epoch, so RMSE equals sqrt(MSE) exactly and the tail extremes are true
        global extrema rather than per-batch averages.
        """
        if is_cast:
            batch_future = batch_future.to(self.device)
            batch_history = batch_history.to(self.device)
            pos_w = pos_w.to(self.device)
            pos_d = pos_d.to(self.device)

            pred = self.model(batch_history, pos_w, pos_d)
            pred = pred[:, -T_h:]

            # Composite objective: L = L_pred + lambda_cdg * L_cdg.
            # L_pred is the direct trajectory forecasting MSE; L_cdg is the CDG
            # conditional denoising (noise-prediction) loss. cdg_loss returns 0
            # when the diffusion generator is disabled (CDG ablation).
            l_pred = self.loss_fn(batch_future, pred)
            lambda_cdg = getattr(self.config.train, 'lambda_cdg', 1.0)
            if lambda_cdg != 0 and hasattr(self.model, 'cdg_loss'):
                l_cdg = self.model.cdg_loss(batch_future)
                loss = l_pred + lambda_cdg * l_cdg
            else:
                loss = l_pred

            eval_future = batch_future[:, :T_h]
            eval_pred = pred[:, :T_h]

            a, b, c, _ = eval_future.shape
            final_future_np = air_data.reverse_normalization(
                eval_future.detach().cpu().numpy()
            ).reshape(a, b, -1)
            final_pred_np = air_data.reverse_normalization(
                eval_pred.detach().cpu().numpy()
            ).reshape(a, b, -1)
        else:
            x, y, z, k = batch_future.shape
            batch_future = batch_future.view(x, y, -1)
            batch_history = batch_history.view(x, y, -1)

            batch_future = batch_future.to(self.device)
            batch_history = batch_history.to(self.device)
            pos_w = pos_w.to(self.device)
            pos_d = pos_d.to(self.device)

            pred = _baseline_forward(self.model, batch_history, batch_future, model_name)
            loss = self.loss_fn(batch_future, pred)

            eval_future = batch_future[:, :T_h]
            eval_pred = pred[:, :T_h]

            a, b, c = eval_future.shape
            V = self.config.model.V
            final_future_np = air_data.reverse_normalization(
                eval_future.detach().cpu().numpy()
                .reshape(a, b, V, -1)
            ).reshape(a, b, -1)
            final_pred_np = air_data.reverse_normalization(
                eval_pred.detach().cpu().numpy()
                .reshape(a, b, V, -1)
            ).reshape(a, b, -1)

        return loss, final_future_np[:, :, 3:], final_pred_np[:, :, 3:]

    def _save_checkpoint(self, tag: str) -> None:
        path = self.save_dir / f"{tag}.pth"
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, path)

    def _log_metrics(self, metrics: Dict[str, float], prefix: str, step: int) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{key}', value, step)


def _baseline_forward(model, history, future, model_name: str):
    """Forward pass for non-CAST baseline models."""
    name = model_name.lower()
    if name in ("autoformer", "fedformer"):
        return model(history, None, future, None)
    if name == "itransformer":
        return model(history, None, None, None)
    if name in ("pathformer", "tsamd", "duet"):
        # These models return (prediction, aux) tuples; keep only the forecast.
        out = model(history)
        return out if not isinstance(out, (tuple, list)) else out[0]
    return model(history)
