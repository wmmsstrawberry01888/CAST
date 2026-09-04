"""
run.py — CAST unified training entry point

Usage:
    # YAML config file (recommended)
    python run.py --config configs/table1_exchange.yaml --model cast

    # Legacy CLI mode (backward compatible)
    python run.py --model cast --data weather --seq_len 12

    # K-fold cross-validation
    python run.py --config configs/fig2b_kfold.yaml --model cast

    # Multi-horizon robustness
    python run.py --config configs/fig2a_horizon.yaml --model cast --model.T_h 96

Supported --model: cast / ogat / otem / ospa / onet /
                   autoformer / fedformer / pdf / xpatch /
                   itransformer / pathformer / tsamd / duet

Supported datasets: air_1 / weather / pci / turbulence / Air_A / exchange_rate
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from utils.common_utils import setup_seed
from utils.config import load_config, Config
from utils.dataset_names import display_name
from utils.paramters import get_diffusion_config
from utils.select_data import (select_data, select_k_fold_data,
                                select_data_splits, select_k_fold_data_splits,
                                _apply_dataset_config)
from engine.trainer import Trainer
from engine.evaluator import Evaluator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CAST unified training entry point")
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file (recommended)")
    p.add_argument("--model", type=str, default="cast")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--mode", type=str, default="standard",
                   choices=["standard", "k_fold", "select"])
    p.add_argument("--seq_len", type=int, default=12)
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Legacy config builder (backward compatible)
# ---------------------------------------------------------------------------

def _build_legacy_config(args) -> edict:
    """Build an easydict config from CLI args (old paramters.py style)."""
    config = get_diffusion_config()
    config.T_h = args.seq_len
    config.model.T_h = args.seq_len
    config.model.T_p = args.seq_len
    config.train = edict()
    config.train.epochs = args.epochs or config.epoch
    config.train.lr = args.lr or config.lr
    config.train.batch_size = args.batch or config.batch_size
    config.train.weight_decay = config.wd
    config.train.early_stop_patience = config.early_stop
    config.train.seed = args.seed
    config.train.lambda_cdg = getattr(config, 'lambda_cdg', 1.0)
    config.train.use_amp = args.use_amp
    config.train.grad_clip = 0.0
    config.train.lr_scheduler = "none"
    return config


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

CAST_FAMILY = {"cast", "ogat", "otem", "ospa", "onet"}


def _build_model(model_name: str, config, n_utils):
    name = model_name.lower()
    V = config.model.V
    F = config.model.F
    T = config.model.T_h
    d = V * F

    # ---- CAST family ------------------------------------------------
    if name == "cast":
        from model.cast import Cast
        return Cast(config.model, n_utils, [4, 1], V)

    if name == "ogat":
        from model.cast import Cast
        return Cast(config.model, n_utils, [4, 1], V, use_gat=False)

    if name == "otem":
        from model.cast import Cast
        return Cast(config.model, n_utils, [4, 1], V, use_temporal=False)

    if name == "ospa":
        from model.cast import Cast
        return Cast(config.model, n_utils, [4, 1], V, use_spatio=False)

    if name == "onet":
        from model.cast import Cast
        return Cast(config.model, n_utils, [4, 1], V, use_network=False)

    # ---- Autoformer ------------------------------------------------
    if name == "autoformer":
        from models.Autoformer import Model
        mc = edict()
        mc.version = "Fourier"; mc.mode_select = "random"; mc.modes = 64
        mc.seq_len = T; mc.label_len = T; mc.pred_len = T
        mc.output_attention = False; mc.moving_avg = [24]
        mc.enc_in = d; mc.d_model = d; mc.embed = "timeF"; mc.freq = "h"
        mc.dropout = 0.0; mc.L = 3; mc.base = "legendre"
        mc.cross_activation = "tanh"; mc.dec_in = d
        mc.e_layers = 2; mc.n_heads = V; mc.d_ff = 2048
        mc.activation = "gelu"; mc.c_out = d; mc.d_layers = 1
        mc.node_num = V; mc.factor = 1.2
        return Model(mc)

    # ---- FEDformer ------------------------------------------------
    if name == "fedformer":
        from models.FEDformer import Model
        mc = edict()
        mc.version = "Fourier"; mc.mode_select = "random"; mc.modes = 64
        mc.seq_len = T; mc.label_len = T; mc.pred_len = T
        mc.output_attention = False; mc.moving_avg = [24]
        mc.enc_in = d; mc.d_model = d; mc.embed = "timeF"; mc.freq = "h"
        mc.dropout = 0.0; mc.L = 3; mc.base = "legendre"
        mc.cross_activation = "tanh"; mc.dec_in = d
        mc.e_layers = 2; mc.n_heads = V; mc.d_ff = 2048
        mc.activation = "gelu"; mc.c_out = d; mc.d_layers = 1
        mc.node_num = V; mc.factor = 1.2
        return Model(mc)

    # ---- PDF -------------------------------------------------------
    if name == "pdf":
        mc = edict()
        mc.enc_in = d; mc.seq_len = T; mc.pred_len = T
        mc.e_layers = 3; mc.n_heads = V; mc.d_model = d; mc.d_ff = 128
        mc.dropout = 0.25; mc.fc_dropout = 0.15; mc.head_dropout = 0.0
        mc.individual = False; mc.add = False; mc.wo_conv = False
        mc.serial_conv = False; mc.kernel_list = [3, 7, 11]
        mc.patch_len = [1]; mc.period = [2]; mc.stride = [1]
        # RevIN enabled to match PDF's official instance-normalization setup.
        mc.padding_patch = "end"; mc.revin = 1; mc.affine = 1
        mc.subtract_last = 0
        from models.PDF import Model
        return Model(mc)

    # ---- xPatch ---------------------------------------------------
    if name == "xpatch":
        # xPatch exposes the class XPatch with positional args (there is no
        # `Model` symbol in models/xPatch.py). RevIN is enabled to match the
        # official xPatch instance-normalization configuration.
        from models.xPatch import XPatch
        return XPatch(
            seq_len=T, pred_len=T, c_in=d,
            patch_len=16, stride=8, padding_patch="end",
            revin=1, ma_type="ema", alpha=0.3, beta=0.3,
        )

    # ---- iTransformer ---------------------------------------------
    if name == "itransformer":
        from models.iTransformer import Model
        # use_norm=True enables the Non-stationary instance normalization that
        # iTransformer relies on for multi-scale / multi-magnitude inputs, as in
        # its official configuration.
        return Model(
            seq_len=T, pred_len=T, d_model=d, embed="timeF", n_heads=1,
            d_ff=d, output_attention=0, e_layers=1, use_norm=True,
            freq="h", activation="gelu", dropout=0.5,
            class_strategy="projection", factor=3,
        )

    # ---- PathFormer -----------------------------------------------
    if name == "pathformer":
        from models.PathFormer import Model
        return Model(
            layer_nums=3, num_nodes=d, pred_len=T, seq_len=T, k=2,
            num_experts_list=[4, 4, 4],
            patch_size_list=[[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]],
            d_model=d, d_ff=64, residual_connection=True, revin=True,
            device=config.device, batch_norm=False,
        )

    # ---- tsAMD ----------------------------------------------------
    if name == "tsamd":
        from models.tsAMD import AMD
        # norm=True enables tsAMD's RevIN, matching its official configuration.
        return AMD(
            input_shape=(T, d), pred_len=T, dropout=0.1, n_block=1,
            patch=16, k=3, c=2, alpha=0.0, target_slice=slice(0, None),
            norm=True, layernorm=True,
        )

    # ---- Duet -----------------------------------------------------
    if name == "duet":
        # DUETModel enables RevIN unconditionally inside its extractor cluster
        # (layers/linear_extractor_cluster.py), so instance normalization is
        # already active in its official configuration. All constructor
        # arguments below are required by DUETModel.__init__.
        from models.duet import DUETModel as Model
        return Model(
            seq_len=T, pred_len=T, d_model=d, hidden_size=d, enc_in=d,
            n_heads=1, e_layers=1, num_experts=2, d_ff=d, k=1, factor=3,
            noisy_gating=True, activation="gelu", output_attention=False,
            moving_avg=25, CI=True, dropout=0.1, fc_dropout=0.1,
        )

    raise ValueError(f"Unknown model: {model_name!r}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(config, args, index_fold=None):
    """Return (config, n_utils, train_loader, val_loader, test_loader, air_data).

    Uses chronological train/val/test splits so that reported numbers come from
    a strictly held-out future test set, with validation driving early stopping
    and checkpoint selection.
    """
    data_name = args.data
    if args.mode == "k_fold":
        return select_k_fold_data_splits(
            config, data_name, index_fold, args.k_folds,
        )
    else:
        return select_data_splits(config, data_name)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(config, args, index_fold=None) -> Dict[str, Any]:
    """Train + evaluate one model configuration. Returns metrics dict."""
    model_name = args.model.lower()
    is_cast = model_name in CAST_FAMILY
    device = torch.device(args.device) if args.device else config.device

    # Data (chronological train / val / test splits)
    config, n_utils, train_loader, val_loader, test_loader, air_data = \
        _load_data(config, args, index_fold)

    # Model
    model = _build_model(model_name, config, n_utils)
    model = model.to(device)

    tag = f"{model_name}_{args.data}"
    if index_fold is not None:
        tag += f"_fold{index_fold}"
    save_dir = os.path.join(config.PATH_MOD, tag + "_" +
                            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_dir = os.path.join(config.PATH_LOG, tag)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}  |  Data: {display_name(args.data)}  |  "
          f"Mode: {args.mode}  |  T_h: {config.model.T_h}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}\n")

    # Trainer — validation drives early stopping and best-checkpoint selection.
    trainer = Trainer(model, config, device, save_dir=save_dir, log_dir=log_dir)
    result = trainer.fit(train_loader, val_loader, air_data, model_name, is_cast)

    print(f"\nBest epoch: {result['best_epoch']}, "
          f"Best val loss: {result['best_val_loss']:.4f}")

    # ---- Independent test-set evaluation --------------------------------
    # fit() has already restored the best-validation weights. We now evaluate
    # once on the strictly held-out test set through the shared Evaluator
    # pipeline (same denormalization / metric code for CAST and every
    # baseline). These are the numbers reported in the paper tables.
    evaluator = Evaluator(model, config, device)
    test_metrics = evaluator.evaluate(test_loader, air_data, model_name, is_cast)

    print("\nTest-set metrics (held-out):")
    for k, v in test_metrics.items():
        print(f"  test_{k}: {v:.6f}")

    # Machine-parseable line consumed by batch_run.py. Emitting explicit
    # test_* metrics (rather than scraping the training progress bar) ensures
    # the aggregated tables average independent held-out test scores.
    tagged = {f"test_{k}": (None if v is None or (isinstance(v, float) and np.isnan(v))
                            else float(v))
              for k, v in test_metrics.items()}
    tagged["best_epoch"] = result["best_epoch"]
    tagged["best_val_loss"] = (None if np.isnan(result["best_val_loss"])
                               else float(result["best_val_loss"]))
    print("TEST_METRICS_JSON: " + json.dumps(tagged))

    result["test_metrics"] = test_metrics
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_seed(args.seed)

    # Decide device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Build config
    if args.config:
        # New YAML config path — CLI args override YAML values
        cli_overrides = {}
        if args.model: cli_overrides['experiment'] = {'model': args.model}
        if args.data: cli_overrides.setdefault('experiment', {})['data'] = args.data
        if args.mode: cli_overrides.setdefault('experiment', {})['mode'] = args.mode
        if args.seq_len != 12:
            cli_overrides.setdefault('model', {})['T_h'] = args.seq_len
        if args.k_folds: cli_overrides.setdefault('experiment', {})['k_folds'] = args.k_folds
        if args.epochs:
            cli_overrides.setdefault('train', {})['epochs'] = args.epochs
        if args.batch:
            cli_overrides.setdefault('train', {})['batch_size'] = args.batch
        if args.lr:
            cli_overrides.setdefault('train', {})['lr'] = args.lr

        new_config = load_config(args.config, cli_overrides)
        # Convert to legacy easydict for backward compat with models/select_data
        config = edict()
        config.device = device
        config.model = edict()
        config.model.T_h = new_config.model.T_h
        config.model.T_p = new_config.model.T_p
        config.model.V = new_config.model.V
        config.model.F = new_config.model.F
        config.model.device = device
        config.train = edict()
        config.train.epochs = new_config.train.epochs
        config.train.lr = new_config.train.lr
        config.train.batch_size = new_config.train.batch_size
        config.train.weight_decay = new_config.train.weight_decay
        config.train.early_stop_patience = new_config.train.early_stop_patience
        config.train.seed = new_config.train.seed
        config.train.lambda_cdg = new_config.train.lambda_cdg
        config.train.use_amp = new_config.train.use_amp
        config.train.grad_clip = new_config.train.grad_clip
        config.train.lr_scheduler = new_config.train.lr_scheduler
        config.data = edict()
        config.data.num_features = new_config.data.num_features
        config.data.num_vertices = new_config.data.num_vertices
        config.data.points_per_hour = 12
        config.epoch = new_config.train.epochs
        config.lr = new_config.train.lr
        config.batch_size = new_config.train.batch_size
        config.wd = new_config.train.weight_decay
        config.early_stop = new_config.train.early_stop_patience
        config.PATH_MOD = os.path.join(new_config.experiment.output_dir, "model")
        config.PATH_LOG = os.path.join(new_config.experiment.output_dir, "log")
        config.PATH_FORECAST = os.path.join(new_config.experiment.output_dir, "forecast")
        from utils.common_utils import Logger
        config.logger = Logger()

        # Override data from CLI
        args.data = new_config.experiment.data
        args.mode = new_config.experiment.mode
        args.k_folds = new_config.experiment.k_folds
    else:
        config = _build_legacy_config(args)
        config.device = device
        config.model.device = device

    # Run
    if args.mode == "k_fold":
        for fold_i in range(args.k_folds):
            cfg = _build_legacy_config(args) if not args.config else config
            cfg.device = device
            cfg.model.device = device
            print(f"\n{'#'*60}")
            print(f"# Fold {fold_i + 1}/{args.k_folds}")
            print(f"{'#'*60}")
            run_single(cfg, args, index_fold=fold_i)
    else:
        run_single(config, args)


if __name__ == "__main__":
    main()
