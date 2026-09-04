import torch
import tempfile, os
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, TensorDataset
from engine.trainer import Trainer


class DummyAirData:
    def reverse_normalization(self, x):
        return x


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 12)

    def forward(self, x, pos_w, pos_d):
        B, T, V, F = x.shape
        return self.linear(x.view(B, T, -1)).view(B, T, V, F)


def _make_config():
    cfg = edict()
    cfg.model = edict()
    cfg.model.T_h = 12; cfg.model.T_p = 12
    cfg.model.V = 4; cfg.model.F = 3
    cfg.model.device = torch.device("cpu")
    cfg.train = edict()
    cfg.train.epochs = 1; cfg.train.lr = 1e-4
    cfg.train.batch_size = 2; cfg.train.weight_decay = 0.0
    cfg.train.early_stop_patience = 10; cfg.train.seed = 42
    cfg.train.use_amp = False; cfg.train.grad_clip = 0.0
    cfg.train.lr_scheduler = "none"
    return cfg


def _make_dataloader(num=4, T=12, V=4, F=3):
    x = torch.randn(num, T, V, F)
    label = torch.randn(num, T, V, F)
    pw = torch.randn(num, T, 1, 1)
    pd = torch.randn(num, T, 1, 1)
    edge = torch.ones(num, 1)
    return DataLoader(TensorDataset(label, x, pw, pd, edge), batch_size=2)


def test_one_epoch_no_crash():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config()
        model = DummyModel()
        loader = _make_dataloader()
        air = DummyAirData()
        trainer = Trainer(model, config, torch.device("cpu"),
                          save_dir=tmpdir, log_dir=tmpdir)
        result = trainer.fit(loader, None, air, "dummy", is_cast=True)
        assert result["best_epoch"] == 0
        assert len(result["train_losses"]) == 1
        assert result["train_losses"][0] > 0
