from pathlib import Path
from utils.config import load_config, Config


def test_load_defaults():
    cfg = load_config("configs/defaults.yaml")
    assert isinstance(cfg, Config)
    assert cfg.data.name == "weather"
    assert cfg.model.T_h == 12
    assert cfg.train.epochs == 50


def test_load_all_experiment_configs():
    for cfg_file in sorted(Path("configs").glob("*.yaml")):
        if cfg_file.name == "defaults.yaml":
            continue
        cfg = load_config(str(cfg_file))
        assert isinstance(cfg, Config), f"Failed: {cfg_file.name}"


def test_merge_override():
    cfg = load_config("configs/table1_exchange.yaml")
    assert cfg.experiment.data == "exchange_rate"


def test_model_data_sync():
    cfg = load_config("configs/table2_pci.yaml")
    assert cfg.model.T_h == cfg.model.T_p


def test_default_config():
    cfg = Config()
    assert cfg.data.name == "weather"
    assert cfg.train.lr == 1e-4


def test_config_to_dict():
    d = Config().to_dict()
    assert isinstance(d, dict)
    assert d["data"]["name"] == "weather"
