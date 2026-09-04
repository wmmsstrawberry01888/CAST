"""
Dataset loading utilities. Provides select_data (full data) and select_k_fold_data (K-fold data).
Dataset configs are managed centrally by _get_dataset_config,
eliminating duplicated code between the two functions.
"""

import importlib
from torch.utils.data import DataLoader, Subset


# ---- Dataset Registry -----------------------------------------------------------

_DATASET_CONFIGS = {}

def _reg(name, standard_module, kfold_module, file_path, node_index_list,
         num_vertices, num_features, n_utils_override=None):
    """Register a dataset configuration."""
    _DATASET_CONFIGS[name] = {
        'standard_module': standard_module,
        'kfold_module': kfold_module or standard_module,
        'file_path': file_path,
        'node_index_list': node_index_list,
        'num_vertices': num_vertices,
        'num_features': num_features,
        'n_utils_override': n_utils_override,
    }


# air_1
_reg('air_1',
     standard_module='algorithm.air1_preprocess',
     kfold_module='algorithm.air1_preprocess_k_fold',
     file_path='./data/dataset/AIR_GZ/1.txt',
     node_index_list=[
         ['TIME'],
         ['GPS经度', 'GPS纬度', 'GPS高度'],
         ['侧滑角'],
         ['纵向过载', '法向过载', '侧向过载'],
         ['俯仰角速度', '滚转角速度', '偏航角速度'],
     ],
     num_vertices=5, num_features=3,
     n_utils_override=[3, 16, 16])

# weather
_reg('weather',
     standard_module='algorithm.weather_preprocess',
     kfold_module='algorithm.weather_preprocess_k_fold',
     file_path='./data/weather/Hyperlocal_Temperature_Monitoring_20240928.csv',
     node_index_list=[
         ['Day', 'Hour'],
         ['Longitude', 'Latitude'],
         ['Install.Type', 'Borough', 'ntacode'],
         ['AirTemp'],
     ],
     num_vertices=4, num_features=3)

# pci
_reg('pci',
     standard_module='algorithm.pci_preprocess',
     kfold_module=None,
     file_path='./data/pci/DMSP_final_training.csv',
     node_index_list=[
         ['Datetimes', 'SC_AACGM_LAT', 'ID_SC'],
         ['sin_ut', 'cos_ut', 'sin_doy', 'cos_doy',
          'sin_SC_AACGM_LTIME', 'cos_SC_AACGM_LTIME'],
         ['F107', 'Bz', 'By', 'Bx', 'AL', 'AU', 'SymH', 'PC', 'vsw', 'psw'],
         ['Bz_6hr', 'By_6hr', 'Bx_6hr', 'AL_6hr', 'AU_6hr', 'SymH_6hr',
          'PC_6hr', 'vsw_6hr', 'psw_6hr'],
         ['Bz_3hr', 'By_3hr', 'Bx_3hr', 'AL_3hr', 'AU_3hr', 'SymH_3hr',
          'PC_3hr', 'vsw_3hr', 'psw_3hr'],
         ['Bz_1hr', 'By_1hr', 'Bx_1hr', 'AL_1hr', 'AU_1hr', 'SymH_1hr',
          'PC_1hr', 'vsw_1hr', 'psw_1hr'],
         ['Bz_45min', 'By_45min', 'Bx_45min', 'AL_45min', 'AU_45min',
          'SymH_45min', 'PC_45min', 'vsw_45min', 'psw_45min'],
         ['Bz_30min', 'By_30min', 'Bx_30min', 'AL_30min', 'AU_30min',
          'SymH_30min', 'PC_30min', 'vsw_30min', 'psw_30min'],
         ['Bz_10min', 'By_10min', 'Bx_10min', 'AL_10min', 'AU_10min',
          'SymH_10min', 'PC_10min', 'vsw_10min', 'psw_10min'],
     ],
     num_vertices=9, num_features=10)

# turbulence
_reg('turbulence',
     standard_module='algorithm.turbulence_preprocess',
     kfold_module=None,
     file_path='./data/turbulence/kepsilon.csv',
     node_index_list=[
         ['time', 'time', 'time', 'time', 'time', 'time'],
         ['kepsilon_Tensors_1_11', 'kepsilon_Tensors_1_12', 'kepsilon_Tensors_1_13',
          'kepsilon_Tensors_1_22', 'kepsilon_Tensors_1_23', 'kepsilon_Tensors_1_33'],
         ['kepsilon_Tensors_2_11', 'kepsilon_Tensors_2_12', 'kepsilon_Tensors_2_13',
          'kepsilon_Tensors_2_22', 'kepsilon_Tensors_2_23', 'kepsilon_Tensors_2_33'],
         ['kepsilon_Tensors_3_11', 'kepsilon_Tensors_3_12', 'kepsilon_Tensors_3_13',
          'kepsilon_Tensors_3_22', 'kepsilon_Tensors_3_23', 'kepsilon_Tensors_3_33'],
         ['kepsilon_Tensors_4_11', 'kepsilon_Tensors_4_12', 'kepsilon_Tensors_4_13',
          'kepsilon_Tensors_4_22', 'kepsilon_Tensors_4_23', 'kepsilon_Tensors_4_33'],
         ['kepsilon_Tensors_5_11', 'kepsilon_Tensors_5_12', 'kepsilon_Tensors_5_13',
          'kepsilon_Tensors_5_22', 'kepsilon_Tensors_5_23', 'kepsilon_Tensors_5_33'],
         ['kepsilon_Tensors_6_11', 'kepsilon_Tensors_6_12', 'kepsilon_Tensors_6_13',
          'kepsilon_Tensors_6_22', 'kepsilon_Tensors_6_23', 'kepsilon_Tensors_6_33'],
         ['kepsilon_Tensors_7_11', 'kepsilon_Tensors_7_12', 'kepsilon_Tensors_7_13',
          'kepsilon_Tensors_7_22', 'kepsilon_Tensors_7_23', 'kepsilon_Tensors_7_33'],
         ['kepsilon_Tensors_8_11', 'kepsilon_Tensors_8_12', 'kepsilon_Tensors_8_13',
          'kepsilon_Tensors_8_22', 'kepsilon_Tensors_8_23', 'kepsilon_Tensors_8_33'],
         ['kepsilon_Tensors_9_11', 'kepsilon_Tensors_9_12', 'kepsilon_Tensors_9_13',
          'kepsilon_Tensors_9_22', 'kepsilon_Tensors_9_23', 'kepsilon_Tensors_9_33'],
         ['kepsilon_Tensors_10_11', 'kepsilon_Tensors_10_12', 'kepsilon_Tensors_10_13',
          'kepsilon_Tensors_10_22', 'kepsilon_Tensors_10_23', 'kepsilon_Tensors_10_33'],
     ],
     num_vertices=11, num_features=6)

# Air_A — multi-sensor flight parameters
_reg('Air_A',
     standard_module='algorithm.air_A_preprocess',
     kfold_module=None,
     file_path='./data/dataset/AIR_GZ/A.txt',
     node_index_list=[
         ['IRU1004-L-311', 'IRU1004-L-310', 'IRU1004-L-312', 'IRU1004-L-315',
          'IRU1004-L-316', 'IRU1004-L-365', 'IRU1004-L-366', 'IRU1004-L-367',
          'IRU1004-L-324', 'IRU1004-L-325', 'IRU1004-L-314', 'IRU1004-L-336',
          'IRU1004-L-337', 'IRU1004-L-340', 'IRU1004-L-331', 'IRU1004-L-332',
          'IRU1004-L-333', 'IRU1004-L-313', 'IRU1004-L-317', 'IRU1004-L-321',
          'IRU1004-L-322', 'IRU1004-L-320', 'IRU1004-L-326', 'IRU1004-L-327',
          'IRU1004-L-330'],
         ['IRU2004-L-312', 'IRU2004-L-315', 'IRU2004-L-316', 'IRU2004-L-365',
          'IRU2004-L-366', 'IRU2004-L-367', 'IRU2004-L-324', 'IRU2004-L-325',
          'IRU2004-L-314', 'IRU2004-L-336', 'IRU2004-L-337', 'IRU2004-L-340',
          'IRU2004-L-331', 'IRU2004-L-332', 'IRU2004-L-333'],
         ['ADRU1004-L-206', 'ADRU1004-L-203', 'ADRU1004-L-204', 'ADRU1004-L-205',
          'ADRU1004-L-210', 'ADRU1004-L-234', 'ADRU1004-L-212', 'ADRU1004-L-211',
          'ADRU1004-L-213', 'ADRU1004-L-242', 'ADRU1004-L-246', 'ADRU1004-L-215'],
         ['ADRU2004-L-206', 'ADRU2004-L-203', 'ADRU2004-L-204', 'ADRU2004-L-205',
          'ADRU2004-L-210', 'ADRU2004-L-234', 'ADRU2004-L-212', 'ADRU2004-L-211',
          'ADRU2004-L-213', 'ADRU2004-L-242', 'ADRU2004-L-246', 'ADRU2004-L-215'],
         ['ADRU3001-L-206', 'ADRU3001-L-203', 'ADRU3001-L-204', 'ADRU3001-L-205',
          'ADRU3001-L-234', 'ADRU3001-L-212', 'ADRU3001-L-242', 'ADRU3001-L-246',
          'ADRU3001-L-215'],
         ['RDC3001-L-323', 'RDC3001-L-324', 'RDC3001-L-300', 'RDC3001-L-301',
          'RDC3001-L-327', 'RDC3001-L-330', 'RDC3001-L-270-11', 'RDC3001-L-270-12',
          'RDC3001-L-270-13', 'RDC3001-L-270-14', 'RDC3001-L-270-15',
          'RDC3001-L-270-16', 'RDC3001-L-270-17', 'RDC3001-L-270-18',
          'RDC3001-L-270-19', 'RDC3001-L-270-20', 'RDC3001-L-270-21',
          'RDC3001-L-270-22', 'RDC3001-L-306', 'RDC3001-L-307'],
         ['RDC5001-L-323', 'RDC5001-L-324', 'RDC5001-L-303', 'RDC5001-L-304',
          'RDC5001-L-367', 'RDC5001-L-271-11', 'RDC5001-L-271-12',
          'RDC5001-L-271-13', 'RDC5001-L-271-14', 'RDC5001-L-271-15',
          'RDC5001-L-271-16', 'RDC5001-L-271-17', 'RDC5001-L-271-18',
          'RDC5001-L-271-19', 'RDC5001-L-271-20', 'RDC5001-L-271-21',
          'RDC5001-L-271-22'],
         ['PDCU1001-L-271-11', 'PDCU1001-L-271-12', 'PDCU1001-L-271-13',
          'PDCU1001-L-271-14', 'PDCU1001-L-271-19', 'PDCU1001-L-271-15',
          'PDCU1001-L-271-23', 'PDCU1001-L-271-16', 'PDCU1001-L-271-20',
          'PDCU1001-L-271-17', 'PDCU1001-L-271-24', 'PDCU1001-L-271-18',
          'PDCU1001-L-271-21', 'PDCU1001-L-271-22', 'PDCU1001-L-271-27',
          'PDCU1001-L-271-28'],
         ['BCU001-L-005', 'BCU001-L-006', 'BCU001-L-070', 'BCU001-L-072',
          'BCU001-L-001', 'BCU001-L-002', 'BCU001-L-003', 'BCU001-L-004',
          'BCU001-L-273-11'],
     ],
     num_vertices=9, num_features=25)

# exchange_rate
_reg('exchange_rate',
     standard_module='algorithm.exchange_rate_preprocess',
     kfold_module=None,
     file_path='./data/exchange_rate/exchange_rate.csv',
     node_index_list=[
         ['date'],
         ['0', '1', '2'],
         ['3', '4', '5'],
         ['6', 'OT'],
     ],
     num_vertices=4, num_features=3)

# etth1
# node 0: date expanded into 3 time features (month, day, hour), skipped in
# metrics (channels 0:3). The 7 real ETTh1 variables are packed into nodes 1-3
# (3 / 3 / 1 columns, zero-padded to num_features=3), matching the V=4, F=3
# layout of the shared pipeline.
_reg('etth1',
     standard_module='algorithm.etth1_preprocess',
     kfold_module=None,
     file_path='./data/ETTh1.csv',
     node_index_list=[
         ['date'],
         ['HUFL', 'HULL', 'MUFL'],
         ['MULL', 'LUFL', 'LULL'],
         ['OT'],
     ],
     num_vertices=4, num_features=3)

# c919_1
_reg('c919_1',
     standard_module='algorithm.air1_preprocess',
     kfold_module=None,
     file_path='./data/C919/1.txt',
     node_index_list=[
         ['TIME'],
         ['GPS经度', 'GPS纬度', 'GPS高度'],
         ['侧滑角'],
         ['纵向过载', '法向过载', '侧向过载'],
         ['俯仰角速度', '滚转角速度', '偏航角速度'],
     ],
     num_vertices=5, num_features=3,
     n_utils_override=[3, 16, 16])

# c919_2
_reg('c919_2',
     standard_module='algorithm.air1_preprocess',
     kfold_module=None,
     file_path='./data/C919/2.txt',
     node_index_list=[
         ['TIME'],
         ['GPS经度', 'GPS纬度', 'GPS高度'],
         ['侧滑角'],
         ['纵向过载', '法向过载', '侧向过载'],
         ['俯仰角速度', '滚转角速度', '偏航角速度'],
     ],
     num_vertices=5, num_features=3,
     n_utils_override=[3, 16, 16])

# c919_3
_reg('c919_3',
     standard_module='algorithm.air1_preprocess',
     kfold_module=None,
     file_path='./data/C919/3.txt',
     node_index_list=[
         ['TIME'],
         ['GPS经度', 'GPS纬度', 'GPS高度'],
         ['侧滑角'],
         ['纵向过载', '法向过载', '侧向过载'],
         ['俯仰角速度', '滚转角速度', '偏航角速度'],
     ],
     num_vertices=5, num_features=3,
     n_utils_override=[3, 16, 16])

# c919_4
_reg('c919_4',
     standard_module='algorithm.air1_preprocess',
     kfold_module=None,
     file_path='./data/C919/4.txt',
     node_index_list=[
         ['TIME'],
         ['GPS经度', 'GPS纬度', 'GPS高度'],
         ['侧滑角'],
         ['纵向过载', '法向过载', '侧向过载'],
         ['俯仰角速度', '滚转角速度', '偏航角速度'],
     ],
     num_vertices=5, num_features=3,
     n_utils_override=[3, 16, 16])

# ---- Lite Datasets for Quick Validation -----------------------------------------------------------
# Lightweight versions (1000 samples) for reviewer validation

_reg('etth1_lite',
     standard_module='algorithm.etth1_preprocess',
     kfold_module=None,
     file_path='./data/lite/ETTh1.csv',
     node_index_list=[
         ['date'],
         ['HUFL', 'HULL', 'MUFL'],
         ['MULL', 'LUFL', 'LULL'],
         ['OT'],
     ],
     num_vertices=4, num_features=3)

_reg('exchange_rate_lite',
     standard_module='algorithm.exchange_rate_preprocess',
     kfold_module=None,
     file_path='./data/lite/exchange_rate.csv',
     node_index_list=[
         ['date'],
         ['0', '1', '2'],
         ['3', '4', '5'],
         ['6', 'OT'],
     ],
     num_vertices=4, num_features=3)

_reg('weather_lite',
     standard_module='algorithm.weather_preprocess',
     kfold_module=None,
     file_path='./data/lite/weather.csv',
     node_index_list=[
         ['Day', 'Hour'],
         ['Longitude', 'Latitude'],
         ['Install.Type', 'Borough', 'ntacode'],
         ['AirTemp'],
     ],
     num_vertices=4, num_features=3)

_reg('c919_lite',
     standard_module='algorithm.air1_preprocess',
     kfold_module=None,
     file_path='./data/lite/C919_1.txt',
     node_index_list=[
         ['TIME'],
         ['GPS经度', 'GPS纬度', 'GPS高度'],
         ['侧滑角'],
         ['纵向过载', '法向过载', '侧向过载'],
         ['俯仰角速度', '滚转角速度', '偏航角速度'],
     ],
     num_vertices=5, num_features=3,
     n_utils_override=[3, 16, 16])

_reg('pci_lite',
     standard_module='algorithm.pci_preprocess',
     kfold_module=None,
     file_path='./data/lite/pci.csv',
     node_index_list=[
         ['Datetimes', 'SC_AACGM_LAT', 'ID_SC'],
         ['sin_ut', 'cos_ut', 'sin_doy', 'cos_doy',
          'sin_SC_AACGM_LTIME', 'cos_SC_AACGM_LTIME'],
         ['F107', 'Bz', 'By', 'Bx', 'AL', 'AU', 'SymH', 'PC', 'vsw', 'psw'],
         ['Bz_6hr', 'By_6hr', 'Bx_6hr', 'AL_6hr', 'AU_6hr', 'SymH_6hr',
          'PC_6hr', 'vsw_6hr', 'psw_6hr'],
         ['Bz_3hr', 'By_3hr', 'Bx_3hr', 'AL_3hr', 'AU_3hr', 'SymH_3hr',
          'PC_3hr', 'vsw_3hr', 'psw_3hr'],
         ['Bz_1hr', 'By_1hr', 'Bx_1hr', 'AL_1hr', 'AU_1hr', 'SymH_1hr',
          'PC_1hr', 'vsw_1hr', 'psw_1hr'],
         ['Bz_45min', 'By_45min', 'Bx_45min', 'AL_45min', 'AU_45min',
          'SymH_45min', 'PC_45min', 'vsw_45min', 'psw_45min'],
         ['Bz_30min', 'By_30min', 'Bx_30min', 'AL_30min', 'AU_30min',
          'SymH_30min', 'PC_30min', 'vsw_30min', 'psw_30min'],
         ['Bz_10min', 'By_10min', 'Bx_10min', 'AL_10min', 'AU_10min',
          'SymH_10min', 'PC_10min', 'vsw_10min', 'psw_10min'],
     ],
     num_vertices=9, num_features=10)


# ---- Core Loading Functions -----------------------------------------------------------

def _apply_dataset_config(config, use_data, is_kfold=False):
    """Populate config from dataset name; return (module, file_path, node_index_list,
       data_range, n_utils)."""
    if use_data not in _DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {use_data!r}。"
            f"Available: {list(_DATASET_CONFIGS.keys())}"
        )
    cfg = _DATASET_CONFIGS[use_data]

    # Select the correct preprocessing module
    module_name = cfg['kfold_module'] if is_kfold else cfg['standard_module']
    if module_name is None:
        module_name = cfg['standard_module']
    module = importlib.import_module(module_name)

    # Populate config
    config.data.num_vertices = cfg['num_vertices']
    config.data.num_features = cfg['num_features']
    config.model.F = cfg['num_features']
    config.model.V = cfg['num_vertices']

    n_utils = (cfg['n_utils_override']
               if cfg['n_utils_override'] is not None
               else [cfg['num_features'], 16, 16])

    file_path = cfg['file_path']
    node_index_list = cfg['node_index_list']

    # Use smaller data_range for lite datasets
    if use_data.endswith('_lite'):
        data_range = [0, 1000]
    else:
        data_range = [0, 200000]

    return module, file_path, node_index_list, data_range, n_utils


# Fallback batch size used only when the config carries no train.batch_size
# (e.g. bare legacy callers). Normal runs take the value from the YAML/CLI.
_DEFAULT_BATCH_SIZE = 16

# Chronological split fractions for train / val / test. The remaining fraction
# (1 - train - val) is the held-out test set. This matches the 70/20/10 protocol
# used in the paper's experiments.
_TRAIN_FRAC = 0.7
_VAL_FRAC = 0.2


def _resolve_batch_size(config):
    """Read the effective batch size from config, honoring YAML/CLI overrides.

    Falls back to ``_DEFAULT_BATCH_SIZE`` only when neither ``config.train.
    batch_size`` nor a top-level ``config.batch_size`` is set.
    """
    train = getattr(config, 'train', None)
    bs = getattr(train, 'batch_size', None) if train is not None else None
    if bs is None:
        bs = getattr(config, 'batch_size', None)
    return int(bs) if bs else _DEFAULT_BATCH_SIZE


def _make_loader(config, dataset, shuffle=False):
    """Wrap a dataset/Subset in a DataLoader with the shared batch settings.

    The batch size comes from the config (YAML ``train.batch_size`` / CLI
    ``--batch``), not a hardcoded constant.

    drop_last=True discards a trailing incomplete batch. A leftover batch of
    size 1 breaks any model with batch normalization (e.g. tsAMD) in training
    mode ("Expected more than 1 value per channel"), which surfaces on datasets
    whose sample count is 1 mod batch_size (e.g. C919 lite: 977).
    """
    return DataLoader(dataset, batch_size=_resolve_batch_size(config),
                      shuffle=shuffle, drop_last=True)


def _chronological_split(dataset):
    """Split a dataset into chronological train / val / test Subsets.

    ``dataset.idx_lst`` is built by scanning ``label_start_idx`` in ascending
    order, so the sample order is already chronological. We therefore split on
    the sample index directly (no shuffling): the earliest 70% of windows are
    training, the next 20% validation, and the final 10% the held-out test set.
    Splitting at the sample level keeps the protocol identical for every dataset
    and for both standard and K-fold modes without touching each preprocessor.
    """
    n = len(dataset)
    n_train = int(n * _TRAIN_FRAC)
    n_val = int(n * _VAL_FRAC)
    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_train + n_val))
    test_idx = list(range(n_train + n_val, n))
    return (Subset(dataset, train_idx),
            Subset(dataset, val_idx),
            Subset(dataset, test_idx))


def _build_dataset(config, use_data, module, file_path, node_index_list,
                   data_range, train_ranges=None):
    """Construct the full BoeingDataset and its DataPreprocess (air_data)."""
    DataPreprocess = module.DataPreprocess
    BoeingDataset = module.BoeingDataset

    air_data = DataPreprocess(file_path, config.data.num_features,
                              node_index_list)
    construct_data = air_data.get_data()

    ranges = train_ranges if train_ranges is not None else data_range
    dataset = BoeingDataset(construct_data, ranges, config)
    return dataset, air_data


def _build_dataloader(config, use_data, module, file_path, node_index_list,
                      data_range, n_utils, train_ranges=None):
    """Build a single DataLoader over the full range.

    Retained for backward compatibility (SelectData mode / external callers).
    New training code should use the split-aware ``select_data`` return values.
    Returns ``(config, n_utils, dataloader, air_data)``.
    """
    dataset, air_data = _build_dataset(
        config, use_data, module, file_path, node_index_list,
        data_range, train_ranges=train_ranges,
    )
    return config, n_utils, _make_loader(config, dataset), air_data


# ---- Public API --------------------------------------------------------------

def select_data(config, use_data):
    """Load full data (standard training / SelectData mode).

    Returns a single DataLoader over the full range, for backward compatibility
    with SelectData mode and external callers. For train/val/test-split training
    use :func:`select_data_splits`.
    """
    module, file_path, node_index_list, data_range, n_utils = \
        _apply_dataset_config(config, use_data, is_kfold=False)
    return _build_dataloader(config, use_data, module, file_path,
                             node_index_list, data_range, n_utils)


def select_data_splits(config, use_data):
    """Load data as chronological train / val / test loaders.

    Returns ``(config, n_utils, train_loader, val_loader, test_loader,
    air_data)``. The split is chronological (no shuffling of the time axis):
    the model is trained on the earliest 70% of windows, validated (early
    stopping / checkpoint selection) on the next 20%, and evaluated on the
    final held-out 10% — a strictly future test set that shares no windows with
    training. The same protocol and single ``air_data`` normalizer are applied
    to CAST and every baseline.
    """
    module, file_path, node_index_list, data_range, n_utils = \
        _apply_dataset_config(config, use_data, is_kfold=False)
    dataset, air_data = _build_dataset(
        config, use_data, module, file_path, node_index_list, data_range,
    )
    train_set, val_set, test_set = _chronological_split(dataset)
    # Shuffle only the training set; val/test keep chronological order.
    train_loader = _make_loader(config, train_set, shuffle=True)
    val_loader = _make_loader(config, val_set)
    test_loader = _make_loader(config, test_set)
    return config, n_utils, train_loader, val_loader, test_loader, air_data


def select_k_fold_data(config, use_data, index_fold, k_fold):
    """Load K-fold data; return training-set DataLoader (backward compat)."""
    module, file_path, node_index_list, data_range, n_utils = \
        _apply_dataset_config(config, use_data, is_kfold=True)

    # Partition into train / test ranges
    max_data_len = data_range[1] - data_range[0]
    max_item_len = max_data_len // k_fold
    test_range = []
    train_range = []
    for count, i in enumerate(range(data_range[0], data_range[1], max_item_len)):
        if count == index_fold:
            test_range.append([i, i + max_item_len])
        else:
            train_range.append([i, i + max_item_len])

    return _build_dataloader(config, use_data, module, file_path,
                             node_index_list, data_range, n_utils,
                             train_ranges=train_range)


def select_k_fold_data_splits(config, use_data, index_fold, k_fold):
    """Load K-fold data as (train, val, test) loaders for one fold.

    Fold ``index_fold`` is the held-out test block; the remaining blocks form
    the training pool, from which the last :data:`_VAL_FRAC` (chronologically)
    is carved off as validation. Returns ``(config, n_utils, train_loader,
    val_loader, test_loader, air_data)``.
    """
    module, file_path, node_index_list, data_range, n_utils = \
        _apply_dataset_config(config, use_data, is_kfold=True)

    max_data_len = data_range[1] - data_range[0]
    max_item_len = max_data_len // k_fold
    test_range = []
    train_range = []
    for count, i in enumerate(range(data_range[0], data_range[1], max_item_len)):
        if count == index_fold:
            test_range.append([i, i + max_item_len])
        else:
            train_range.append([i, i + max_item_len])

    train_pool, air_data = _build_dataset(
        config, use_data, module, file_path, node_index_list, data_range,
        train_ranges=train_range,
    )
    test_set, _ = _build_dataset(
        config, use_data, module, file_path, node_index_list, data_range,
        train_ranges=test_range,
    )

    # Carve a chronological validation tail off the training pool.
    n = len(train_pool)
    n_train = int(n * (1.0 - _VAL_FRAC))
    train_set = Subset(train_pool, list(range(0, n_train)))
    val_set = Subset(train_pool, list(range(n_train, n)))

    train_loader = _make_loader(config, train_set, shuffle=True)
    val_loader = _make_loader(config, val_set)
    test_loader = _make_loader(config, test_set)
    return config, n_utils, train_loader, val_loader, test_loader, air_data
