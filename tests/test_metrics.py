import numpy as np
from utils.metrics_new import MAE, MSE, RMSE, SMAPE, max_absolute_error, extreme_value_error, metric


def test_mae_perfect():
    x = np.array([1., 2., 3.])
    assert MAE(x, x) == 0.0


def test_mae_known():
    assert MAE(np.array([1., 2., 3.]), np.array([2., 2., 2.])) == 2.0 / 3.0


def test_mse_perfect():
    x = np.array([1., 2., 3.])
    assert MSE(x, x) == 0.0


def test_mse_known():
    assert MSE(np.array([0., 0.]), np.array([2., 2.])) == 4.0


def test_rmse_known():
    assert RMSE(np.array([0., 0.]), np.array([2., 2.])) == 2.0


def test_smape_perfect():
    x = np.array([1., 2., 3.])
    assert SMAPE(x, x) < 1e-3


def test_smape_nonzero():
    assert SMAPE(np.array([2., 3.]), np.array([1., 2.])) > 0


def test_max_ae():
    assert float(max_absolute_error(np.array([1., 0.]), np.array([1., 10.]))) == 10.0


def test_extreme_value_error():
    max_err, _ = extreme_value_error(np.array([0., 0.]), np.array([1., 5.]))
    assert float(max_err) == 5.0


def test_metric_output_shape():
    x = np.random.randn(4, 12, 4)
    assert len(metric(x, x, 4)) == 12
