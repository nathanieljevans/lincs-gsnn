import pytest
import torch

from lincs_gsnn.train.metrics import mean_time_series_correlation


def _mean_r(y_true, y_pred):
    r_sum, r_n = mean_time_series_correlation(y_true, y_pred)
    return r_sum / max(r_n, 1)


def test_perfect_linear_correlation_is_one():
    t = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    y_true = t.expand(2, 5, 3)
    y_pred = 2.0 * y_true + 1.0
    assert _mean_r(y_true, y_pred) == pytest.approx(1.0, abs=1e-5)


def test_shuffled_time_lowers_correlation():
    torch.manual_seed(0)
    y_true = torch.randn(2, 6, 4)
    y_pred = y_true.clone()
    y_pred[:, torch.randperm(6), :] = y_pred[:, torch.randperm(6), :]
    assert _mean_r(y_true, y_pred) < 0.99


def test_constant_series_excluded():
    y_true = torch.ones(2, 4, 3)
    y_pred = torch.randn(2, 4, 3)
    varying = torch.randn(2, 4)
    y_true[:, :, 0] = varying
    y_pred[:, :, 0] = varying * 2.0 + 0.5
    r_sum, r_n = mean_time_series_correlation(y_true, y_pred)
    assert r_n == 2
    assert r_sum / r_n == pytest.approx(1.0, abs=1e-5)


def test_empty_valid_series_returns_zero():
    y_true = torch.ones(1, 3, 2)
    y_pred = torch.ones(1, 3, 2)
    r_sum, r_n = mean_time_series_correlation(y_true, y_pred)
    assert r_n == 0
    assert r_sum == 0.0
