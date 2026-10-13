"""
accuracy_metrics.py
===================
Forecast-accuracy metrics shared by rebuild_models_v2.py (retrain report) and inventory_forecasting.py (/api/metrics).

All functions take a DataFrame / arrays of ACTUAL and PREDICTED values for held-out months.

  MAE    Mean Absolute Error            mean(|actual - predicted|)                  same unit as the target (units / PHP)
  RMSE   Root Mean Squared Error        sqrt(mean((actual - predicted)^2))          same unit; punishes large misses
  WAPE   Weighted Absolute Pct Error    sum(|actual - predicted|) / sum(actual)     scale-free, volume-weighted, safe with zeros
  NRMSE  Normalised RMSE (CV of RMSE)   RMSE / mean(actual)                         scale-free
  Bias   Forecast bias                  (sum(predicted) / sum(actual) - 1)          + = over-forecast, - = under-forecast
  Total-level APE                       |sum_c predicted - sum_c actual| / sum_c actual, per month
         (aggregate / bottom-up accuracy: how good the overall total is when category forecasts are added up)

MAPE is intentionally NOT used: many categories sell 0 units in some months, which makes it undefined or explosive.
MAE and RMSE depend on the scale of the series, so compare categories with WAPE / NRMSE, not with MAE / RMSE.
"""
import numpy as np
import pandas as pd


def _arr(x):
    return np.asarray(x, dtype=float)


def mae(actual, predicted):
    return float(np.mean(np.abs(_arr(actual) - _arr(predicted)))) if len(actual) else float('nan')


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((_arr(actual) - _arr(predicted)) ** 2))) if len(actual) else float('nan')


def wape(actual, predicted):
    a = _arr(actual)
    total = np.abs(a).sum()
    return float(np.abs(a - _arr(predicted)).sum() / total) if total > 0 else float('nan')


def nrmse(actual, predicted):
    a = _arr(actual)
    mean = a.mean() if len(a) else 0.0
    return float(rmse(actual, predicted) / mean) if mean > 0 else float('nan')


def bias(actual, predicted):
    total = _arr(actual).sum()
    return float(_arr(predicted).sum() / total - 1) if total > 0 else float('nan')


def total_level_ape(log, month_col='Month', actual_col='actual', predicted_col='predicted'):
    """Per-month absolute % error of the SUM over all categories. Returns a Series indexed by month."""
    t = log.groupby(month_col)[[actual_col, predicted_col]].sum()
    return ((t[predicted_col] - t[actual_col]).abs() / t[actual_col].where(t[actual_col] > 0)).rename('total_ape')


def summarize(log, month_col='Month', actual_col='actual', predicted_col='predicted'):
    """One dict with every metric above for a prediction log (rows = category x held-out month)."""
    a, p = log[actual_col], log[predicted_col]
    tl = total_level_ape(log, month_col, actual_col, predicted_col)
    return {
        'rows': int(len(log)),
        'test_months': sorted(str(m) for m in log[month_col].unique()),
        'mae': round(mae(a, p), 2),
        'rmse': round(rmse(a, p), 2),
        'wape_pct': round(wape(a, p) * 100, 1),
        'nrmse_pct': round(nrmse(a, p) * 100, 1),
        'bias_pct': round(bias(a, p) * 100, 1),
        'total_level_ape_pct_by_month': {str(m): round(float(v) * 100, 1) for m, v in tl.items()},
        'total_level_mape_pct': round(float(tl.mean()) * 100, 1),
    }


def per_category_wape(log, category_col='Category', actual_col='actual', predicted_col='predicted'):
    """WAPE per category (NaN when the category sold nothing in the held-out months)."""
    return pd.Series({c: wape(g[actual_col], g[predicted_col]) for c, g in log.groupby(category_col)})
