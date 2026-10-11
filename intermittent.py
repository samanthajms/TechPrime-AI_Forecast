"""
intermittent.py
===============
Forecasting for LOW-VOLUME ("sparse") categories, used by inventory_forecasting.get_forecast() and backtest_forecast.py.

The pooled XGBoost model is trained on categories that sell hundreds of units a month. For a category that sells 0-5 a
month it keeps predicting a small positive number (e.g. LAPTOP NO2 / EXTERNAL STORAGE DEVICES never sold in the test window
but are still forecast ~1.6 units, which looks like real stock to order). Croston's method (with the Syntetos-Boylan
correction) forecasts the average demand per month of an intermittent series and goes to ~0 when nothing has sold for
a long time, which is the honest answer for those categories.

Backtest (rolling origin, 13 one-month-ahead forecasts per category, 26 months of history):
    sparse categories, units WAPE:   current XGBoost 194%  ->  Croston/SBA 62%
    sparse categories, revenue WAPE: 6-month average 62%   ->  Croston units x trailing price 54%

Croston is OPT-IN (SPARSE_METHOD=croston). Its sparse threshold is SPARSE_UNITS_PER_MONTH (default 10, compared with the 12-month median; 0 turns it off).

Zero rule (all categories, applied before any model): if the category sold NOTHING in each of the last ZERO_RULE_MONTHS
months (default 3) the forecast is 0 units / 0 revenue. Backtest on this shop's history: the rule fires 92 times at N=3
and demand comes back the very next month only 11 times (12%, 28 units in total) - a model that keeps predicting ~1.6
units for those categories is wrong 88% of the time. ZERO_RULE_MONTHS=0 turns it off.

Which method runs, per category (intermittent.method_for):
    1. zero_rule  - last N months all zero
    2. croston    - only if SPARSE_METHOD=croston AND the category is sparse (see is_sparse)
    3. xgboost    - everything else (default; SPARSE_METHOD defaults to 'xgboost')
"""
import os

import numpy as np

DEFAULT_SPARSE_UNITS_PER_MONTH = 10.0
RECENT_MONTHS = 12
DEFAULT_ZERO_RULE_MONTHS = 3


def sparse_threshold():
    raw = os.environ.get('SPARSE_UNITS_PER_MONTH', '').strip()
    if not raw:
        return DEFAULT_SPARSE_UNITS_PER_MONTH
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_SPARSE_UNITS_PER_MONTH


def is_sparse(units):
    """True when the MEDIAN month of the last 12 sold fewer than SPARSE_UNITS_PER_MONTH units. (Median, not mean: one
    bulk order -- LAPTOP PR3 sold 68 units in a single month -- would otherwise make a 5-a-month category look dense.)"""
    thr = sparse_threshold()
    if thr <= 0:
        return False
    recent = np.asarray(units, float)[-RECENT_MONTHS:]
    return len(recent) > 0 and float(np.median(recent)) < thr


def zero_rule_months():
    raw = os.environ.get('ZERO_RULE_MONTHS', '').strip()
    if not raw:
        return DEFAULT_ZERO_RULE_MONTHS
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_ZERO_RULE_MONTHS


def zero_rule_applies(units):
    """True when each of the last N months (N = ZERO_RULE_MONTHS) had zero sales. Needs at least N months of history."""
    n = zero_rule_months()
    y = np.asarray(units, float)
    return n > 0 and len(y) >= n and float(y[-n:].sum()) == 0.0


def sparse_method():
    return 'croston' if os.environ.get('SPARSE_METHOD', '').strip().lower() == 'croston' else 'xgboost'


def method_for(units):
    """'zero_rule' | 'croston' | 'xgboost' for one category's monthly unit history."""
    if zero_rule_applies(units):
        return 'zero_rule'
    if sparse_method() == 'croston' and is_sparse(units):
        return 'croston'
    return 'xgboost'


def croston_sba(units, alpha=0.1):
    """Syntetos-Boylan approximation of Croston's method: expected units per month for an intermittent series.
    Returns 0.0 for a series with no sales at all."""
    y = np.asarray(units, float)
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return 0.0
    size = y[nz[0]]                          # smoothed size of a non-zero month
    interval = float(nz[0] + 1)              # smoothed number of months between sales
    gap = 1
    for t in range(nz[0] + 1, len(y)):
        gap += 1
        if y[t] > 0:
            size += alpha * (y[t] - size)
            interval += alpha * (gap - interval)
            gap = 0
    return float((1 - alpha / 2) * size / max(interval, 1.0))


def trailing_price(units, revenue, months=RECENT_MONTHS):
    """Revenue per unit over the last `months` months (0.0 if nothing sold)."""
    u, r = np.asarray(units, float)[-months:], np.asarray(revenue, float)[-months:]
    return float(r.sum() / u.sum()) if u.sum() > 0 else 0.0
