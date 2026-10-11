"""
tune_xgboost.py
===============
XGBoost-only tuning experiments on the same rolling-origin backtest as backtest_forecast.py (XGBoost stays the model;
this only asks "which XGBoost setup is best on OUR data?"). READ-ONLY - never touches the deployed artifacts.

Every candidate is trained at every origin on months < t only and scored on month t. Anything that is chosen from data
(hyper-parameters, tree count) is chosen by a time-series cross-validation INSIDE the training months, never on the
month being scored, so the numbers are honest.

Candidates (all XGBoost):
  xgb_current          what rebuild_models_v2.py does today (reference)
  + zero_N             rule: if the category sold nothing in the last N months, forecast 0
  poisson / tweedie    count-data objectives instead of squared error (poisson is a drop-in: same features, no serving change)
  log1p                train on log(1 + units)
  clean                outliers in the HISTORY (one-off bulk orders) capped before features are built
  scaled               scale-free pooled model: every row divided by the category's own level, no category_code
  scaled+bag           scaled, averaged over 5 seeds
  scaled+cv            scaled, hyper-parameters picked by inner time-series CV
  scaled+recent        scaled, recent months weighted more
  combo / final        scaled + clean + bag + zero rule (+ inner-CV params for `final`) = the recommended stack

USAGE:   python tune_xgboost.py [--features preprocessed_monthly_features.csv] [--first-origin 13] [--csv out.csv]
"""
import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import backtest_forecast as bt

warnings.filterwarnings('ignore')
SEEDS = (11, 22, 33, 44, 55)

BASE_PARAMS = dict(max_depth=3, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, objective='reg:squarederror',
                   random_state=42)
SCALED_PARAMS = dict(max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                     reg_lambda=5.0, objective='reg:squarederror', random_state=42)


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------
def clean_history(wide, k=3.0, floor=5.0):
    """Cap one-off spikes using ONLY earlier months: y[t] <= k * (median of the previous 12 months + 1) + floor.
    (LAPTOP PR3 sold 68 units in one month against a normal 4-7.)"""
    out = wide.copy().astype(float)
    for c in out.columns:
        y = out[c].values.copy()
        for t in range(3, len(y)):
            cap = k * (np.median(y[max(0, t - 12):t]) + 1.0) + floor
            if y[t] > cap:
                y[t] = cap
        out[c] = y
    return out


def zero_rule(pred, test, hist, n):
    """If a category sold nothing in each of the last n months, the forecast is 0."""
    pred = np.array(pred, float)
    for i, (c, m) in enumerate(zip(test['Category'], test['m'])):
        m = int(m)
        if m >= n and hist[c].values[m - n:m].sum() == 0:
            pred[i] = 0.0
    return pred


def _tf(y, log):
    return np.log1p(y) if log else y


def _inv(p, log):
    return np.expm1(p) if log else p


def _weights(d, recency):
    if not recency:
        return None
    age = d['m'].max() - d['m'].values
    return 0.5 ** (age / recency)                         # half-life of `recency` months


def fit_predict(train, test, features, params, scaled=False, log=False, seeds=(42,), recency=None, n_trees=None,
                use_target='target'):
    """Fit on `train`, predict `test`. Tree count by early stopping on the last 3 TRAIN months unless n_trees given."""
    tr, te = train, test
    if scaled:
        tr, te = bt._add_scaled(train), bt._add_scaled(test)
        tr = tr[tr['m'] >= 3]
    months = sorted(tr['m'].unique())

    def target(d):
        y = d[use_target].values / (d['scale'].values if scaled else 1.0)
        return _tf(y, log)

    if n_trees is None:
        fit, val = tr[tr['m'].isin(months[:-3])], tr[tr['m'].isin(months[-3:])]
        probe = xgb.XGBRegressor(n_estimators=300, early_stopping_rounds=15, **params)
        probe.fit(fit[features], target(fit), sample_weight=_weights(fit, recency),
                  eval_set=[(val[features], target(val))], verbose=False)
        n_trees = int(probe.best_iteration) + 1
    preds = []
    for sd in seeds:
        p = dict(params); p['random_state'] = sd
        mdl = xgb.XGBRegressor(n_estimators=n_trees, **p)
        mdl.fit(tr[features], target(tr), sample_weight=_weights(tr, recency), verbose=False)
        preds.append(_inv(mdl.predict(te[features]), log))
    out = np.clip(np.mean(preds, axis=0), 0, None)
    return out * te['scale'].values if scaled else out


# inner time-series CV: pick (params, n_trees) using ONLY the training months
GRID = [dict(max_depth=d, learning_rate=lr, min_child_weight=mcw, reg_lambda=l2, subsample=ss, colsample_bytree=cs,
             objective='reg:squarederror', random_state=42, n_estimators=n)
        for d, lr, mcw, l2, ss, cs, n in itertools.product([1, 2, 3], [0.03, 0.08], [3, 10], [1.0, 10.0], [0.8], [0.8], [80, 250])]


def inner_cv_params(train, features, folds=3, scaled=True, objective='reg:squarederror'):
    """Score every grid config by volume-weighted absolute error on the last `folds` months of the training window,
    each fold trained on the months before it. Returns (params, n_trees) of the best config."""
    tr = bt._add_scaled(train) if scaled else train.copy()
    tr = tr[tr['m'] >= 3]
    if not scaled:
        tr['scale'] = 1.0
    last = int(tr['m'].max())
    best, best_err = None, np.inf
    for cfg in GRID:
        cfg = dict(cfg); n = cfg.pop('n_estimators'); cfg['objective'] = objective
        err, tot = 0.0, 0.0
        for f in range(last - folds + 1, last + 1):
            fit, val = tr[tr['m'] < f], tr[tr['m'] == f]
            mdl = xgb.XGBRegressor(n_estimators=n, **cfg)
            mdl.fit(fit[features], fit['target'] / fit['scale'], verbose=False)
            p = np.clip(mdl.predict(val[features]), 0, None) * val['scale'].values
            err += np.abs(p - val['target'].values).sum(); tot += val['target'].sum()
        score = err / max(tot, 1e-9)
        if score < best_err:
            best, best_err = (cfg, n), score
    return best


# ----------------------------------------------------------------------------------------------------------------
# candidates.  Signature: (train, test, hist_raw, hist_clean) -> predictions
# ----------------------------------------------------------------------------------------------------------------
CUR = bt.CURRENT_FEATURES
SC = bt.SCALED_FEATURES


def c_current(tr, te, h, hc):
    return fit_predict(tr, te, CUR, BASE_PARAMS)


def c_zero(n, base=c_current):
    return lambda tr, te, h, hc: zero_rule(base(tr, te, h, hc), te, h, n)


def c_poisson(tr, te, h, hc):
    return fit_predict(tr, te, CUR, dict(BASE_PARAMS, objective='count:poisson'))


def c_tweedie(tr, te, h, hc):
    return fit_predict(tr, te, CUR, dict(BASE_PARAMS, objective='reg:tweedie', tweedie_variance_power=1.3))


def c_log(tr, te, h, hc):
    return fit_predict(tr, te, CUR, BASE_PARAMS, log=True)


def c_scaled(tr, te, h, hc):
    return fit_predict(tr, te, SC, SCALED_PARAMS, scaled=True)


def c_scaled_bag(tr, te, h, hc):
    return fit_predict(tr, te, SC, SCALED_PARAMS, scaled=True, seeds=SEEDS)


def c_scaled_recent(tr, te, h, hc):
    return fit_predict(tr, te, SC, SCALED_PARAMS, scaled=True, recency=6)


def c_scaled_huber(tr, te, h, hc):
    return fit_predict(tr, te, SC, dict(SCALED_PARAMS, objective='reg:pseudohubererror', huber_slope=0.5), scaled=True)


def c_scaled_cv(tr, te, h, hc):
    cfg, n = inner_cv_params(tr, SC)
    return fit_predict(tr, te, SC, cfg, scaled=True, n_trees=n)


def c_poisson_cv(tr, te, h, hc):
    """Drop-in for the deployed feature set: same 10 features, Poisson objective, params from inner CV."""
    cfg, n = inner_cv_params(tr, CUR, scaled=False, objective='count:poisson')
    return fit_predict(tr, te, CUR, cfg, n_trees=n)


def c_final(tr, te, h, hc):
    """Scale-free features + inner-CV params + 5-seed bagging (history cleaned by the caller) ."""
    cfg, n = inner_cv_params(tr, SC)
    return fit_predict(tr, te, SC, cfg, scaled=True, n_trees=n, seeds=SEEDS)


CANDIDATES = {                     # name -> (function, uses cleaned history?)
    'xgb_current': (c_current, False),
    'xgb_current+zero3': (c_zero(3), False),
    'xgb_current+zero6': (c_zero(6), False),
    'xgb_poisson': (c_poisson, False),
    'xgb_tweedie': (c_tweedie, False),
    'xgb_log1p': (c_log, False),
    'poisson+zero3': (c_zero(3, c_poisson), False),
    'poisson+cv+zero3': (c_zero(3, c_poisson_cv), False),
    'final(scaled+cv+clean+bag+zero3)': (c_zero(3, c_final), True),
    'xgb_current+clean': (c_current, True),
    'xgb_scaled': (c_scaled, False),
    'scaled+zero3': (c_zero(3, c_scaled), False),
    'scaled+zero6': (c_zero(6, c_scaled), False),
    'scaled+clean': (c_scaled, True),
    'scaled+huber': (c_scaled_huber, False),
    'scaled+recent': (c_scaled_recent, False),
    'scaled+bag': (c_scaled_bag, False),
    'scaled+cv': (c_scaled_cv, False),
    'combo(scaled+clean+bag+zero3)': (c_zero(3, c_scaled_bag), True),
    'combo(scaled+clean+bag+zero6)': (c_zero(6, c_scaled_bag), True),
}


def evaluate(wide, first_origin, dense_min, names):
    wide_clean = clean_history(wide)
    long_raw, long_clean = bt.build_long_features(wide), bt.build_long_features(wide_clean)
    # evaluate against RAW actuals; cleaned frames only change features / training targets
    raw_target = long_raw.set_index(['Category', 'm'])['target']
    rows = []
    for t in range(first_origin, len(wide)):
        for name in names:
            fn, use_clean = CANDIDATES[name]
            L, H = (long_clean, wide_clean) if use_clean else (long_raw, wide)
            train, test = L[L['m'] < t], L[L['m'] == t]
            pred = np.clip(fn(train, test, wide, H), 0, None)
            for cat, p in zip(test['Category'], pred):
                rows.append({'model': name, 'origin': str(wide.index[t]), 'Category': cat,
                             'actual': raw_target.loc[(cat, t)], 'forecast': float(p)})
        print(f"  origin {wide.index[t]} done", flush=True)
    res = pd.DataFrame(rows)
    res['naive_mae'] = res['Category'].map({c: np.abs(np.diff(wide[c].values[:first_origin])).mean() for c in wide.columns})
    avg_act = res[res['model'] == names[0]].groupby('Category')['actual'].mean()
    res['group'] = np.where(res['Category'].map(avg_act) >= dense_min, 'dense', 'sparse')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default=str(bt.FEATURES_PATH))
    ap.add_argument('--first-origin', type=int, default=13)
    ap.add_argument('--dense-min', type=float, default=10)
    ap.add_argument('--models', default=','.join(CANDIDATES))
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()
    wide = bt.load_matrix(Path(args.features))
    names = [m.strip() for m in args.models.split(',') if m.strip()]
    print(f"{wide.shape[1]} categories, {wide.shape[0]} months; scoring {wide.shape[0] - args.first_origin} origins x {len(names)} candidates")
    res = evaluate(wide, args.first_origin, args.dense_min, names)
    summ = bt.summarise(res)
    pd.set_option('display.width', 170)
    for g in ('ALL', 'dense', 'sparse'):
        print(f"\n=== {g} ===")
        print(summ[summ['group'] == g].drop(columns='group').to_string(index=False, float_format=lambda v: f'{v:0.3f}'))
    if args.csv:
        res.to_csv(args.csv, index=False)


if __name__ == '__main__':
    main()
