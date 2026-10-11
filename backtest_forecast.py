"""
backtest_forecast.py
====================
Rolling-origin backtest for the TechPrime AI demand model. READ-ONLY: it never touches the deployed artifacts.

WHY: the 80/20 gate in rebuild_models_v2.py scores ONE ~5-month window, and with 26 months of history that number moves
a lot with the window. This script re-trains at every origin ("what would we have forecast for month t, knowing only
months < t?") and scores the next month, then compares the current XGBoost setup with naive baselines and a few
candidate improvements, so a change is only adopted when it beats the baselines across many origins.

Metrics (per category and pooled):
  WAPE  = sum|actual - forecast| / sum(actual)          (volume-weighted, does not blow up on small categories)
  MASE  = MAE / MAE of the one-step naive forecast on the training months   (<1 means better than "same as last month")
  BIAS  = sum(forecast - actual) / sum(actual)
Categories are split into DENSE (avg >= --dense-min units/month over the test origins) and SPARSE; WAPE/MASE on a category
that sells ~1 unit a month is noise, so the two groups are reported separately.

USAGE (from the repo root, next to preprocessed_monthly_features.csv):
    python backtest_forecast.py
    python backtest_forecast.py --first-origin 14 --dense-min 10 --csv backtest_results.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

from intermittent import croston_sba   # same implementation the service uses

warnings.filterwarnings('ignore')
BASE_DIR = Path(__file__).parent
FEATURES_PATH = BASE_DIR / 'preprocessed_monthly_features.csv'

# the features the deployed demand model uses today
CURRENT_FEATURES = ['lag_1', 'lag_2', 'lag_3', 'lag_12', 'roll_mean_3', 'roll_mean_6', 'roll_std_3',
                    'month_num', 'quarter', 'category_code']


# ----------------------------------------------------------------------------------------------------------------
# data: wide matrix (months x categories) of units sold + leak-free feature frames
# ----------------------------------------------------------------------------------------------------------------
def load_matrix(path=FEATURES_PATH):
    df = pd.read_csv(path)
    df = df[df['Category'] != 'All']                     # 'All' is an aggregate row, not a product bucket
    wide = df.pivot(index='Month', columns='Category', values='units_sold').sort_index().fillna(0.0)
    wide.index = pd.PeriodIndex(wide.index, freq='M')
    return wide


def build_long_features(wide):
    """One row per (category, month m) whose features use ONLY months < m (no leakage by construction)."""
    codes = {c: i for i, c in enumerate(wide.columns)}
    rows = []
    for cat in wide.columns:
        y = wide[cat].values.astype(float)
        for m in range(len(y)):
            def lag(k):
                return y[m - k] if m - k >= 0 else np.nan
            prev3, prev6 = y[max(0, m - 3):m], y[max(0, m - 6):m]
            month = wide.index[m].month
            rows.append({
                'Category': cat, 'm': m, 'target': y[m],
                'lag_1': lag(1) if m >= 1 else 0.0, 'lag_2': lag(2) if m >= 2 else 0.0, 'lag_3': lag(3) if m >= 3 else 0.0,
                'lag_6': lag(6), 'lag_12': lag(12) if m >= 12 else (lag(3) if m >= 3 else 0.0),
                'roll_mean_3': prev3.mean() if len(prev3) else 0.0,
                'roll_mean_6': prev6.mean() if len(prev6) else 0.0,
                'roll_std_3': prev3.std(ddof=1) if len(prev3) > 1 else 0.0,
                'roll_mean_12': y[max(0, m - 12):m].mean() if m >= 1 else 0.0,
                'nonzero_share_6': (prev6 > 0).mean() if len(prev6) else 0.0,
                'month_num': month, 'quarter': (month - 1) // 3 + 1,
                'month_sin': np.sin(2 * np.pi * month / 12), 'month_cos': np.cos(2 * np.pi * month / 12),
                'peak_month': int(month in (6, 7, 8, 11, 12)),   # back-to-school (Jun-Aug), 11.11 / 12.12 / Christmas
                'category_code': codes[cat],
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
# forecasters.  Each takes (train_long, test_long, wide_history) and returns predictions aligned with test_long.
# ----------------------------------------------------------------------------------------------------------------
def f_naive(train, test, hist):                        # same as last month
    return test['lag_1'].values


def f_ma3(train, test, hist):
    return test['roll_mean_3'].values


def f_ma6(train, test, hist):
    return test['roll_mean_6'].values


def f_seasonal_blend(train, test, hist):              # half "same month last year", half "recent level" (needs 12 months)
    has12 = test['m'].values >= 12
    seas = np.where(has12, test['lag_12'].values, test['roll_mean_6'].values)
    return 0.5 * seas + 0.5 * test['roll_mean_6'].values


def f_croston(train, test, hist):
    out = []
    for _, r in test.iterrows():
        out.append(croston_sba(hist[r['Category']].values[:int(r['m'])]))
    return np.array(out)


def _fit_xgb(train, features, target, params, val_months=3, scale=None, log=False, rounds=300):
    """Early-stop on the last `val_months` TRAIN months only, then refit on all train rows with that tree count."""
    months = sorted(train['m'].unique())
    fit, val = train[train['m'].isin(months[:-val_months])], train[train['m'].isin(months[-val_months:])]

    def tf(d):
        y = d[target].values
        if scale is not None:
            y = y / d[scale].values
        return np.log1p(y) if log else y
    probe = xgb.XGBRegressor(n_estimators=rounds, early_stopping_rounds=15, eval_metric='rmse', **params)
    probe.fit(fit[features], tf(fit), eval_set=[(val[features], tf(val))], verbose=False)
    model = xgb.XGBRegressor(n_estimators=int(probe.best_iteration) + 1, **params)
    model.fit(train[features], tf(train), verbose=False)
    return model


def f_xgb_current(train, test, hist):
    """What rebuild_models_v2.py does today (same features and hyper-parameters)."""
    params = dict(max_depth=3, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, objective='reg:squarederror',
                  random_state=42)
    mdl = _fit_xgb(train, CURRENT_FEATURES, 'target', params)
    return np.clip(mdl.predict(test[CURRENT_FEATURES]), 0, None)


# candidate: scale-free pooled model. Every row is divided by the category's own level (roll_mean_6 + 1) so the model
# learns "next month relative to recent level" shared across all categories instead of memorising each category's size.
SCALED_FEATURES = ['r_lag_1', 'r_lag_2', 'r_lag_3', 'r_lag_6', 'r_lag_12', 'r_roll_3', 'r_roll_12', 'cv_3',
                   'nonzero_share_6', 'month_sin', 'month_cos', 'peak_month']


def _add_scaled(d):
    d = d.copy()
    d['scale'] = d['roll_mean_6'] + 1.0
    for k in ('lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12'):
        d['r_' + k] = d[k] / d['scale']
    d['r_roll_3'] = d['roll_mean_3'] / d['scale']
    d['r_roll_12'] = d['roll_mean_12'] / d['scale']
    d['cv_3'] = d['roll_std_3'] / d['scale']
    return d


def f_xgb_scaled(train, test, hist):
    params = dict(max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=5.0, objective='reg:squarederror', random_state=42)
    tr, te = _add_scaled(train), _add_scaled(test)
    tr = tr[tr['m'] >= 3]                                     # drop warm-up rows with no history
    mdl = _fit_xgb(tr, SCALED_FEATURES, 'target', params, scale='scale')
    return np.clip(mdl.predict(te[SCALED_FEATURES]), 0, None) * te['scale'].values


def f_blend(train, test, hist):
    """50/50 of the scale-free XGBoost and a 3-month moving average: cheap, robust, hard to beat on small data."""
    return 0.5 * f_xgb_scaled(train, test, hist) + 0.5 * f_ma3(train, test, hist)


def f_blend6(train, test, hist):
    """50/50 of the scale-free XGBoost and a 6-month moving average."""
    return 0.5 * f_xgb_scaled(train, test, hist) + 0.5 * f_ma6(train, test, hist)


def f_routed(train, test, hist, dense_min=10):
    """Dense categories -> XGBoost/6-month-average blend; intermittent categories -> Croston/SBA (zero is a legitimate
    forecast there, the current model instead keeps predicting a phantom 1-2 units)."""
    dense = np.array([np.median(hist[c].values[max(0, int(m) - 12):int(m)]) >= dense_min      # same median rule as intermittent.is_sparse
                      for c, m in zip(test['Category'], test['m'])])
    return np.where(dense, f_blend6(train, test, hist), f_croston(train, test, hist))


MODELS = {
    'naive_last_month': f_naive,
    'moving_avg_3': f_ma3,
    'moving_avg_6': f_ma6,
    'seasonal_blend': f_seasonal_blend,
    'croston_sba': f_croston,
    'xgb_current': f_xgb_current,
    'xgb_scaled': f_xgb_scaled,
    'blend_xgb_ma3': f_blend,
    'blend_xgb_ma6': f_blend6,
    'routed_blend_croston': f_routed,
}


# ----------------------------------------------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------------------------------------------
def run(wide, first_origin, dense_min, models):
    long_df = build_long_features(wide)
    n = len(wide)
    results = []
    for t in range(first_origin, n):                     # origin t: train on months < t, forecast month t
        train, test = long_df[long_df['m'] < t], long_df[long_df['m'] == t]
        for name in models:
            pred = np.clip(MODELS[name](train, test, wide), 0, None)
            for cat, act, p, lag1 in zip(test['Category'], test['target'], pred, test['lag_1']):
                results.append({'model': name, 'origin': str(wide.index[t]), 'Category': cat, 'actual': act,
                                'forecast': float(p)})
    res = pd.DataFrame(results)
    # in-sample naive MAE per category (scale for MASE) over all months before first origin
    naive_mae = {c: np.abs(np.diff(wide[c].values[:first_origin])).mean() for c in wide.columns}
    res['naive_mae'] = res['Category'].map(naive_mae)
    avg_act = res[res['model'] == 'naive_last_month'].groupby('Category')['actual'].mean()
    res['group'] = np.where(res['Category'].map(avg_act) >= dense_min, 'dense', 'sparse')
    return res


def summarise(res):
    def agg(g):
        a, f = g['actual'].values, g['forecast'].values
        err = np.abs(a - f)
        mase_den = g.groupby('Category')['naive_mae'].first()
        mae_cat = g.assign(e=err).groupby('Category')['e'].mean()
        ok = mase_den > 0
        return pd.Series({'WAPE': err.sum() / max(a.sum(), 1e-9),
                          'MASE': float((mae_cat[ok] / mase_den[ok]).mean()) if ok.any() else np.nan,
                          'BIAS': (f.sum() - a.sum()) / max(a.sum(), 1e-9), 'MAE': err.mean()})
    out = res.groupby(['group', 'model']).apply(agg).reset_index()
    pooled = res.groupby('model').apply(agg).reset_index()
    pooled.insert(0, 'group', 'ALL')
    return pd.concat([pooled, out]).sort_values(['group', 'WAPE']).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default=str(FEATURES_PATH))
    ap.add_argument('--first-origin', type=int, default=13, help="index of the first month to forecast (default 13 = need 13 months of history)")
    ap.add_argument('--dense-min', type=float, default=10, help="avg units/month at/above which a category counts as dense")
    ap.add_argument('--models', default=','.join(MODELS), help="comma-separated subset of: " + ', '.join(MODELS))
    ap.add_argument('--csv', default=None, help="write the per-category / per-origin results here")
    args = ap.parse_args()

    wide = load_matrix(Path(args.features))
    names = [m.strip() for m in args.models.split(',') if m.strip()]
    print(f"{wide.shape[1]} categories, {wide.shape[0]} months ({wide.index[0]}..{wide.index[-1]}); "
          f"origins: {wide.index[args.first_origin]}..{wide.index[-1]} ({wide.shape[0] - args.first_origin} one-month-ahead forecasts per category)")
    res = run(wide, args.first_origin, args.dense_min, names)
    summ = summarise(res)
    pd.set_option('display.width', 160)
    print(summ.to_string(index=False, float_format=lambda v: f'{v:0.3f}'))
    d = res[res['group'] == 'dense']
    if 'moving_avg_6' in names and len(names) > 1:
        wape_o = d.assign(e=(d['actual'] - d['forecast']).abs()).groupby(['model', 'origin']).agg(e=('e', 'sum'), a=('actual', 'sum'))
        wape_o = (wape_o['e'] / wape_o['a']).unstack(0)
        print("\nDENSE: share of origins (months) where the model's WAPE beats moving_avg_6  [ranking stability]")
        for m in wape_o.columns:
            if m != 'moving_avg_6':
                print(f"  {m:24s} {(wape_o[m] < wape_o['moving_avg_6']).mean():.0%} of {len(wape_o)} origins")
    dense = sorted(res[res['group'] == 'dense']['Category'].unique())
    print(f"\ndense categories ({len(dense)}): {dense}")
    print(f"sparse categories ({res['Category'].nunique() - len(dense)}): "
          f"{sorted(res[res['group'] == 'sparse']['Category'].unique())}")
    print("\nHow to read: adopt a candidate only if its WAPE/MASE beat 'naive_last_month' AND 'xgb_current' in the DENSE "
          "group, and its BIAS stays near 0. MASE < 1 = better than repeating last month.")
    if args.csv:
        res.to_csv(args.csv, index=False)
        print(f"per-forecast results written to {args.csv}")


if __name__ == '__main__':
    main()
