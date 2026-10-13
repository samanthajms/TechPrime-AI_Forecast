"""
rebuild_models_v2.py
=====================
Monthly retrain script for the XGBoost demand model, plus the held-out accuracy of the derived revenue forecast.

THIS IS THE SCRIPT `inventory_forecasting.py` REFERS TO under "WHAT TO SWAP
FOR PRODUCTION" -- run it every time you retrain (recommended: monthly, once
a new actual month of transactions has landed).

DESIGN (matches the "80/20 gate vs. ongoing accuracy log" plan):
  1. PRE-DEPLOYMENT GATE -- chronological 80/20 split, REDRAWN each time this
     script runs. As more actual months accumulate, the 80/20 cutoff moves
     forward automatically -- you never validate against a stale slice.
     This is a generalization check ("does the newly retrained model predict
     unseen recent months reasonably well?"), NOT the long-run scoreboard.
  2. The long-run scoreboard is `compare_actual_vs_forecast()` in
     inventory_forecasting.py, which logs real forecasts served in
     production against actuals once each month closes. That log lives
     independently of this script and is never overwritten by a retrain.

WHAT THIS SCRIPT DOES:
  - Loads preprocessed_monthly_features.csv (swap for a real feature-build
    query in production, same as `_load_transactions()` in
    inventory_forecasting.py)
  - Splits chronologically: first 80% of available months = train,
    most recent 20% = test. The split is by MONTH, not by row, so no
    category's test rows are ever chronologically earlier than its train rows.
  - Trains the demand model fresh on the 80% train months. Revenue is NOT a second learned model: it is predicted units x
    the mean of the category's last 3 monthly average prices (the same rule inventory_forecasting.get_forecast uses), so the
    revenue accuracy files below describe that derivation. (A separate XGBoost revenue model scored 35.0% WAPE against 18.7%
    for this rule on the same held-out months and was retired.) Early stopping uses
    only the last few TRAIN months, so the 20% test months are never used
    for fitting or tuning and the reported test error is out-of-sample
  - Writes/overwrites:
      xgb_monthly_demand_model.json
      category_confidence.csv, revenue_category_confidence.csv   (per-category
          MAE/RMSE/avg_actual over the held-out 20% -- feeds
          get_forecast_confidence() and flag_high_variance_categories())
      monthly_predictions.csv, revenue_predictions.csv           (per-category,
          per-month actual vs. predicted over the held-out 20% -- feeds
          compare_actual_vs_forecast() as a demo log; replace with a real
          forecast_log table in production, per that function's docstring)
  - Regression gate (ON by default): if the new models' overall held-out
    relative error is worse than the deployed models', NOTHING is overwritten
    and the script exits with status 2 (so a scheduled job fails loudly).
    Pass --no-enforce-gate to override.
  - Prints a PRE-DEPLOYMENT GATE report: overall + per-category relative
    error, and which categories fail the reliability threshold (informational).

USAGE:
    python rebuild_models_v2.py
    python rebuild_models_v2.py --gate-threshold 0.35
    python rebuild_models_v2.py --no-enforce-gate
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import accuracy_metrics as am

BASE_DIR = Path(__file__).parent

FEATURES_PATH = BASE_DIR / 'preprocessed_monthly_features.csv'
DEMAND_MODEL_PATH = BASE_DIR / 'xgb_monthly_demand_model.json'
CATEGORY_CONFIDENCE_PATH = BASE_DIR / 'category_confidence.csv'
REVENUE_CONFIDENCE_PATH = BASE_DIR / 'revenue_category_confidence.csv'
MONTHLY_PREDICTIONS_PATH = BASE_DIR / 'monthly_predictions.csv'
REVENUE_PREDICTIONS_PATH = BASE_DIR / 'revenue_predictions.csv'
ACCURACY_PATH = BASE_DIR / 'model_accuracy.json'      # MAE / RMSE / WAPE / NRMSE / bias / total-level error of the held-out months

DEMAND_FEATURES = ['lag_1', 'lag_2', 'lag_3', 'lag_12', 'roll_mean_3', 'roll_mean_6',
                    'roll_std_3', 'month_num', 'quarter', 'category_code']
TRAIN_FRACTION = 0.8  # 80/20 -- redrawn fresh every run, not a fixed historical slice


def _chronological_split(df, train_fraction=TRAIN_FRACTION):
    """
    Split by MONTH (not by row) so the split point is identical across every
    category: everything before the cutoff month is train, everything from
    the cutoff month onward is test. This is what makes the 80/20 a genuine
    "can this model predict months it has never seen" check rather than a
    randomly-shuffled fit-quality check.
    """
    months = sorted(df['Month'].unique())
    if len(months) < 5:
        raise ValueError(
            f"Only {len(months)} distinct months available -- need more "
            "history before a chronological 80/20 split is meaningful.")

    cutoff_idx = max(1, int(round(len(months) * train_fraction)))
    cutoff_idx = min(cutoff_idx, len(months) - 1)  # always leave >=1 test month
    train_months, test_months = months[:cutoff_idx], months[cutoff_idx:]

    train_df = df[df['Month'].isin(train_months)].copy()
    test_df = df[df['Month'].isin(test_months)].copy()
    return train_df, test_df, train_months, test_months


VAL_MONTHS = 3  # most recent TRAIN months held out for early stopping (never the 20% test months)


def _train_one_model(train_df, test_df, features, target_col, objective='reg:squarederror'):
    """Fit on the 80% train months only. Early stopping picks the number of trees using the last VAL_MONTHS train
    months, then the model is refit on the full 80% with that tree count. The 20% test months are never seen
    during fitting or tuning, so the reported test error is an honest out-of-sample number."""
    params = dict(max_depth=3, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                  objective=objective, random_state=42)

    months = sorted(train_df['Month'].unique())
    n_val = min(VAL_MONTHS, max(1, len(months) // 5))
    fit_df = train_df[train_df['Month'].isin(months[:-n_val])]
    val_df = train_df[train_df['Month'].isin(months[-n_val:])]

    probe = xgb.XGBRegressor(n_estimators=200, early_stopping_rounds=15, eval_metric='rmse', **params)
    probe.fit(fit_df[features], fit_df[target_col], eval_set=[(val_df[features], val_df[target_col])], verbose=False)
    best_n = int(probe.best_iteration) + 1

    model = xgb.XGBRegressor(n_estimators=best_n, **params)
    model.fit(train_df[features], train_df[target_col], verbose=False)
    return model


def _per_category_confidence(test_df, predictions, target_col):
    out = test_df[['Month', 'Category', target_col]].copy()
    out['predicted'] = predictions
    out = out.rename(columns={target_col: 'actual'})

    rows = []
    for category, g in out.groupby('Category'):
        err = g['actual'] - g['predicted']
        rows.append({
            'Category': category,
            'mae': float(err.abs().mean()),
            'rmse': float(np.sqrt((err ** 2).mean())),
            'avg_actual': float(g['actual'].mean()),
            'wape': am.wape(g['actual'], g['predicted']),      # scale-free, comparable across categories
        })
    confidence = pd.DataFrame(rows).set_index('Category')
    return out, confidence


def _gate_report(confidence, target_name, threshold):
    df = confidence.copy()
    df['relative_error'] = df['rmse'] / df['avg_actual'].replace(0, np.nan)
    overall_relative_error = float(df['rmse'].mean() / df['avg_actual'].replace(0, np.nan).mean())

    print(f"\n--- PRE-DEPLOYMENT GATE: {target_name} ---")
    print(f"Overall avg relative error (equal-weighted NRMSE, RMSE / avg_actual): "
          f"{overall_relative_error:.1%}" if np.isfinite(overall_relative_error) else "N/A")

    failing = df[df['relative_error'] > threshold].sort_values('relative_error', ascending=False)
    if failing.empty:
        print(f"All categories within {threshold:.0%} relative-error threshold.")
    else:
        print(f"{len(failing)} categor(y/ies) EXCEED {threshold:.0%} relative-error threshold "
              f"-- review before trusting these in production, or route them to manual ordering:")
        for cat, row in failing.iterrows():
            print(f"  - {cat}: relative error {row['relative_error']:.1%} "
                  f"(rmse={row['rmse']:.2f}, avg_actual={row['avg_actual']:.2f})")
    return overall_relative_error, failing


ARTIFACTS = [DEMAND_MODEL_PATH, CATEGORY_CONFIDENCE_PATH, REVENUE_CONFIDENCE_PATH,
             MONTHLY_PREDICTIONS_PATH, REVENUE_PREDICTIONS_PATH, ACCURACY_PATH]


def _baselines(test_df, target_col, lag_col, roll6_col):
    """WAPE of two no-model forecasts on the same held-out rows. The XGBoost model has to beat these to be worth using."""
    a = test_df[target_col]
    return {'naive_last_month_wape_pct': round(am.wape(a, test_df[lag_col]) * 100, 1),
            'moving_avg_6m_wape_pct': round(am.wape(a, test_df[roll6_col]) * 100, 1)}


def _accuracy_report(name, summary, unit):
    """Print the study metrics for one model. WAPE and the total-level error are the headline numbers; the
    equal-weighted 'relative error' printed by _gate_report() lets tiny categories dominate the average."""
    print(f"\n--- ACCURACY ({name}, held-out months {', '.join(summary['test_months'])}) ---")
    print(f"MAE   {summary['mae']:,.2f} {unit}   RMSE  {summary['rmse']:,.2f} {unit}   (scale-dependent, pooled over all categories)")
    print(f"WAPE  {summary['wape_pct']:.1f}%   NRMSE {summary['nrmse_pct']:.1f}%   Bias {summary['bias_pct']:+.1f}%   (scale-free)")
    by_month = ', '.join(f"{m}: {v:.1f}%" for m, v in summary['total_level_ape_pct_by_month'].items())
    print(f"Total-level error (all categories summed): mean {summary['total_level_mape_pct']:.1f}%  [{by_month}]")
    b = summary['baselines']
    print(f"Benchmarks (WAPE, no model): last month {b['naive_last_month_wape_pct']:.1f}%   6-month average {b['moving_avg_6m_wape_pct']:.1f}%"
          f"   -> the model should be at or below these")


def _overall_rel_error(conf_df):
    mean_actual = conf_df['avg_actual'].replace(0, np.nan).mean()
    return float(conf_df['rmse'].mean() / mean_actual)


def _passes_regression_gate(demand_confidence, revenue_confidence):
    """New models must not be worse than the deployed ones (compared on each model's own held-out relative error)."""
    ok = True
    for name, new_conf, path in [('demand', demand_confidence, CATEGORY_CONFIDENCE_PATH),
                                 ('revenue', revenue_confidence, REVENUE_CONFIDENCE_PATH)]:
        if not path.exists():
            continue
        old, new = _overall_rel_error(pd.read_csv(path).set_index('Category')), _overall_rel_error(new_conf)
        print(f"Gate [{name}]: deployed {old:.1%} -> new {new:.1%}")
        if not np.isfinite(new) or (np.isfinite(old) and new > old):
            ok = False
    return ok


def _backup_artifacts():
    """Copy the current artifacts to artifacts_backup/<timestamp>/ so a bad retrain can be rolled back."""
    existing = [p for p in ARTIFACTS if p.exists()]
    if not existing:
        return None
    dest = BASE_DIR / 'artifacts_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
    dest.mkdir(parents=True, exist_ok=True)
    for p in existing:
        shutil.copy2(p, dest / p.name)
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate-threshold', type=float, default=0.35,
                         help="Relative-error (RMSE/avg_actual) threshold above which a "
                              "category is flagged as unreliable. Default 0.35, matching "
                              "flag_high_variance_categories()'s default in inventory_forecasting.py.")
    parser.add_argument('--enforce-gate', action=argparse.BooleanOptionalAction, default=True,
                         help="(default on) Do not overwrite the live artifacts if the new models' overall relative "
                              "error is worse than the currently deployed models'. --no-enforce-gate to override. "
                              "Backups are always taken before an overwrite.")
    parser.add_argument('--train-fraction', type=float, default=TRAIN_FRACTION,
                         help="Chronological train fraction. Default 0.8 (80/20).")
    args = parser.parse_args()

    df = pd.read_csv(FEATURES_PATH)
    df['Month'] = pd.PeriodIndex(df['Month'], freq='M')

    train_df, test_df, train_months, test_months = _chronological_split(df, args.train_fraction)
    print(f"Chronological split (redrawn this run): "
          f"train = {train_months[0]}..{train_months[-1]} ({len(train_months)} months), "
          f"test = {test_months[0]}..{test_months[-1]} ({len(test_months)} months)")

    # ---------------- Demand model ----------------
    # Units are non-negative counts: the Poisson objective beat squared error in both backtests run on this project
    # (26-month POS history and the 13-month sample). Revenue is continuous, so it keeps squared error.
    demand_model = _train_one_model(train_df, test_df, DEMAND_FEATURES, 'units_sold', objective='count:poisson')
    demand_preds = np.clip(demand_model.predict(test_df[DEMAND_FEATURES]), 0, None)
    demand_log, demand_confidence = _per_category_confidence(test_df, demand_preds, 'units_sold')

    # ---------------- Revenue = predicted units x expected unit price (no second model) ----------------
    # avg_price_roll_mean_3 = mean of the 3 monthly average prices BEFORE the month (leakage-free); the units are the
    # demand model's PREDICTIONS, so this scores exactly what the service serves.
    revenue_preds = demand_preds * test_df['avg_price_roll_mean_3'].values
    revenue_log, revenue_confidence = _per_category_confidence(test_df, revenue_preds, 'revenue')

    # ---------------- Promotion gate + backup ----------------
    if args.enforce_gate and not _passes_regression_gate(demand_confidence, revenue_confidence):
        print("\nGATE FAILED: new models are worse than the deployed ones. Nothing was overwritten.")
        raise SystemExit(2)
    backup_dir = _backup_artifacts()
    if backup_dir:
        print(f"Previous artifacts backed up to {backup_dir.name}/")

    # ---------------- Persist artifacts ----------------
    demand_model.save_model(str(DEMAND_MODEL_PATH))
    demand_confidence.to_csv(CATEGORY_CONFIDENCE_PATH)
    revenue_confidence.to_csv(REVENUE_CONFIDENCE_PATH)

    demand_log_out = demand_log.copy()
    demand_log_out['Month'] = demand_log_out['Month'].astype(str)
    demand_log_out.to_csv(MONTHLY_PREDICTIONS_PATH, index=False)

    revenue_log_out = revenue_log.copy()
    revenue_log_out['Month'] = revenue_log_out['Month'].astype(str)
    revenue_log_out.to_csv(REVENUE_PREDICTIONS_PATH, index=False)

    print(f"\nSaved: {DEMAND_MODEL_PATH.name}, "
          f"{CATEGORY_CONFIDENCE_PATH.name}, {REVENUE_CONFIDENCE_PATH.name}, "
          f"{MONTHLY_PREDICTIONS_PATH.name}, {REVENUE_PREDICTIONS_PATH.name}, {ACCURACY_PATH.name}")

    # ---------------- Accuracy metrics (MAE, RMSE, WAPE, NRMSE, bias, total-level error) ----------------
    accuracy = {'generated_at': datetime.now().isoformat(timespec='seconds'),
                'split': {'train_months': [str(m) for m in train_months], 'test_months': [str(m) for m in test_months]},
                'demand': {**am.summarize(demand_log), 'baselines': _baselines(test_df, 'units_sold', 'lag_1', 'roll_mean_6')},
                'revenue': {**am.summarize(revenue_log), 'method': 'predicted units x mean of the last 3 monthly average prices', 'baselines': _baselines(test_df, 'revenue', 'revenue_lag_1', 'revenue_roll_mean_6')}}
    ACCURACY_PATH.write_text(json.dumps(accuracy, indent=2))
    _accuracy_report('demand', accuracy['demand'], 'units')
    _accuracy_report('revenue', accuracy['revenue'], 'PHP')

    # ---------------- Pre-deployment gate report ----------------
    _gate_report(demand_confidence, 'demand', args.gate_threshold)
    _gate_report(revenue_confidence, 'revenue', args.gate_threshold)

    print("\nNOTE: the per-category report above is informational. Only the overall regression gate "
          "(--enforce-gate, on by default) blocks an overwrite. If a category fails its threshold, "
          "route it to flag_high_variance_categories() review.")
    print("\nReminder: this gate is a pre-deployment check on held-out history. It is "
          "NOT the long-run scoreboard -- keep using compare_actual_vs_forecast() every "
          "month once real actuals come in, independent of when this script last ran.")


if __name__ == '__main__':
    main()
