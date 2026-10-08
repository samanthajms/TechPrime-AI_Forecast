"""
inventory_forecasting.py
=========================
Drop-in module for integrating the monthly XGBoost demand + revenue
forecasting models into an inventory system.

TWO SEPARATE MODELS (not one derived from the other):
  - Demand model  -> predicts units_sold per category/month
  - Revenue model -> predicts revenue (PHP) per category/month directly,
    using its own price-based features (avg price, price trend), so it can
    learn revenue-specific patterns that units alone can't capture. See the
    "TechPrime AI alignment" conversation for why these are kept separate
    rather than derived (units x price) -- some categories (e.g. Memory,
    Solid State Drive) have real price volatility that a derived number
    would miss.

PRODUCT-LEVEL FORECASTING (top-down allocation, not a separate model):
  438 products is too sparse per-SKU to train reliable individual models --
  many products sell in only a handful of months. Instead, product forecasts
  are the category forecast allocated by that product's historical share of
  category volume. This is standard hierarchical/top-down forecasting
  practice and is far more stable than 438 noisy per-product models.

WHAT TO SWAP FOR PRODUCTION:
  - `_load_transactions()` currently reads the sample .xlsx file. Replace its
    body with a query against your real transactions table/database. It must
    return columns: ['POS Order Date', 'Category', 'Product Name', 'Quantity',
    'Unit Price', 'Total Sales VAT Inclusive']. Everything else works
    unchanged once that function returns real data in that shape.
  - Retrain both models monthly using rebuild_models_v2.py's logic, then keep
    CATEGORY_CODE_MAP and product_category_share.csv in sync.

FUNCTIONS PROVIDED:
  - get_forecast(category=None, product=None, horizon=1)
  - get_historical_actuals(category=None, product=None, start_date=None, end_date=None)
  - get_forecast_confidence(category, target='demand')
  - compare_actual_vs_forecast(month, category=None, target='demand')
  - flag_high_variance_categories(threshold=0.5, target='demand')
"""

import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import functools
import category_alignment as ca
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).parent

DEMAND_MODEL_PATH = BASE_DIR / 'xgb_monthly_demand_model.json'
REVENUE_MODEL_PATH = BASE_DIR / 'xgb_monthly_revenue_model.json'
CATEGORY_MAP_PATH = BASE_DIR / 'category_code_map.json'
CATEGORY_CONFIDENCE_PATH = BASE_DIR / 'category_confidence.csv'
REVENUE_CONFIDENCE_PATH = BASE_DIR / 'revenue_category_confidence.csv'
PRODUCT_SHARE_PATH = BASE_DIR / 'product_category_share.csv'  # legacy, no longer used
HISTORY_PATH = Path(os.environ.get('HISTORY_PATH', BASE_DIR / 'preprocessed_monthly_product_features.csv'))

DEMAND_FEATURES = ['lag_1', 'lag_2', 'lag_3', 'lag_12', 'roll_mean_3', 'roll_mean_6',
                    'roll_std_3', 'month_num', 'quarter', 'category_code']
REVENUE_FEATURES = ['revenue_lag_1', 'revenue_lag_12', 'revenue_roll_mean_3', 'revenue_roll_mean_6',
                     'avg_price_lag_1', 'avg_price_roll_mean_3', 'price_trend',
                     'month_num', 'quarter', 'category_code']

# -----------------------------------------------------------------
# Load models + supporting artifacts once at import time
# -----------------------------------------------------------------
MAX_HORIZON = 3   # product requirement: forecasts are limited to 1-3 months ahead


def check_horizon(horizon):
    """Single source of truth for the 1..MAX_HORIZON month window (the service and every public forecast entry point call it)."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be a whole number of months from 1 to {MAX_HORIZON}")
    return horizon


_demand_model = _revenue_model = None
CATEGORY_CODE_MAP = {}
_cat_confidence = _revenue_confidence = None


def reload_models():
    """(Re)load the trained models, category codes and confidence tables from disk. Called at import and again after a
    monthly retrain so the running process picks up the new artifacts without a restart. Everything is loaded first and
    swapped in together, so a half-written file can never leave the module in a mixed state."""
    global _demand_model, _revenue_model, CATEGORY_CODE_MAP, _cat_confidence, _revenue_confidence
    demand = xgb.XGBRegressor()
    demand.load_model(str(DEMAND_MODEL_PATH))
    revenue = xgb.XGBRegressor()
    revenue.load_model(str(REVENUE_MODEL_PATH))
    with open(CATEGORY_MAP_PATH) as f:
        code_map = json.load(f)
    cat_conf = pd.read_csv(CATEGORY_CONFIDENCE_PATH).set_index('Category')
    rev_conf = pd.read_csv(REVENUE_CONFIDENCE_PATH).set_index('Category')
    _demand_model, _revenue_model, CATEGORY_CODE_MAP = demand, revenue, code_map
    _cat_confidence, _revenue_confidence = cat_conf, rev_conf


reload_models()


# -----------------------------------------------------------------
# Internal: data access layer -- REPLACE THIS FOR PRODUCTION
# -----------------------------------------------------------------
def history_cutoff():
    """Optional HISTORY_CUTOFF=YYYY-MM. When set, every forecast treats that month as the latest actual month, so the
    forecast window can be moved back to demonstrate output against months whose real sales are already known.
    Needs reload_history() (or a restart) to take effect on a running process."""
    raw = os.environ.get('HISTORY_CUTOFF', '').strip()
    if not raw:
        return None
    try:
        return str(pd.Period(raw, freq='M'))
    except Exception:  # noqa: BLE001
        raise ValueError("HISTORY_CUTOFF must look like YYYY-MM (e.g. 2026-05).") from None


@functools.lru_cache(maxsize=1)
def _history():
    """Monthly PRODUCT history (with product names). Written by build_product_history.py (legacy POS)
    or by the live-DB rebuild job -- same columns either way."""
    h = pd.read_csv(HISTORY_PATH, dtype={'MSKU': str})
    cutoff = history_cutoff()
    if cutoff:                                                     # demo / back-test: forecast as if `cutoff` were the latest month
        h = h[h['Month'] <= cutoff]
        if h.empty:
            raise ValueError(f"HISTORY_CUTOFF {cutoff} is before the first month in the history file.")
    h['Category'] = h['Category'].str.strip().str.upper()          # MODEL category (what the models are trained on)
    if 'Shop Category' not in h.columns:                           # older file without shop columns -> derive
        cls = {m: ca.classify_product(m, n, c) for m, n, c in
               h.drop_duplicates('MSKU')[['MSKU', 'Product Name', 'Category']].itertuples(index=False)}
        h['Shop Group'] = h['MSKU'].map(lambda m: cls[m][1])
        h['Shop Category'] = h['MSKU'].map(lambda m: cls[m][2])
    return h


def reload_history():
    """Reload the models and history after a retrain / data refresh and drop every cache that depends on them."""
    reload_models()
    _history.cache_clear(); _product_shares.cache_clear(); _product_forecast.cache_clear()
    get_model_metrics.cache_clear(); ca.reload_map()


def _load_transactions():
    """Compatibility shim: exposes the monthly product history in the transaction-like shape
    the rest of this module expects (one row per product-month)."""
    h = _history()
    return pd.DataFrame({
        'POS Order Date': pd.to_datetime(h['Month'] + '-01'),
        'Category': h['Category'], 'Product Name': h['Product Name'], 'MSKU': h['MSKU'],
        'Quantity': h['units_sold'], 'Unit Price': h['avg_price'],
        'Total Sales VAT Inclusive': h['revenue'],
    })


@functools.lru_cache(maxsize=1)
def _product_shares(window=12, active_within=6):
    """Top-down allocation weights from the LAST `window` months, only for products that sold in the last
    `active_within` months. Units use unit share; revenue uses REVENUE share (a GPU and a cable have
    very different prices, so revenue must not be allocated by unit share)."""
    h = _history()
    months = sorted(h['Month'].unique())
    recent = h[h['Month'].isin(months[-window:])]
    active = set(h[h['Month'].isin(months[-active_within:])]['MSKU'])
    g = recent[recent['MSKU'].isin(active)].groupby(['Category', 'MSKU']).agg(
        units=('units_sold', 'sum'), revenue=('revenue', 'sum'),
        months_active=('Month', 'nunique')).reset_index()
    last = h.sort_values('Month').groupby('MSKU')[['Product Name', 'Shop Group', 'Shop Category']].last()
    g['Product Name'] = g['MSKU'].map(last['Product Name'])
    g['Shop Group'] = g['MSKU'].map(last['Shop Group'])
    g['Shop Category'] = g['MSKU'].map(last['Shop Category'])
    g['unit_share'] = g['units'] / g.groupby('Category')['units'].transform('sum')
    g['revenue_share'] = g['revenue'] / g.groupby('Category')['revenue'].transform('sum')
    g['last3_units'] = g['MSKU'].map(h[h['Month'].isin(months[-3:])].groupby('MSKU')['units_sold'].sum()).fillna(0)
    return g.fillna(0)


def _resolve_category(category=None, product=None):
    """Returns (category, product_row_or_None). `product` may be a product name or an MSKU."""
    if category is None and product is None:
        raise ValueError("Provide either `category` or `product`.")
    if product is not None:
        sh = _product_shares()
        match = sh[(sh['Product Name'] == product) | (sh['MSKU'] == str(product))]
        if match.empty:
            raise ValueError(f"Unknown or inactive product '{product}'.")
        row = match.iloc[0]
        if category is not None and str(category).upper() != row['Category']:
            raise ValueError(f"Product '{product}' belongs to '{row['Category']}', not '{category}'.")
        return row['Category'], row
    return str(category).upper(), None


def _monthly_series(category, target_col='units_sold'):
    """Aggregate transactions to a monthly series (units_sold or revenue) for one category."""
    df = _load_transactions()
    df['Month'] = df['POS Order Date'].dt.to_period('M')
    df = df[df['Category'] == category]

    monthly = df.groupby('Month').agg(
        units_sold=('Quantity', 'sum'),
        revenue=('Total Sales VAT Inclusive', 'sum'),
    ).reset_index()
    monthly['avg_price'] = np.where(monthly['units_sold'] > 0,
                                     monthly['revenue'] / monthly['units_sold'], np.nan)

    all_transactions = _load_transactions()
    all_months = pd.period_range(
        all_transactions['POS Order Date'].dt.to_period('M').min(),
        all_transactions['POS Order Date'].dt.to_period('M').max(), freq='M')
    monthly = monthly.set_index('Month').reindex(all_months).reset_index().rename(columns={'index': 'Month'})
    monthly['units_sold'] = monthly['units_sold'].fillna(0)
    monthly['revenue'] = monthly['revenue'].fillna(0)
    monthly['avg_price'] = monthly['avg_price'].ffill().bfill()
    return monthly.sort_values('Month').reset_index(drop=True)


def _build_demand_features(series_df):
    """Returns feature dict for the NEXT month's demand prediction."""
    s = series_df['units_sold'].values
    if len(s) < 3:
        raise ValueError("Need at least 3 months of history to build features.")

    def safe(k):
        return s[-k] if len(s) >= k else np.nan

    lag_1, lag_2, lag_3 = safe(1), safe(2), safe(3)
    lag_12 = safe(12) if len(s) >= 12 else lag_3
    roll_mean_3 = np.mean(s[-3:])
    roll_mean_6 = np.mean(s[-6:]) if len(s) >= 6 else np.mean(s)
    roll_std_3 = np.std(s[-3:], ddof=1) if len(s[-3:]) > 1 else 0.0
    next_month = series_df['Month'].iloc[-1] + 1
    feats = {'lag_1': lag_1, 'lag_2': lag_2, 'lag_3': lag_3, 'lag_12': lag_12,
             'roll_mean_3': roll_mean_3, 'roll_mean_6': roll_mean_6, 'roll_std_3': roll_std_3,
             'month_num': next_month.month, 'quarter': next_month.quarter}
    return feats, next_month


def _build_revenue_features(series_df):
    """Returns feature dict for the NEXT month's revenue prediction."""
    rev = series_df['revenue'].values
    price = series_df['avg_price'].values
    if len(rev) < 4:
        raise ValueError("Need at least 4 months of history to build revenue features.")

    def safe(arr, k):
        return arr[-k] if len(arr) >= k else np.nan

    revenue_lag_1 = safe(rev, 1)
    revenue_lag_12 = safe(rev, 12) if len(rev) >= 12 else revenue_lag_1
    revenue_roll_mean_3 = np.mean(rev[-3:])
    revenue_roll_mean_6 = np.mean(rev[-6:]) if len(rev) >= 6 else np.mean(rev)
    avg_price_lag_1 = safe(price, 1)
    avg_price_roll_mean_3 = np.mean(price[-3:])
    price_trend = ((safe(price, 1) - safe(price, 4)) / safe(price, 4)) if len(price) >= 4 and safe(price, 4) else 0.0

    next_month = series_df['Month'].iloc[-1] + 1
    feats = {'revenue_lag_1': revenue_lag_1, 'revenue_lag_12': revenue_lag_12,
             'revenue_roll_mean_3': revenue_roll_mean_3, 'revenue_roll_mean_6': revenue_roll_mean_6,
             'avg_price_lag_1': avg_price_lag_1, 'avg_price_roll_mean_3': avg_price_roll_mean_3,
             'price_trend': price_trend, 'month_num': next_month.month, 'quarter': next_month.quarter}
    return feats, next_month


# -----------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------
def get_forecast(category=None, product=None, horizon=1):
    """
    Forecast demand AND revenue for the next `horizon` months.

    Category/product filter:
      - get_forecast(category='MEMORY')             -> category-level forecast
      - get_forecast(product='Team Elite Plus 8gb...') -> product-level forecast,
        computed as the category forecast x that product's historical share
        of category volume (top-down allocation -- see module docstring for why
        products are NOT individually modeled).

    Both the demand model and the revenue model are run independently (NOT
    revenue = units x price) -- see module docstring.

    Returns: list of dicts, one per forecasted month:
        [{'month': 'YYYY-MM',
          'predicted_units': float, 'units_confidence_low': float, 'units_confidence_high': float,
          'predicted_revenue': float, 'revenue_confidence_low': float, 'revenue_confidence_high': float}, ...]
    """
    check_horizon(horizon)
    if product is None and category is not None:
        kind, val = ca.resolve_filter(category)
        if kind != 'model':                       # shop label or shop group -> sum of its products' forecasts
            return _aggregate_forecast(kind, val, horizon)
        category = val
    cat, prow = _resolve_category(category, product)
    if cat not in CATEGORY_CODE_MAP:
        raise ValueError(f"Unknown category '{cat}'. Valid: {list(CATEGORY_CODE_MAP)}")
    cat_code = CATEGORY_CODE_MAP[cat]
    product_share = None if prow is None else float(prow['unit_share'])
    rev_share = None if prow is None else float(prow['revenue_share'])

    demand_series = _monthly_series(cat)[['Month', 'units_sold']].copy()
    revenue_series = _monthly_series(cat)[['Month', 'revenue', 'avg_price']].copy()

    demand_rmse = float(_cat_confidence.loc[cat, 'rmse']) if cat in _cat_confidence.index else float(demand_series['units_sold'].std())
    revenue_rmse = float(_revenue_confidence.loc[cat, 'rmse']) if cat in _revenue_confidence.index else float(revenue_series['revenue'].std())

    results = []
    for step in range(1, horizon + 1):
        # --- demand model ---
        d_feats, next_month = _build_demand_features(demand_series)
        d_feats['category_code'] = cat_code
        Xd = pd.DataFrame([d_feats])[DEMAND_FEATURES]
        units_pred = float(np.clip(_demand_model.predict(Xd)[0], 0, None))
        d_band = demand_rmse * (1 + 0.3 * (step - 1))

        # --- revenue model ---
        r_feats, _ = _build_revenue_features(revenue_series)
        r_feats['category_code'] = cat_code
        Xr = pd.DataFrame([r_feats])[REVENUE_FEATURES]
        revenue_pred = float(np.clip(_revenue_model.predict(Xr)[0], 0, None))
        r_band = revenue_rmse * (1 + 0.3 * (step - 1))

        row = {
            'month': str(next_month),
            'predicted_units': round(units_pred, 1),
            'units_confidence_low': round(max(units_pred - d_band, 0), 1),
            'units_confidence_high': round(units_pred + d_band, 1),
            'predicted_revenue': round(revenue_pred, 2),
            'revenue_confidence_low': round(max(revenue_pred - r_band, 0), 2),
            'revenue_confidence_high': round(revenue_pred + r_band, 2),
        }

        if prow is not None:
            row = {
                'month': row['month'],
                'predicted_units': round(units_pred * product_share, 1),
                'units_confidence_low': round(max((units_pred - d_band) * product_share, 0), 1),
                'units_confidence_high': round((units_pred + d_band) * product_share, 1),
                'predicted_revenue': round(revenue_pred * rev_share, 2),
                'revenue_confidence_low': round(max((revenue_pred - r_band) * rev_share, 0), 2),
                'revenue_confidence_high': round((revenue_pred + r_band) * rev_share, 2),
                'note': f"'{cat}' forecast x {product_share:.1%} unit share / {rev_share:.1%} revenue share (last 12 months)",
            }

        results.append(row)

        # feed predictions back in as if observed, for the next recursive step
        demand_series = pd.concat([demand_series, pd.DataFrame(
            [{'Month': next_month, 'units_sold': units_pred}])], ignore_index=True)
        implied_price = revenue_pred / units_pred if units_pred > 0 else revenue_series['avg_price'].iloc[-1]
        revenue_series = pd.concat([revenue_series, pd.DataFrame(
            [{'Month': next_month, 'revenue': revenue_pred, 'avg_price': implied_price}])], ignore_index=True)

    return results


def get_historical_actuals(category=None, product=None, start_date=None, end_date=None):
    """
    REAL historical monthly sales (units + revenue) for a shop label / shop group / model category, or one product
    (name or MSKU). Read-only -- never touches the models. Filters DISPLAY, not training.
    Returns: DataFrame ['month', 'actual_units', 'actual_revenue']
    """
    h = _history()
    if product is not None:
        h = h[(h['Product Name'] == product) | (h['MSKU'] == str(product))]
        if h.empty:
            raise ValueError(f"Unknown product '{product}'.")
    elif category is not None:
        kind, val = ca.resolve_filter(category)
        col = {'label': 'Shop Category', 'group': 'Shop Group', 'model': 'Category'}[kind]
        h = h[h[col] == val]
    else:
        raise ValueError("Provide either `category` or `product`.")
    out = h.groupby('Month').agg(actual_units=('units_sold', 'sum'), actual_revenue=('revenue', 'sum')).reset_index()
    out = out.rename(columns={'Month': 'month'})
    if start_date is not None:
        out = out[out['month'] >= str(pd.Period(pd.Timestamp(start_date), freq='M'))]
    if end_date is not None:
        out = out[out['month'] <= str(pd.Period(pd.Timestamp(end_date), freq='M'))]
    return out.reset_index(drop=True)


def get_forecast_confidence(category, target='demand'):
    """
    Return a category's historical forecast error (from the held-out 20%
    validation), so a caller can judge how much to trust its forecasts.

    target: 'demand' or 'revenue' -- the two models have independent
            reliability, since revenue depends on both volume AND price
    Returns: dict with mae, rmse, avg_actual, and a reliability label.
    """
    conf_table = _cat_confidence if target == 'demand' else _revenue_confidence
    if category not in conf_table.index:
        return {'category': category, 'target': target, 'mae': None, 'rmse': None,
                'avg_actual': None, 'reliability': 'unknown (no validation history)'}

    row = conf_table.loc[category]
    relative_error = row['rmse'] / row['avg_actual'] if row['avg_actual'] > 0 else np.inf
    if relative_error < 0.15:
        reliability = 'high'
    elif relative_error < 0.35:
        reliability = 'medium'
    else:
        reliability = 'low'

    return {
        'category': category, 'target': target,
        'mae': float(row['mae']), 'rmse': float(row['rmse']), 'avg_actual': float(row['avg_actual']),
        'relative_error_pct': round(relative_error * 100, 1) if np.isfinite(relative_error) else None,
        'reliability': reliability,
    }


def compare_actual_vs_forecast(month, category=None, target='demand'):
    """
    Once a forecasted month has become actual, compare predicted vs. real.
    Use this to build an ongoing accuracy log.

    month: 'YYYY-MM'
    category: optional, restrict to one category
    target: 'demand' or 'revenue'
    Returns: DataFrame with columns [Category, actual, predicted, error, pct_error]

    NOTE: reads from the 80/20 chronological test-set validation log as a
    demo. In production,
    log every forecast you actually serve to a `forecast_log` table instead,
    so you can compare it against the real outcome once the month closes.
    """
    fname = 'monthly_predictions.csv' if target == 'demand' else 'revenue_predictions.csv'
    preds = pd.read_csv(BASE_DIR / fname)
    preds = preds[preds['Month'] == month]
    if category is not None:
        preds = preds[preds['Category'] == category]
    if preds.empty:
        raise ValueError(f"No logged forecast found for month={month}, category={category}, target={target}")

    preds = preds.copy()
    preds['error'] = preds['actual'] - preds['predicted']
    preds['pct_error'] = np.where(
        preds['actual'] > 0, (preds['error'].abs() / preds['actual'] * 100).round(1), np.nan)
    return preds[['Category', 'actual', 'predicted', 'error', 'pct_error']].reset_index(drop=True)


def flag_high_variance_categories(threshold=0.35, target='demand'):
    """
    Return categories whose forecast error (RMSE relative to average) exceeds
    `threshold` -- categories to manually review rather than auto-trust.

    target: 'demand' or 'revenue' -- check separately, since a category can be
            reliable for units but unreliable for revenue (price volatility)
            or vice versa
    Returns: DataFrame sorted by relative error, worst first
    """
    conf_table = _cat_confidence if target == 'demand' else _revenue_confidence
    df = conf_table.copy()
    df['relative_error'] = df['rmse'] / df['avg_actual'].replace(0, np.nan)
    flagged = df[df['relative_error'] > threshold].sort_values('relative_error', ascending=False)
    flagged = flagged.reset_index()
    flagged['relative_error_pct'] = (flagged['relative_error'] * 100).round(1)
    return flagged[['Category', 'mae', 'rmse', 'avg_actual', 'relative_error_pct']]


# -----------------------------------------------------------------
# Screen-level helpers (Product Demand Forecast / Sales-Revenue Forecast)
# All category output uses the SHOP vocabulary (Client + Custodian labels, Client groups) -- see category_alignment.py
# -----------------------------------------------------------------
SKIP_CATEGORIES = {'ALL'}
_BAND_COLS = ['u', 'u_lo', 'u_hi', 'r', 'r_lo', 'r_hi']        # mid, and distance mid->low / mid->high


def _model_categories():
    """Model categories that have history AND a trained code."""
    return [c for c in _history()['Category'].unique() if c in CATEGORY_CODE_MAP and c.upper() not in SKIP_CATEGORIES]


@functools.lru_cache(maxsize=8)
def _product_forecast(horizon):
    """One row per product x forecast month = model category forecast x the product's 12-month share
    (unit share for units, revenue share for revenue). Band half-widths are kept so rows can be re-aggregated to any
    level: bands add linearly inside one model category and in quadrature across different ones."""
    shares = _product_shares()
    rows = []
    for mc in _model_categories():
        prods = shares[shares['Category'] == mc].to_dict('records')
        if not prods:
            continue
        for r in get_forecast(category=f'model:{mc}', horizon=horizon):
            for p in prods:
                us, rs = p['unit_share'], p['revenue_share']
                rows.append({
                    'month': r['month'], 'msku': p['MSKU'], 'product_name': p['Product Name'],
                    'shop_group': p['Shop Group'], 'category': p['Shop Category'], 'model_category': mc,
                    'u': r['predicted_units'] * us,
                    'u_lo': (r['predicted_units'] - r['units_confidence_low']) * us,
                    'u_hi': (r['units_confidence_high'] - r['predicted_units']) * us,
                    'r': r['predicted_revenue'] * rs,
                    'r_lo': (r['predicted_revenue'] - r['revenue_confidence_low']) * rs,
                    'r_hi': (r['revenue_confidence_high'] - r['predicted_revenue']) * rs,
                })
    return pd.DataFrame(rows)


def _filter(df, kind, val):
    col = {'label': 'category', 'group': 'shop_group', 'model': 'model_category'}[kind]
    return df[df[col] == val]


def _rollup(df, keys):
    """Sum product rows to `keys` (+ month). Linear within a model category, quadrature across model categories."""
    lin = df.groupby(['month', 'model_category'] + keys)[_BAND_COLS].sum().reset_index()
    for c in ['u_lo', 'u_hi', 'r_lo', 'r_hi']:
        lin[c] = lin[c] ** 2
    out = lin.groupby(['month'] + keys)[_BAND_COLS].sum().reset_index()
    for c in ['u_lo', 'u_hi', 'r_lo', 'r_hi']:
        out[c] = out[c] ** 0.5
    return out


def _fmt(row):
    return {'predicted_units': round(row['u'], 1),
            'units_confidence_low': round(max(row['u'] - row['u_lo'], 0), 1),
            'units_confidence_high': round(row['u'] + row['u_hi'], 1),
            'predicted_revenue': round(row['r'], 2),
            'revenue_confidence_low': round(max(row['r'] - row['r_lo'], 0), 2),
            'revenue_confidence_high': round(row['r'] + row['r_hi'], 2)}


def _aggregate_forecast(kind, val, horizon):
    """get_forecast() for a shop label or shop group: same shape as the model-category result."""
    df = _filter(_product_forecast(horizon), kind, val)
    if df.empty:
        raise ValueError(f"No forecastable products with enough history in '{val}'.")
    agg = _rollup(df, [])
    return [{'month': r['month'], **_fmt(r)} for _, r in agg.sort_values('month').iterrows()]


def get_categories():
    """The canonical category list every role should use (feeds the Retail filters)."""
    h = _history()
    active = set(h['Shop Category'])
    tax = ca.taxonomy()
    return {
        'source': 'Client/Custodian taxonomy (includes/client_shop_taxonomy.php)',
        'groups': [{'group': g, 'categories': [{'category': lab, 'has_forecast_history': lab in active}
                                               for lab in labs]} for g, labs in tax['groups'].items()],
        'model_categories_internal': sorted(_model_categories()),
    }


def get_revenue_forecast(horizon=3, category=None):
    """Sales/Revenue screen. `category` may be a shop label, shop group, or 'model:X'. Returns the total for the
    selection, a breakdown by shop category and by shop group, and the monthly history for the chart."""
    check_horizon(horizon)
    df = _product_forecast(horizon)
    h = _history()
    if category:
        kind, val = ca.resolve_filter(category)
        df = _filter(df, kind, val)
        h = _filter(h.rename(columns={'Shop Category': 'category', 'Shop Group': 'shop_group', 'Category': 'model_category'}),
                    kind, val)
    else:
        h = h.rename(columns={'Shop Category': 'category', 'Shop Group': 'shop_group', 'Category': 'model_category'})
    if df.empty:                                  # valid category, but nothing to forecast: answer 200 with an empty state
        hist = h.groupby('Month').agg(actual_revenue=('revenue', 'sum'), actual_units=('units_sold', 'sum')).reset_index() \
            .rename(columns={'Month': 'month'})
        return {'filter': {'kind': kind, 'value': val} if category else None, 'total': [], 'by_category': [],
                'by_group': [], 'history': hist.round(2).to_dict('records'), 'history_by_category': [],
                'message': "No forecast is available for this category yet (it has no trained model category, or no product in it sold in the last 6 months)."}
    total = [{'month': r['month'], **_fmt(r)} for _, r in _rollup(df, []).sort_values('month').iterrows()]
    by_cat = [{'category': r['category'], 'group': ca.group_of(r['category']), 'month': r['month'], **_fmt(r)}
              for _, r in _rollup(df, ['category']).sort_values(['category', 'month']).iterrows()]
    by_grp = [{'group': r['shop_group'], 'month': r['month'], **_fmt(r)}
              for _, r in _rollup(df, ['shop_group']).sort_values(['shop_group', 'month']).iterrows()]
    hist = h.groupby('Month').agg(actual_revenue=('revenue', 'sum'), actual_units=('units_sold', 'sum')).reset_index()
    hist = hist.rename(columns={'Month': 'month'})
    hist_cat = h.groupby(['Month', 'category']).agg(actual_revenue=('revenue', 'sum'), actual_units=('units_sold', 'sum')) \
        .reset_index().rename(columns={'Month': 'month'})
    return {'filter': {'kind': kind, 'value': val} if category else None, 'total': total, 'by_category': by_cat,
            'by_group': by_grp, 'history': hist.round(2).to_dict('records'),
            'history_by_category': hist_cat.round(2).to_dict('records')}


def get_product_demand(horizon=3, category=None, top=50):
    """Product Demand screen: per-product forecast with product NAME, MSKU, shop category/group, last-3-month units,
    forecast units + range, forecast revenue, trend and a data-quality / reliability flag."""
    check_horizon(horizon)
    df = _product_forecast(horizon)
    if category:
        kind, val = ca.resolve_filter(category)
        df = _filter(df, kind, val)
    sh = _product_shares().set_index('MSKU')
    g = df.groupby('msku').agg(product_name=('product_name', 'first'), category=('category', 'first'),
                               group=('shop_group', 'first'), model_category=('model_category', 'first'),
                               u=('u', 'sum'), u_lo=('u_lo', 'sum'), u_hi=('u_hi', 'sum'), r=('r', 'sum')).reset_index()
    rel = {mc: get_forecast_confidence(mc, 'demand')['reliability'] for mc in g['model_category'].unique()}
    rows = []
    for p in g.itertuples(index=False):
        last3 = float(sh.loc[p.msku, 'last3_units']); active = int(sh.loc[p.msku, 'months_active'])
        rows.append({
            'msku': p.msku, 'product_name': p.product_name, 'category': p.category, 'group': p.group,
            'model_category': p.model_category,
            'last_3m_units': int(last3), 'forecast_units': round(p.u, 1),
            'forecast_units_low': round(max(p.u - p.u_lo, 0), 1), 'forecast_units_high': round(p.u + p.u_hi, 1),
            'forecast_revenue': round(p.r, 2),
            'trend_pct': round((p.u / horizon * 3 / last3 - 1) * 100, 1) if last3 > 0 else None,
            'months_active_12m': active,
            'confidence': 'low (thin history)' if active < 6 else 'category-based',
            'model_reliability': rel.get(p.model_category, 'unknown'),
        })
    rows.sort(key=lambda r: r['forecast_units'], reverse=True)
    out = {'horizon_months': horizon, 'filter': {'kind': kind, 'value': val} if category else None,
            'products': rows[:int(top)], 'total_products': len(rows)}
    if not rows:
        out['message'] = "No forecast is available for this category yet (it has no trained model category, or no product in it sold in the last 6 months)."
    return out


@functools.lru_cache(maxsize=1)
def get_model_metrics():
    """Held-out accuracy vs naive baselines, from the prediction logs written by rebuild_models_v2.py."""
    feats = pd.read_csv(BASE_DIR / 'preprocessed_monthly_features.csv')
    skip = {'All', 'CUSTOMIZATION'}
    out = {}
    for target, fname, lag in [('demand', 'monthly_predictions.csv', 'lag_1'),
                               ('revenue', 'revenue_predictions.csv', 'revenue_lag_1')]:
        p = pd.read_csv(BASE_DIR / fname)
        m = p.merge(feats[['Month', 'Category', lag]], on=['Month', 'Category'])
        m = m[~m['Category'].isin(skip)]
        a = m['actual'].abs().sum()
        err = m['actual'] - m['predicted']
        out[target] = {
            'test_months': sorted(m['Month'].unique().tolist()),
            'wape_pct': round(float(err.abs().sum() / a * 100), 1),
            'naive_last_month_wape_pct': round(float((m['actual'] - m[lag]).abs().sum() / a * 100), 1),
            'mae': round(float(err.abs().mean()), 2), 'rmse': round(float((err ** 2).mean() ** 0.5), 2),
            'bias_pct': round(float((m['predicted'].sum() / m['actual'].sum() - 1) * 100), 1),
        }
    h = _history()
    out['data'] = {'first_month': h['Month'].min(), 'last_month': h['Month'].max(), 'history_cutoff': history_cutoff(),
                   'products': int(h['MSKU'].nunique()), 'rows': int(len(h))}
    return out
