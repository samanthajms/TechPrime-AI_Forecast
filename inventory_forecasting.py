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
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

BASE_DIR = Path(__file__).parent

DEMAND_MODEL_PATH = BASE_DIR / 'xgb_monthly_demand_model.json'
REVENUE_MODEL_PATH = BASE_DIR / 'xgb_monthly_revenue_model.json'
CATEGORY_MAP_PATH = BASE_DIR / 'category_code_map.json'
CATEGORY_CONFIDENCE_PATH = BASE_DIR / 'category_confidence.csv'
REVENUE_CONFIDENCE_PATH = BASE_DIR / 'revenue_category_confidence.csv'
PRODUCT_SHARE_PATH = BASE_DIR / 'product_category_share.csv'
TRANSACTIONS_PATH = '/mnt/user-data/outputs/For-sample.xlsx'  # swap for a DB query in production

DEMAND_FEATURES = ['lag_1', 'lag_2', 'lag_3', 'lag_12', 'roll_mean_3', 'roll_mean_6',
                    'roll_std_3', 'month_num', 'quarter', 'category_code']
REVENUE_FEATURES = ['revenue_lag_1', 'revenue_lag_12', 'revenue_roll_mean_3', 'revenue_roll_mean_6',
                     'avg_price_lag_1', 'avg_price_roll_mean_3', 'price_trend',
                     'month_num', 'quarter', 'category_code']

# -----------------------------------------------------------------
# Load models + supporting artifacts once at import time
# -----------------------------------------------------------------
_demand_model = xgb.XGBRegressor()
_demand_model.load_model(str(DEMAND_MODEL_PATH))

_revenue_model = xgb.XGBRegressor()
_revenue_model.load_model(str(REVENUE_MODEL_PATH))

with open(CATEGORY_MAP_PATH) as f:
    CATEGORY_CODE_MAP = json.load(f)

_cat_confidence = pd.read_csv(CATEGORY_CONFIDENCE_PATH).set_index('Category')
_revenue_confidence = pd.read_csv(REVENUE_CONFIDENCE_PATH).set_index('Category')
_product_share = pd.read_csv(PRODUCT_SHARE_PATH)


# -----------------------------------------------------------------
# Internal: data access layer -- REPLACE THIS FOR PRODUCTION
# -----------------------------------------------------------------
def _load_transactions():
    """
    Returns cleaned, demand-only transaction rows.
    SWAP THIS to a real query, e.g.:

        return db.query('''
            SELECT order_date AS "POS Order Date", category AS "Category",
                   product_name AS "Product Name", quantity AS "Quantity",
                   unit_price AS "Unit Price", total AS "Total Sales VAT Inclusive"
            FROM sales_transactions
            WHERE quantity > 0 AND category != 'Customer Advances'
        ''')
    """
    df = pd.read_excel(TRANSACTIONS_PATH)
    df = df[df['Category'] != 'Customer Advances']
    df = df[df['Quantity'] > 0]
    df['POS Order Date'] = pd.to_datetime(df['POS Order Date'])
    cols = ['POS Order Date', 'Category', 'Product Name', 'Quantity',
            'Unit Price', 'Total Sales VAT Inclusive']
    return df[cols]


def _resolve_category(category=None, product=None):
    """
    Category/product filter resolution: if a product is given, look up its
    parent category (a product forecast is always derived from its category's
    model -- see module docstring on top-down allocation).
    Returns (category, product_share) where product_share is None if no
    product filter was given.
    """
    if category is None and product is None:
        raise ValueError("Provide either `category` or `product`.")

    if product is not None:
        match = _product_share[_product_share['Product Name'] == product]
        if match.empty:
            raise ValueError(f"Unknown product '{product}'.")
        resolved_category = match['Category'].iloc[0]
        if category is not None and category != resolved_category:
            raise ValueError(f"Product '{product}' belongs to category "
                              f"'{resolved_category}', not '{category}'.")
        share = float(match['share'].iloc[0])
        return resolved_category, share

    return category, None


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
    cat, product_share = _resolve_category(category, product)
    if cat not in CATEGORY_CODE_MAP:
        raise ValueError(f"Unknown category '{cat}'. Valid: {list(CATEGORY_CODE_MAP)}")
    cat_code = CATEGORY_CODE_MAP[cat]

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

        if product_share is not None:
            row = {
                'month': row['month'],
                'predicted_units': round(units_pred * product_share, 1),
                'units_confidence_low': round(max((units_pred - d_band) * product_share, 0), 1),
                'units_confidence_high': round((units_pred + d_band) * product_share, 1),
                'predicted_revenue': round(revenue_pred * product_share, 2),
                'revenue_confidence_low': round(max((revenue_pred - r_band) * product_share, 0), 2),
                'revenue_confidence_high': round((revenue_pred + r_band) * product_share, 2),
                'note': f"derived from '{cat}' category forecast x {product_share:.1%} historical product share",
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
    Return REAL historical monthly sales (units + revenue) for a category or
    a specific product, between two dates. Read-only -- does NOT touch either
    model. Safe to expose directly as a dashboard category/product + date
    filter (see the "is historical-range filtering advisable" discussion --
    this is the safe version of that idea: it filters DISPLAY, not training).

    category / product: filter by one or the other (product implies its category)
    start_date, end_date: 'YYYY-MM-DD' strings or pd.Timestamp; default to
                           full available range if omitted
    Returns: DataFrame with columns ['month', 'actual_units', 'actual_revenue']
    """
    cat, product_share = _resolve_category(category, product)
    if cat not in CATEGORY_CODE_MAP:
        raise ValueError(f"Unknown category '{cat}'. Valid: {list(CATEGORY_CODE_MAP)}")

    series = _monthly_series(cat)
    if start_date is not None:
        start_p = pd.Period(pd.Timestamp(start_date), freq='M')
        series = series[series['Month'] >= start_p]
    if end_date is not None:
        end_p = pd.Period(pd.Timestamp(end_date), freq='M')
        series = series[series['Month'] <= end_p]

    out = series[['Month', 'units_sold', 'revenue']].rename(
        columns={'Month': 'month', 'units_sold': 'actual_units', 'revenue': 'actual_revenue'})

    if product_share is not None:
        # NOTE: for a specific product, exact historical actuals ARE available
        # (unlike the forecast, which must be allocated) -- use real transaction
        # data directly rather than the category-share approximation
        df = _load_transactions()
        df['Month'] = df['POS Order Date'].dt.to_period('M')
        prod_actual = df[df['Product Name'] == product].groupby('Month').agg(
            actual_units=('Quantity', 'sum'), actual_revenue=('Total Sales VAT Inclusive', 'sum')
        ).reset_index()
        out = out[['month']].merge(prod_actual, left_on='month', right_on='Month', how='left').drop(columns='Month')
        out[['actual_units', 'actual_revenue']] = out[['actual_units', 'actual_revenue']].fillna(0)

    out['month'] = out['month'].astype(str)
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
