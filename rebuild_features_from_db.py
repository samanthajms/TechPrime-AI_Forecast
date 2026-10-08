"""
rebuild_features_from_db.py
============================
Pulls REAL sales transactions from the live TechPrime-AI database and turns
them into a fresh preprocessed_monthly_features.csv -- the same file
rebuild_models_v2.py trains on and inventory_forecasting.py reads for
inference-time features.

In plain terms: this is the "top-up" script. Run it, and it ADDS whatever has
actually sold in the live system since the last run, on top of the existing
file -- it does NOT throw away and rebuild from scratch.

THIS MATTERS BECAUSE: the 26 months already in preprocessed_monthly_features.csv
(2024-07 onward) came from an OLDER, separate POS system, used to originally
train/test the model, before TechPrime-AI's live orders table existed. The
live `orders` table is newer and does not cover that same history. So this
script treats the existing CSV as the historical foundation and only ADDS
months that come from real live orders and aren't already covered -- it never
deletes or overwrites the old POS-era rows.

WHAT IT DOES, STEP BY STEP:
  1. Reads the CURRENT preprocessed_monthly_features.csv and keeps every row
     already in it (the old POS history) untouched.
  2. Connects to the same Postgres (Supabase) database the PHP app uses,
     using the same .env file already sitting in this folder.
  3. Asks the database: "for every month and every product category, how
     many units sold and how much revenue came in?" -- this is the live data.
  4. For each (category, month) combination in the live data: if that exact
     month is already covered by the old file, the old file's row wins and
     the live row is skipped (so POS history is never overwritten). If it's
     a month the old file doesn't have, the live row is added.
  5. Fills in any remaining gaps (a category with a month of zero sales still
     needs a row of zeros -- the model expects an unbroken monthly timeline).
  6. Re-calculates the lag/rolling-average columns (lag_1, roll_mean_3, etc.)
     across the FULL combined timeline (old + newly added), so months near
     the boundary between old and new data still get correct trend values.
  7. Writes the combined result back to preprocessed_monthly_features.csv.
  8. Keeps category_code_map.json in sync: if the database has a category
     the model has never seen before, it's added with a new code instead of
     silently crashing.

WHEN TO RUN THIS:
  Monthly, after a calendar month closes -- e.g. run it on the 1st for the
  month that just ended. Then run rebuild_models_v2.py right after, to
  retrain the models on the refreshed data.

WHAT COUNTS AS A SALE (so a spam or cancelled order cannot skew the forecast):
  - online orders whose status is in COUNTED_ORDER_STATUSES (not cancelled / unpaid),
  - cashier POS sales with status 'completed' (walk-in sales; voided ones are ignored),
  - CLOSED calendar months only (Asia/Manila) -- a half-finished month is never fed to the model.

It ALSO tops up preprocessed_monthly_product_features.csv (the per-product history the running service reads for
inference), so both history files always end on the same month.

The connection is opened READ ONLY; give it a database user that can only SELECT orders, order_items, products,
pos_sales and pos_sale_items.

USAGE:
    python rebuild_features_from_db.py
    python rebuild_features_from_db.py --dry-run               # preview only, don't overwrite anything
    python rebuild_features_from_db.py --allow-new-categories  # let unseen product categories get a new model code
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / '.env')

FEATURES_PATH = BASE_DIR / 'preprocessed_monthly_features.csv'
PRODUCT_HISTORY_PATH = BASE_DIR / 'preprocessed_monthly_product_features.csv'
CATEGORY_MAP_PATH = BASE_DIR / 'category_code_map.json'

DB_CONFIG = {
    'host': os.environ.get('DB_HOST'),
    'port': os.environ.get('DB_PORT'),
    'dbname': os.environ.get('DB_NAME'),
    'user': os.environ.get('DB_USER'),
    'password': os.environ.get('DB_PASS'),
    'sslmode': 'require',
}

# Online-order statuses that count as real demand (see ias_order_display_status in the PHP app). Everything else --
# 'Pending', 'to_pay' (unpaid), 'cancelled', anything unknown -- is left out on purpose.
COUNTED_ORDER_STATUSES = ('to_ship', 'to_receive', 'delivered', 'completed')

# Calendar month in Asia/Manila. created_at columns are timestamp-without-tz holding UTC.
_ORDER_MONTH = "date_trunc('month', (o.created_at AT TIME ZONE 'UTC') AT TIME ZONE 'Asia/Manila')::date"
_POS_MONTH = "date_trunc('month', (s.created_at AT TIME ZONE 'UTC') AT TIME ZONE 'Asia/Manila')::date"
_THIS_MONTH_START = "date_trunc('month', now() AT TIME ZONE 'Asia/Manila')::date"

# Column order must match the existing CSV exactly -- rebuild_models_v2.py
# and inventory_forecasting.py both expect these names.
OUTPUT_COLUMNS = [
    'Month', 'Category', 'units_sold', 'revenue', 'avg_price',
    'lag_1', 'lag_2', 'lag_3', 'lag_12', 'roll_mean_3', 'roll_mean_6', 'roll_std_3',
    'avg_price_lag_1', 'avg_price_roll_mean_3', 'price_trend',
    'revenue_lag_1', 'revenue_lag_12', 'revenue_roll_mean_3', 'revenue_roll_mean_6',
    'month_num', 'quarter', 'category_code',
]


# -----------------------------------------------------------------
# Step 1a: load the EXISTING file -- this is the POS-era historical
# foundation that must never be overwritten or dropped.
# -----------------------------------------------------------------
def load_existing_base() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        return pd.DataFrame(columns=['category', 'month', 'units_sold', 'revenue', 'avg_price'])

    old = pd.read_csv(FEATURES_PATH)
    old = old[old['Category'] != 'All']  # the aggregate row is recomputed fresh, not carried over
    old = old.rename(columns={'Category': 'category', 'Month': 'month'})
    old['month'] = pd.PeriodIndex(old['month'], freq='M')
    return old[['category', 'month', 'units_sold', 'revenue', 'avg_price']].copy()


# -----------------------------------------------------------------
# Step 1b: pull real transactions, grouped by month + category
# -----------------------------------------------------------------
def _read_sql(query: str, params: tuple = ()) -> pd.DataFrame:
    with psycopg2.connect(**DB_CONFIG) as conn:
        conn.set_session(readonly=True)          # this job must never be able to write
        return pd.read_sql(query, conn, params=params)


def _sales_lines_sql(select: str, group_by: str) -> str:
    """Online order lines UNION ALL cashier POS lines, closed months only. `select`/`group_by` are fixed strings
    from this file (never user input)."""
    return f"""
        WITH lines AS (
            SELECT {_ORDER_MONTH} AS month, p.id AS product_id, p.sku AS msku, p.name AS product_name,
                   UPPER(TRIM(p.category)) AS category,
                   oi.quantity::float AS qty, (oi.price * oi.quantity)::float AS amount
            FROM order_items oi
            JOIN products p ON p.id = oi.product_id
            JOIN orders o ON o.id = oi.order_id
            WHERE LOWER(o.status) IN %(statuses)s
              AND p.category IS NOT NULL AND p.category <> ''
            UNION ALL
            SELECT {_POS_MONTH}, p.id, p.sku, p.name, UPPER(TRIM(p.category)),
                   si.quantity::float, si.line_total::float
            FROM pos_sale_items si
            JOIN pos_sales s ON s.id = si.sale_id
            JOIN products p ON p.id = si.product_id
            WHERE s.status = 'completed'
              AND p.category IS NOT NULL AND p.category <> ''
        )
        SELECT {select}
        FROM lines
        WHERE month < {_THIS_MONTH_START}
        GROUP BY {group_by}
        ORDER BY {group_by}
    """


def fetch_monthly_sales() -> pd.DataFrame:
    """One row per (month, category): total units and revenue from counted online orders + completed POS sales."""
    query = _sales_lines_sql("month, category, SUM(qty) AS units_sold, SUM(amount) AS revenue", "category, month")
    df = _read_sql(query, {'statuses': COUNTED_ORDER_STATUSES})
    if df.empty:
        raise RuntimeError("No closed-month sales rows returned -- check DB connection / that sales exist.")
    df['month'] = pd.to_datetime(df['month']).dt.to_period('M')
    df['avg_price'] = np.where(df['units_sold'] > 0, df['revenue'] / df['units_sold'], np.nan)
    return df


def fetch_monthly_product_sales() -> pd.DataFrame:
    """One row per (month, product) in the shape of preprocessed_monthly_product_features.csv."""
    query = _sales_lines_sql(
        "month, msku, MAX(product_name) AS product_name, category, SUM(qty) AS units_sold, SUM(amount) AS revenue",
        "month, msku, category")
    df = _read_sql(query, {'statuses': COUNTED_ORDER_STATUSES})
    df['month'] = pd.to_datetime(df['month']).dt.to_period('M')
    return df


def filter_known_categories(df: pd.DataFrame, col: str, allow_new: bool) -> pd.DataFrame:
    """The models only know the categories in category_code_map.json. A category they have never seen would get a
    brand-new code and almost no history, so by default it is reported and skipped instead of silently added."""
    known = set(json.load(open(CATEGORY_MAP_PATH))) if CATEGORY_MAP_PATH.exists() else set()
    if allow_new or not known:
        return df
    unknown = sorted(set(df[col].unique()) - known)
    if unknown:
        print(f"Skipped {len(unknown)} categor(y/ies) the model has no code for (use --allow-new-categories to keep): {unknown}")
    return df[df[col].isin(known)]


def merge_product_history(existing: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Append live product-months AFTER the last month already in the file. Older rows are never touched."""
    last = existing['Month'].max() if not existing.empty else ''
    live = live.copy()
    live['Month'] = live['month'].astype(str)
    live = live[live['Month'] > last]
    live = live[live['msku'].notna() & (live['msku'].astype(str).str.strip() != '')]
    rows = pd.DataFrame({
        'Month': live['Month'], 'MSKU': live['msku'].astype(str), 'Product Name': live['product_name'],
        'Category': live['category'], 'units_sold': live['units_sold'].round(1), 'revenue': live['revenue'].round(2),
        'returns_units': 0.0, 'returns_amount': 0.0, 'net_revenue': live['revenue'].round(2),
        'avg_price': (live['revenue'] / live['units_sold'].where(live['units_sold'] > 0)).round(2),
    })
    return pd.concat([existing, rows], ignore_index=True).sort_values(['Month', 'Category', 'MSKU'])


# -----------------------------------------------------------------
# Step 2: merge -- old POS-era rows always win; live rows are only
# added for (category, month) pairs the old file doesn't already have.
# -----------------------------------------------------------------
def merge_base(old: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    old = old.copy()
    old['source'] = 'pos_history'
    live = live.copy()
    live['source'] = 'live_orders'

    combined = pd.concat([old, live], ignore_index=True)
    before = len(combined)
    # old rows are listed first, so keep='first' means POS history always wins on a clash
    combined = combined.drop_duplicates(subset=['category', 'month'], keep='first')
    skipped = before - len(combined)

    added_live = combined[combined['source'] == 'live_orders']
    print(f"POS-history rows kept as-is: {len(old)}")
    print(f"Live months skipped because that month already exists in POS history: {skipped}")
    print(f"New months added from live orders: {len(added_live)}"
          + (f" ({added_live['month'].min()} to {added_live['month'].max()})" if not added_live.empty else ""))

    return combined.drop(columns='source')


# -----------------------------------------------------------------
# Step 3: fill gaps so every category has one row per month, even
# months where nothing sold (the model needs an unbroken timeline)
# -----------------------------------------------------------------
def build_full_grid(monthly: pd.DataFrame) -> pd.DataFrame:
    all_months = pd.period_range(monthly['month'].min(), monthly['month'].max(), freq='M')
    categories = sorted(monthly['category'].unique())

    grid = pd.MultiIndex.from_product([categories, all_months], names=['category', 'month']).to_frame(index=False)
    merged = grid.merge(monthly, on=['category', 'month'], how='left')
    merged['units_sold'] = merged['units_sold'].fillna(0)
    merged['revenue'] = merged['revenue'].fillna(0)

    merged['avg_price'] = np.where(merged['units_sold'] > 0, merged['revenue'] / merged['units_sold'], np.nan)
    # a month with zero sales still needs a plausible price for the lag features -- carry the nearest known price
    merged['avg_price'] = merged.groupby('category')['avg_price'].transform(lambda s: s.ffill().bfill())

    # add an "All" row per month too, matching the existing CSV's aggregate total category
    totals = merged.groupby('month', as_index=False).agg(units_sold=('units_sold', 'sum'), revenue=('revenue', 'sum'))
    totals['category'] = 'All'
    totals['avg_price'] = np.where(totals['units_sold'] > 0, totals['revenue'] / totals['units_sold'], np.nan)
    totals['avg_price'] = totals['avg_price'].ffill().bfill()

    return pd.concat([merged, totals[merged.columns]], ignore_index=True).sort_values(['category', 'month'])


# -----------------------------------------------------------------
# Step 3: compute the lag / rolling-average feature columns
# (same formulas as _build_demand_features / _build_revenue_features
# in inventory_forecasting.py, applied across the whole history instead
# of just "the next month")
# -----------------------------------------------------------------
def add_features(full: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for cat, g in full.groupby('category', sort=False):
        g = g.sort_values('month').reset_index(drop=True)
        units = g['units_sold'].values
        revenue = g['revenue'].values
        price = g['avg_price'].values

        for i in range(len(g)):
            def lag(arr, k):
                return arr[i - k] if i - k >= 0 else 0

            lag_1, lag_2, lag_3, lag_12 = lag(units, 1), lag(units, 2), lag(units, 3), lag(units, 12)
            window3 = units[max(0, i - 2):i + 1]
            window6 = units[max(0, i - 5):i + 1]
            roll_mean_3 = float(np.mean(window3))
            roll_mean_6 = float(np.mean(window6))
            roll_std_3 = float(np.std(window3, ddof=1)) if len(window3) > 1 else 0.0

            r_lag_1 = lag(revenue, 1)
            r_lag_12 = lag(revenue, 12) if i - 12 >= 0 else r_lag_1
            r_window3 = revenue[max(0, i - 2):i + 1]
            r_window6 = revenue[max(0, i - 5):i + 1]
            revenue_roll_mean_3 = float(np.mean(r_window3))
            revenue_roll_mean_6 = float(np.mean(r_window6))

            p_lag_1 = lag(price, 1) if i - 1 >= 0 else price[i]
            p_window3 = price[max(0, i - 2):i + 1]
            avg_price_roll_mean_3 = float(np.mean(p_window3))
            p_lag_4 = lag(price, 4) if i - 4 >= 0 else None
            price_trend = ((p_lag_1 - p_lag_4) / p_lag_4) if p_lag_4 else 0.0

            month_period = g['month'].iloc[i]
            out_rows.append({
                'Month': str(month_period),
                'Category': cat,
                'units_sold': int(units[i]),
                'revenue': round(float(revenue[i]), 2),
                'avg_price': round(float(price[i]), 2) if not np.isnan(price[i]) else 0.0,
                'lag_1': int(lag_1), 'lag_2': int(lag_2), 'lag_3': int(lag_3), 'lag_12': int(lag_12),
                'roll_mean_3': roll_mean_3, 'roll_mean_6': roll_mean_6, 'roll_std_3': roll_std_3,
                'avg_price_lag_1': round(float(p_lag_1), 2), 'avg_price_roll_mean_3': round(avg_price_roll_mean_3, 2),
                'price_trend': round(price_trend, 4),
                'revenue_lag_1': round(float(r_lag_1), 2), 'revenue_lag_12': round(float(r_lag_12), 2),
                'revenue_roll_mean_3': round(revenue_roll_mean_3, 2), 'revenue_roll_mean_6': round(revenue_roll_mean_6, 2),
                'month_num': month_period.month, 'quarter': month_period.quarter,
            })

    return pd.DataFrame(out_rows)


# -----------------------------------------------------------------
# Step 4: keep category_code_map.json in sync with whatever
# categories actually exist in the live database
# -----------------------------------------------------------------
def sync_category_codes(df: pd.DataFrame) -> dict:
    if CATEGORY_MAP_PATH.exists():
        with open(CATEGORY_MAP_PATH) as f:
            code_map = json.load(f)
    else:
        code_map = {}

    next_code = max(code_map.values(), default=-1) + 1
    new_categories = sorted(set(df['Category'].unique()) - set(code_map.keys()))
    for cat in new_categories:
        code_map[cat] = next_code
        next_code += 1

    if new_categories:
        print(f"New categories found in the database, added to {CATEGORY_MAP_PATH.name}: {new_categories}")
        with open(CATEGORY_MAP_PATH, 'w') as f:
            json.dump(code_map, f, indent=2)

    df['category_code'] = df['Category'].map(code_map)
    return code_map


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help="Preview the result, don't overwrite the CSV")
    parser.add_argument('--allow-new-categories', action='store_true',
                        help="Keep product categories the models have no code for (they get a new code).")
    args = parser.parse_args()

    print("Reading existing POS-era historical data (kept as-is, never overwritten)...")
    old_base = load_existing_base()

    print("Connecting to the database and pulling closed-month sales (counted orders + completed POS sales)...")
    live_base = filter_known_categories(fetch_monthly_sales(), 'category', args.allow_new_categories)

    print("Merging: POS history stays fixed, only genuinely new live months are added...")
    monthly = merge_base(old_base, live_base)

    print("Filling in months with zero sales so every category has an unbroken timeline...")
    full = build_full_grid(monthly)

    print("Calculating lag / rolling-average features...")
    featured = add_features(full)

    print("Checking for any new categories not yet known to the model...")
    if args.dry_run:
        # sync_category_codes writes category_code_map.json when it finds a new category; keep dry runs read-only
        original_map = CATEGORY_MAP_PATH.read_bytes() if CATEGORY_MAP_PATH.exists() else None
    sync_category_codes(featured)
    if args.dry_run:
        if original_map is None:
            CATEGORY_MAP_PATH.unlink(missing_ok=True)
        else:
            CATEGORY_MAP_PATH.write_bytes(original_map)

    featured = featured[OUTPUT_COLUMNS].sort_values(['Category', 'Month']).reset_index(drop=True)

    print("Topping up the per-product history the running service reads...")
    prod_existing = pd.read_csv(PRODUCT_HISTORY_PATH, dtype={'MSKU': str}) if PRODUCT_HISTORY_PATH.exists() else pd.DataFrame()
    prod_live = filter_known_categories(fetch_monthly_product_sales(), 'category', args.allow_new_categories)
    prod_new = merge_product_history(prod_existing, prod_live)

    old_rows = 0
    if FEATURES_PATH.exists():
        old_rows = sum(1 for _ in open(FEATURES_PATH)) - 1

    print(f"\nCategory file: {old_rows} -> {len(featured)} rows, categories: {featured['Category'].nunique()}, "
          f"months: {featured['Month'].min()} to {featured['Month'].max()}")
    print(f"Product file:  {len(prod_existing)} -> {len(prod_new)} rows, last month: {prod_new['Month'].max()}")

    if args.dry_run:
        print("\n--dry-run set: NOT writing the files. Preview of the newest month per category:")
        print(featured.sort_values('Month').groupby('Category').tail(1).to_string(index=False))
        return

    _write_atomic(featured, FEATURES_PATH)
    _write_atomic(prod_new, PRODUCT_HISTORY_PATH)
    print(f"\nDone. {FEATURES_PATH.name} and {PRODUCT_HISTORY_PATH.name} have been refreshed with real data.")
    print("Next step: run `python rebuild_models_v2.py` to retrain the models on this update.")


if __name__ == '__main__':
    main()
