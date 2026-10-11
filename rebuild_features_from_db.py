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
(2024-07 to 2026-08) came from an OLDER, separate POS system, used to originally
train/test the model, before TechPrime-AI's live orders table existed. The
live `orders` table is newer and does not cover that same history. So this
script treats the existing CSV as the historical foundation and only ADDS
months that come from real live orders and aren't already covered -- it never
deletes or overwrites the old POS-era rows.

WHAT IT DOES, STEP BY STEP:
  1. Reads the CURRENT preprocessed_monthly_features.csv. Every row already in it (the old POS history) is kept
     EXACTLY as it is -- nothing is recomputed, so the warm-up history behind the first months is never lost.
  2. Connects to the same Postgres (Supabase) database the PHP app uses, using the .env file in this folder.
  3. Asks the database: "for every CLOSED month and every product category, how many units sold and how much revenue
     came in?" (the month in progress is excluded in the SQL itself).
  4. APPENDS only the months AFTER the last month already in the file. Months the file already has are never touched.
  5. Fills gaps (a category with a month of zero sales gets a row of zeros -- the model needs an unbroken timeline).
  6. Computes the lag / rolling features for the NEW rows only, from the months BEFORE each one (same convention as the
     existing file and as inference in inventory_forecasting.py: the current month is never part of its own features).
  7. Applies the closed-month filter once more on the combined result, so an unfinished month can never be in the file.
  8. Keeps category_code_map.json in sync: if the database has a category the model has never seen, it is skipped
     (or added with a new code under --allow-new-categories).

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

from month_guard import drop_open_months

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
# Step 1a: load the EXISTING file -- the pre-initialized history (2024-07 to 2026-08). It is kept verbatim: its rows are
# never recomputed, reordered by value or dropped, because their lag columns were built with months (2024-01..06) that
# are no longer in the file and could not be rebuilt from it.
# -----------------------------------------------------------------
def load_existing_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.read_csv(FEATURES_PATH)[OUTPUT_COLUMNS]


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
# Step 2: features for ONE new month, from the months before it.
# Same convention as the existing file and as _build_demand_features/_build_revenue_features in
# inventory_forecasting.py: lags and rolling windows only look at PAST months, never at the month being described.
# (A window that includes the month itself leaks the answer into the features and the test score looks better than the
# model really is.)
# -----------------------------------------------------------------
def next_month_features(units, revenue, price) -> dict:
    """Features for the month that follows the given history (oldest first, same length)."""
    n = len(units)
    last = lambda arr, k: float(arr[-k]) if n >= k else 0.0                 # noqa: E731
    u3, r3, p3 = units[-3:], revenue[-3:], price[-3:]
    price_trend = (price[-1] - price[-4]) / price[-4] if n >= 4 and price[-4] else 0.0
    return {
        'lag_1': last(units, 1), 'lag_2': last(units, 2), 'lag_3': last(units, 3),
        'lag_12': last(units, 12) if n >= 12 else last(units, 3),            # inference falls back to lag_3
        'roll_mean_3': float(np.mean(u3)) if n else 0.0,
        'roll_mean_6': float(np.mean(units[-6:])) if n else 0.0,
        'roll_std_3': float(np.std(u3, ddof=1)) if len(u3) > 1 else 0.0,
        'avg_price_lag_1': float(price[-1]) if n else 0.0,
        'avg_price_roll_mean_3': float(np.mean(p3)) if n else 0.0,
        'price_trend': float(price_trend),
        'revenue_lag_1': last(revenue, 1),
        'revenue_lag_12': last(revenue, 12) if n >= 12 else last(revenue, 1),
        'revenue_roll_mean_3': float(np.mean(r3)) if n else 0.0,
        'revenue_roll_mean_6': float(np.mean(revenue[-6:])) if n else 0.0,
    }


# -----------------------------------------------------------------
# Step 3: APPEND the live months that come after the file's last month. Existing rows are never touched.
# -----------------------------------------------------------------
def append_live_months(existing: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """New rows only (OUTPUT_COLUMNS, no category_code yet). `live` has columns category, month (Period), units_sold,
    revenue. Every category gets a row for every new month (zeros when nothing sold) so the timeline stays unbroken."""
    base = existing[existing['Category'] != 'All']      # 'All' is a legacy junk bucket (negative revenue), not a category
    last_month = pd.Period(base['Month'].max(), freq='M') if not base.empty else None
    live = live[live['month'] > last_month] if last_month is not None else live
    if live.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    new_months = pd.period_range(live['month'].min(), live['month'].max(), freq='M')
    categories = sorted(set(base['Category']) | set(live['category']))
    sales = live.set_index(['category', 'month'])
    rows = []
    for cat in categories:
        hist = base[base['Category'] == cat].sort_values('Month')
        units = [float(x) for x in hist['units_sold']]
        revenue = [float(x) for x in hist['revenue']]
        price = [float(x) for x in hist['avg_price']]
        for month in new_months:
            feats = next_month_features(units, revenue, price)
            u, r = (float(sales.at[(cat, month), 'units_sold']), float(sales.at[(cat, month), 'revenue'])) \
                if (cat, month) in sales.index else (0.0, 0.0)
            u = float(round(u))
            p = (r / u) if u > 0 else (price[-1] if price else 0.0)   # a zero-sales month carries the last known price
            rows.append({
                'Month': str(month), 'Category': cat, 'units_sold': int(u), 'revenue': round(r, 2), 'avg_price': round(p, 2),
                'lag_1': int(feats['lag_1']), 'lag_2': int(feats['lag_2']), 'lag_3': int(feats['lag_3']),
                'lag_12': int(feats['lag_12']), 'roll_mean_3': feats['roll_mean_3'], 'roll_mean_6': feats['roll_mean_6'],
                'roll_std_3': feats['roll_std_3'], 'avg_price_lag_1': round(feats['avg_price_lag_1'], 2),
                'avg_price_roll_mean_3': round(feats['avg_price_roll_mean_3'], 2), 'price_trend': round(feats['price_trend'], 4),
                'revenue_lag_1': round(feats['revenue_lag_1'], 2), 'revenue_lag_12': round(feats['revenue_lag_12'], 2),
                'revenue_roll_mean_3': round(feats['revenue_roll_mean_3'], 2),
                'revenue_roll_mean_6': round(feats['revenue_roll_mean_6'], 2),
                'month_num': month.month, 'quarter': month.quarter,
            })
            units.append(u); revenue.append(r); price.append(p)
    return pd.DataFrame(rows).reindex(columns=[c for c in OUTPUT_COLUMNS if c != 'category_code'])


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

    print("Reading the pre-initialized history (kept verbatim, never recomputed)...")
    existing = load_existing_features()

    print("Connecting to the database and pulling closed-month sales (counted orders + completed POS sales)...")
    live = filter_known_categories(fetch_monthly_sales(), 'category', args.allow_new_categories)

    print("Appending only the live months after the file's last month...")
    new_rows = append_live_months(existing, live)

    print("Checking for any new categories not yet known to the model...")
    if args.dry_run:
        # sync_category_codes writes category_code_map.json when it finds a new category; keep dry runs read-only
        original_map = CATEGORY_MAP_PATH.read_bytes() if CATEGORY_MAP_PATH.exists() else None
    if not new_rows.empty:
        sync_category_codes(new_rows)
    else:
        new_rows['category_code'] = pd.Series(dtype=int)
    if args.dry_run:
        if original_map is None:
            CATEGORY_MAP_PATH.unlink(missing_ok=True)
        else:
            CATEGORY_MAP_PATH.write_bytes(original_map)

    # Closed-month filter, applied AFTER the append so the combined file can never end on an unfinished month.
    combined = pd.concat([existing, new_rows[OUTPUT_COLUMNS]], ignore_index=True)
    featured = drop_open_months(combined).sort_values(['Category', 'Month']).reset_index(drop=True)
    dropped = len(combined) - len(featured)
    if dropped:
        print(f"Dropped {dropped} row(s) belonging to the month in progress.")

    print("Topping up the per-product history the running service reads...")
    prod_existing = pd.read_csv(PRODUCT_HISTORY_PATH, dtype={'MSKU': str}) if PRODUCT_HISTORY_PATH.exists() else pd.DataFrame()
    prod_live = filter_known_categories(fetch_monthly_product_sales(), 'category', args.allow_new_categories)
    prod_new = drop_open_months(merge_product_history(prod_existing, prod_live))

    print(f"\nCategory file: {len(existing)} -> {len(featured)} rows ({len(new_rows)} appended), "
          f"categories: {featured['Category'].nunique()}, months: {featured['Month'].min()} to {featured['Month'].max()}")
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
