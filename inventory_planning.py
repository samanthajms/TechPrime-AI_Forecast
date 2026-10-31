"""
inventory_planning.py
=====================
Inventory actions for the Product Demand screen: for every product that has a stock record, the forecast is turned into

  * Safety stock            buffer for demand variability during the lead time
  * Reorder point (ROP)     stock level at which a new order should be placed
  * Days of inventory (DOI) how long the stock on hand lasts at the forecast daily demand
  * Stockout risk           high / medium / low
  * Overstock warning       stock covers far more than OVERSTOCK_DAYS of forecast demand (or the product has stopped selling)
  * Recommended order qty   units to order now to get back up to the order-up-to level

FORMULAS (standard continuous-review inventory control, one product at a time)
    d            = forecast units over the horizon / days in those months           (units per day)
    sigma_month  = max(std of the product's units over the last 12 months, sqrt(mean))   (sparse sales are count data)
    SS           = ceil(z x sigma_month x sqrt(L / 30.4))                           z from the service level of the ABC class
    ROP          = ceil(d x L + SS)
    DOI          = stock_on_hand / d
    order-up-to  = ceil(ROP + d x REVIEW_DAYS)
    order qty    = order-up-to - (stock_on_hand + on_order)   when stock_on_hand + on_order <= ROP, else 0
    risk         = HIGH   stock is 0, or DOI < L (it runs out before an order placed today could arrive)
                   MEDIUM stock_on_hand + on_order <= ROP (time to reorder)
                   LOW    otherwise
    overstock    = DOI > OVERSTOCK_DAYS, or stock on hand but no forecast demand

NO SUPPLIER ROLE: the system has no supplier module, so the lead time L is a PLACEHOLDER (LEAD_TIME_DAYS, default 7),
optionally overridden per product by a `lead_time_days` column in the stock snapshot, or for one request by
?lead_time_days=. Every response says where the lead time came from. Order quantities are not rounded to supplier pack
sizes / minimum order quantities, because there is no supplier data to know them.

STOCK ON HAND is not in the forecast data. STOCK_SOURCE picks where it comes from:
    file (default)  a snapshot CSV (STOCK_PATH, default current_stock.csv), re-read whenever it changes:
                        MSKU, stock_on_hand            required
                        on_order, lead_time_days       optional
                        snapshot_at                    optional ISO timestamp (the file's modified time is used when absent)
                    Produce it with export_stock_snapshot.py, or write it from the PHP app.
    db              live from the database (same DB_* settings as the retrain job, read-only user), using STOCK_SQL
                    (default: products.sku as MSKU, products.stock as stock_on_hand). The result is cached for
                    STOCK_CACHE_SECONDS (default 600). If a refresh fails, the last good read is served and its age shows.
"""
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

import category_alignment as ca
import inventory_trends as f

BASE_DIR = Path(__file__).parent


def _env_num(name, default, cast=float):
    try:
        return cast(os.environ.get(name, default))
    except ValueError:
        raise RuntimeError(f"{name} must be a number") from None


STOCK_SOURCE = os.environ.get('STOCK_SOURCE', 'file').strip().lower()   # 'file' (snapshot CSV) or 'db' (live)
if STOCK_SOURCE not in ('file', 'db'):
    raise RuntimeError("STOCK_SOURCE must be 'file' or 'db'")
STOCK_PATH = Path(os.environ.get('STOCK_PATH', BASE_DIR / 'current_stock.csv'))
STOCK_SQL = os.environ.get('STOCK_SQL', 'SELECT p.sku AS "MSKU", p.stock AS stock_on_hand FROM products p')
STOCK_CACHE_SECONDS = _env_num('STOCK_CACHE_SECONDS', 600, int)   # live stock is re-read at most this often
STOCK_RETRY_SECONDS = 60                                           # after a failed live read, wait this long to retry
LEAD_TIME_DAYS = _env_num('LEAD_TIME_DAYS', 7, int)              # PLACEHOLDER: there is no supplier role in the system
REVIEW_DAYS = _env_num('REVIEW_DAYS', 30, int)                   # how many days of demand one order should cover
OVERSTOCK_DAYS = _env_num('OVERSTOCK_DAYS', 90, int)             # more than this many days of cover = overstock
STOCK_MAX_AGE_DAYS = _env_num('STOCK_MAX_AGE_DAYS', 2, int)      # older snapshot -> response is flagged stale
SERVICE_LEVEL = {'A': 0.98, 'B': 0.95, 'C': 0.90, None: 0.95}    # unclassified products are planned like B
DAYS_PER_MONTH = 30.4
MAX_LEAD_TIME_DAYS = 180

ACTIONS = ('order_now', 'reorder', 'awaiting_delivery', 'ok', 'overstocked', 'no_demand')
_ACTION_LABEL = {'order_now': 'Order now', 'reorder': 'Reorder',
                 'awaiting_delivery': 'Awaiting delivery: stock is low but an order is already open', 'ok': 'No action needed',
                 'overstocked': 'Overstocked: hold purchasing / consider promotion',
                 'no_demand': 'No forecast demand: review stock'}
_RISK_ORDER = {'high': 0, 'medium': 1, 'low': 2, 'none': 3}


class StockUnavailable(RuntimeError):
    """No usable stock snapshot (reported to the caller as HTTP 503, not as a bad request)."""


# -----------------------------------------------------------------
# Stock snapshot
# -----------------------------------------------------------------
def _clean_stock(raw, fallback_as_of):
    """Validate a raw stock table (from the CSV or the database) into (frame indexed by MSKU, as-of time, counts)."""
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    if not {'MSKU', 'stock_on_hand'} <= set(raw.columns):
        raise StockUnavailable("The stock data must have the columns MSKU and stock_on_hand.")
    raw['MSKU'] = raw['MSKU'].where(raw['MSKU'].notna(), '').astype(str).str.strip()
    raw['stock_on_hand'] = pd.to_numeric(raw['stock_on_hand'], errors='coerce')
    for opt in ('on_order', 'lead_time_days'):
        raw[opt] = pd.to_numeric(raw[opt], errors='coerce') if opt in raw.columns else np.nan
    total = len(raw)
    ok = raw[raw['MSKU'].ne('') & raw['stock_on_hand'].notna() & (raw['stock_on_hand'] >= 0)]
    dupes = int(ok['MSKU'].duplicated().sum())
    ok = ok.drop_duplicates('MSKU', keep='last').copy()
    ok['on_order'] = ok['on_order'].fillna(0).clip(lower=0)
    ok.loc[(ok['lead_time_days'] <= 0) | (ok['lead_time_days'] > MAX_LEAD_TIME_DAYS), 'lead_time_days'] = np.nan
    if 'snapshot_at' in raw.columns and raw['snapshot_at'].notna().any():
        as_of = pd.to_datetime(raw['snapshot_at'], errors='coerce', utc=True).max()
    else:
        as_of = pd.NaT
    if pd.isna(as_of):
        as_of = fallback_as_of
    return ok.set_index('MSKU')[['stock_on_hand', 'on_order', 'lead_time_days']], as_of, {
        'rows_in_file': int(total), 'rows_used': int(len(ok)), 'rows_skipped_invalid': int(total - len(ok) - dupes),
        'duplicate_msku_rows': dupes}


@lru_cache(maxsize=2)
def _read_stock(path, mtime):
    raw = pd.read_csv(path, dtype={'MSKU': str})
    try:
        return _clean_stock(raw, pd.Timestamp(mtime, unit='s', tz='UTC'))
    except StockUnavailable:
        raise StockUnavailable(f"{Path(path).name} must have the columns MSKU and stock_on_hand.") from None


# -----------------------------------------------------------------
# Live stock from the database (STOCK_SOURCE=db)
# -----------------------------------------------------------------
_log = logging.getLogger(__name__)
_live_lock = threading.Lock()                      # one database read at a time; other requests wait for its result
_live_cache = {'data': None, 'next_read': 0.0}


def _query_stock_table():
    """Run STOCK_SQL on the database (read-only session) and return the rows as a DataFrame."""
    import psycopg2                                # only needed when STOCK_SOURCE=db
    cfg = {'host': os.environ.get('DB_HOST'), 'port': os.environ.get('DB_PORT'), 'dbname': os.environ.get('DB_NAME'),
           'user': os.environ.get('DB_USER'), 'password': os.environ.get('DB_PASS'), 'sslmode': 'require',
           'connect_timeout': 10}
    if not (cfg['host'] and cfg['user'] and cfg['password']):
        raise StockUnavailable("STOCK_SOURCE=db needs DB_HOST, DB_PORT, DB_NAME, DB_USER and DB_PASS on the forecast service.")
    conn = psycopg2.connect(**cfg)
    try:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute(STOCK_SQL)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=cols)


def _load_live_stock():
    """Stock read live from the database, cached for STOCK_CACHE_SECONDS; on a failed refresh the last good read is kept."""
    with _live_lock:
        cached = _live_cache['data']
        if cached is not None and time.monotonic() < _live_cache['next_read']:
            return cached
        try:
            data = _clean_stock(_query_stock_table(), pd.Timestamp.now(tz='UTC'))
        except StockUnavailable:
            raise
        except Exception:                          # noqa: BLE001 - connection / SQL errors: details go to the log only
            _log.exception("live stock read failed")
            if cached is None:
                raise StockUnavailable("Stock could not be read from the database. Check the DB_* settings and "
                                       "STOCK_SQL on the forecast service.") from None
            _live_cache['next_read'] = time.monotonic() + STOCK_RETRY_SECONDS
            return cached                          # stock_as_of keeps the old time, so the response shows its age
        _live_cache.update(data=data, next_read=time.monotonic() + STOCK_CACHE_SECONDS)
        return data


def load_stock(path=None):
    """(stock frame indexed by MSKU, snapshot timestamp, row-quality counts). Raises StockUnavailable when absent.
    With STOCK_SOURCE=db and no explicit path, the stock is read live from the database instead of the CSV."""
    if path is None and STOCK_SOURCE == 'db':
        return _load_live_stock()
    path = Path(path or STOCK_PATH)
    if not path.exists():
        raise StockUnavailable(
            f"No stock snapshot found at {path.name}. Run export_stock_snapshot.py or set STOCK_PATH "
            "(columns: MSKU, stock_on_hand, optional on_order, lead_time_days, snapshot_at).")
    return _read_stock(str(path), path.stat().st_mtime)


# -----------------------------------------------------------------
# The math (pure function: easy to test and to explain in the manuscript)
# -----------------------------------------------------------------
def plan_row(stock, on_order, daily_demand, sigma_month, lead_time_days, service_level=0.95,
             review_days=REVIEW_DAYS, overstock_days=OVERSTOCK_DAYS):
    stock, on_order = float(stock), float(on_order or 0)
    d, L = max(float(daily_demand), 0.0), float(lead_time_days)
    z = NormalDist().inv_cdf(service_level)
    safety = math.ceil(z * float(sigma_month) * math.sqrt(L / DAYS_PER_MONTH) - 1e-9) if d > 0 else 0
    rop = math.ceil(d * L + safety - 1e-9) if d > 0 else 0
    order_up_to = math.ceil(rop + d * review_days - 1e-9) if d > 0 else 0
    position = stock + on_order
    doi = stock / d if d > 0 else None

    if d <= 0:
        risk = 'none'
        overstock = stock > 0
        qty = 0
        action = 'no_demand' if stock > 0 else 'ok'
    else:
        if stock <= 0 or doi < L:
            risk = 'high'
        elif position <= rop:
            risk = 'medium'
        else:
            risk = 'low'
        overstock = doi > overstock_days
        qty = max(0, order_up_to - math.floor(position + 1e-9)) if position <= rop else 0
        if qty > 0:
            action = 'order_now' if risk == 'high' else 'reorder'
        elif risk == 'high':                       # low on hand, but what is already on order covers the reorder point
            action = 'awaiting_delivery'
        else:
            action = 'overstocked' if overstock else 'ok'
    excess = max(0, math.floor(stock - d * overstock_days)) if overstock and d > 0 else (int(stock) if overstock else 0)
    return {
        'daily_demand': round(d, 3), 'lead_time_days': int(L), 'service_level': service_level,
        'safety_stock': safety, 'reorder_point': rop, 'order_up_to': order_up_to,
        'stock_on_hand': int(stock) if stock == int(stock) else round(stock, 1), 'on_order': int(on_order) if on_order == int(on_order) else round(on_order, 1),
        'days_of_inventory': None if doi is None else round(doi, 1),
        'stockout_risk': risk, 'overstock': bool(overstock), 'excess_units': excess,
        'recommended_order_qty': int(qty), 'action': action, 'action_label': _ACTION_LABEL[action],
        'message': _message(action, risk, doi, qty, stock, on_order),
    }


def _message(action, risk, doi, qty, stock, qty_on_order=0):
    if action == 'order_now':
        when = 'out of stock' if stock <= 0 else f'stock lasts about {doi:.0f} days, less than the lead time'
        return f'High stockout risk ({when}). Order {qty} units now.'
    if action == 'awaiting_delivery':
        have = 'Out of stock' if stock <= 0 else f'Only {stock:.0f} on hand'
        return f'{have}, but {qty_on_order:.0f} already on order covers the reorder point. Follow up on the delivery.'
    if action == 'reorder':
        return f'Below the reorder point. Order {qty} units.'
    if action == 'overstocked':
        return f'About {doi:.0f} days of cover: hold purchasing and consider a promotion.'
    if action == 'no_demand':
        return 'No demand forecast for this product but stock is on hand: review for markdown or return.'
    return 'Stock level is healthy.' if risk != 'none' else 'No stock and no forecast demand.'


# -----------------------------------------------------------------
# Public API
# -----------------------------------------------------------------
def _monthly_sigma(months=12):
    """Per-MSKU mean and std of monthly units over the last `months` closed months (months without a sale count as 0)."""
    h = f._history()
    ms = sorted(h['Month'].unique())[-months:]
    piv = h[h['Month'].isin(ms)].pivot_table(index='MSKU', columns='Month', values='units_sold', aggfunc='sum') \
        .reindex(columns=ms).fillna(0)
    mean = piv.mean(axis=1)
    sigma = np.maximum(piv.std(axis=1, ddof=1).fillna(0), np.sqrt(mean))
    return mean, sigma


def get_inventory_actions(horizon=3, category=None, abc_class=None, action=None, top=100, lead_time_days=None):
    f.check_horizon(horizon)
    if lead_time_days is not None and not 1 <= int(lead_time_days) <= MAX_LEAD_TIME_DAYS:
        raise ValueError(f"lead_time_days must be between 1 and {MAX_LEAD_TIME_DAYS}")
    if action is not None and action not in ACTIONS:
        raise ValueError(f"action must be one of: {', '.join(ACTIONS)}")
    if abc_class is not None and str(abc_class).strip().upper() not in ('A', 'B', 'C'):
        raise ValueError("class must be A, B or C")
    stock, as_of, quality = load_stock()

    pf = f._product_forecast(horizon)
    flt = None
    if category:
        kind, val = ca.resolve_filter(category)
        pf = f._filter(pf, kind, val); flt = {'kind': kind, 'value': val}
    months = sorted(pf['month'].unique()) if len(pf) else []
    days = sum(pd.Period(m, freq='M').days_in_month for m in months)
    fc_units = pf.groupby('msku')['u'].sum() if len(pf) else pd.Series(dtype=float)
    demand = fc_units / days if days else pd.Series(dtype=float)

    h = f._history()
    info = h.sort_values('Month').groupby('MSKU')[['Product Name', 'Shop Category', 'Shop Group']].last()
    classes = f._abc()['abc_class']
    mean_m, sigma_m = _monthly_sigma()

    msku_set = set(stock.index) & set(info.index)
    if category:                                   # keep products of the category: forecast rows OR stock on hand
        in_cat = set(info.index[info['Shop Category'] == flt['value']]) if flt['kind'] == 'label' else \
            set(info.index[info['Shop Group'] == flt['value']]) if flt['kind'] == 'group' else set(pf['msku'])
        msku_set &= in_cat
    rows = []
    for m in msku_set:
        cls = classes.get(m)
        if abc_class and cls != str(abc_class).strip().upper():
            continue
        s = stock.loc[m]
        lt = lead_time_days if lead_time_days is not None else (None if pd.isna(s['lead_time_days']) else float(s['lead_time_days']))
        source = 'request' if lead_time_days is not None else 'stock snapshot' if lt is not None else 'placeholder default'
        lt = LEAD_TIME_DAYS if lt is None else lt
        r = plan_row(s['stock_on_hand'], s['on_order'], float(demand.get(m, 0.0)), float(sigma_m.get(m, 0.0)), lt,
                     SERVICE_LEVEL.get(cls, SERVICE_LEVEL[None]))
        r.update({'msku': m, 'product_name': info.loc[m, 'Product Name'], 'category': info.loc[m, 'Shop Category'],
                  'group': info.loc[m, 'Shop Group'], 'abc_class': cls, 'lead_time_source': source,
                  'forecast_units': round(float(fc_units.get(m, 0.0)), 1)})
        if action and r['action'] != action:
            continue
        rows.append(r)
    rows.sort(key=lambda r: (_RISK_ORDER[r['stockout_risk']], r['days_of_inventory'] if r['days_of_inventory'] is not None else 1e9,
                             r['abc_class'] or 'Z'))

    summary = {a: sum(1 for r in rows if r['action'] == a) for a in ACTIONS}
    now = datetime.now(timezone.utc)
    age = (now - as_of.to_pydatetime()).total_seconds() / 86400
    no_stock = len(set(demand.index) - set(stock.index))
    return {
        'horizon_months': horizon, 'forecast_months': months, 'filter': flt,
        'stock_source': 'database (live)' if STOCK_SOURCE == 'db' else 'snapshot file',
        'stock_as_of': as_of.isoformat(), 'stock_age_days': round(age, 1), 'stock_stale': bool(age > STOCK_MAX_AGE_DAYS),
        'summary': summary, 'total_products': len(rows),
        'assumptions': {
            'lead_time_days_default': LEAD_TIME_DAYS,
            'lead_time_note': 'PLACEHOLDER: the system has no supplier role, so lead time is a default (or the value in the stock snapshot / request).',
            'service_level_by_abc_class': {k or 'unclassified': v for k, v in SERVICE_LEVEL.items()},
            'review_days': REVIEW_DAYS, 'overstock_days': OVERSTOCK_DAYS,
            'demand_basis': 'category forecast allocated to products by 12-month share (top-down); thin for slow sellers',
            'order_quantity_rounding': 'none (no supplier pack-size / MOQ data)',
        },
        'data_quality': {**quality, 'forecast_products_without_stock_record': no_stock},
        'products': rows[:int(top)],
    }
