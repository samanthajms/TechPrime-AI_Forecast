# TechPrime AI Forecast service

Flask API (XGBoost demand model; revenue derived from it) called server-to-server by the PHP app through `forecast_api.php`.
Deployed on Render as a web service (see `render.yaml`). Security setup: [`docs/RENDER_SECURITY_GUIDE.md`](docs/RENDER_SECURITY_GUIDE.md).

## Rules the service enforces
- **Forecast window: 1-3 months ahead** (`horizon` 1, 2 or 3; anything else returns HTTP 400). The PHP proxy and both forecast screens enforce the same window (a 1 / 2 / 3 month dropdown).
- Every route except `/health` needs the `X-API-Key` header (`FORECAST_API_KEY`). Repeated bad keys lock the caller out (429); all callers are rate limited.
- `POST /api/reload` needs `X-Admin-Key` (`FORECAST_ADMIN_KEY`); it is disabled when that variable is unset.

## Data and the 80/20 split
- Initial input: the historical POS export (2024-01 to 2026-08), already built into `preprocessed_monthly_features.csv` (per category, used for training) and `preprocessed_monthly_product_features.csv` (per product, read by the running service).
- `rebuild_models_v2.py` splits **chronologically by month, 80% train / 20% test**, redrawn each run. Early stopping uses only the last train months, so the test months are never used for fitting or tuning.
- Test results feed `category_confidence.csv`, `revenue_category_confidence.csv`, `monthly_predictions.csv`, `revenue_predictions.csv` and `GET /api/metrics`.

## Categories
`Category` in both history files, `category_code_map.json` and every forecast is one of the **system's own category labels** (`category_alignment.taxonomy()`: Chassis Fan, CPU Cooling, Memory, Processor INTEL, ... grouped by `Category Group`: Component, Peripherals, ...). There is no aggregate "All" category and no legacy POS bucket (Combo -> Keyboard and Mouse, Display -> Monitor, ...). The overall total is the sum of the category forecasts (`total` in `/api/forecast/demand` and `/api/forecast/revenue`).

## Accuracy metrics (`model_accuracy.json`, `GET /api/metrics`)
Computed on the held-out 20% months by `accuracy_metrics.py`: **MAE** and **RMSE** (same unit as the target, scale-dependent), **WAPE** = sum|error| / sum(actual) (scale-free, volume-weighted, safe with zero sales; MAPE is not used for that reason), **NRMSE** = RMSE / mean(actual), **bias**, and the **total-level error** (error of the category forecasts added up). Two no-model benchmarks (last month, 6-month average) are reported next to them; the model should be at or below those.

## How revenue is forecast

One learned model: the XGBoost demand model (units per category and month). **Revenue = predicted units x expected unit price**, where the expected price is the mean of the category's last 3 monthly average prices, held flat over the 1-3 month horizon (`inventory_forecasting.expected_unit_price`; the retrain scores exactly this rule). A separate XGBoost revenue model was retired: on the same held-out months it had 35.0% WAPE (285% on the lower-volume half of the categories) against 18.7% for this rule, and its month-to-month recursion compounded errors. Price *changes* are carried forward, not predicted. The 6-month average of revenue (13.5% WAPE) is still better on these 3 test months: revisit a learned revenue model with 24+ months of data. `xgb_monthly_revenue_model.json` is no longer used or produced.

## Growth, ABC classification and inventory actions

**Month-over-month (MoM) growth.** `GET /api/forecast/revenue` returns a `growth` block (latest closed month vs the month before it, units and revenue) and `units_mom_pct` / `revenue_mom_pct` on every `total`, `by_category` and `by_group` row: the first forecast month is compared with the latest *actual* month, each later one with the forecast month before it. `/api/forecast/demand` has the same fields on its `total`. `from` / `to` never change growth. `null` means the base month had no sales (growth from nothing is undefined). Categories that sold almost nothing in the base month can still show extreme percentages (a jump from 1 to 7 units is +600%).

**ABC classification.** `GET /api/abc[?category=&class=A|B|C&top=]` and an `abc_class` on every product row of `/api/forecast/demand`. A = the products making up the first 80% of net revenue over the last 12 closed months, B = the next 15%, C = the last 5%. Classes are always computed across the whole shop; `category` and `class` only filter what is listed.

**Inventory actions.** `GET /api/inventory/actions[?horizon=&category=&class=&action=&top=&lead_time_days=]` gives, per product, safety stock, reorder point, days of inventory, stockout risk (high / medium / low), an overstock warning, and the recommended order quantity. The formulas are in the header of `inventory_planning.py` and are checked in the tests. Actions: `order_now`, `reorder`, `awaiting_delivery`, `ok`, `overstocked`, `no_demand`.

* **No supplier role.** Lead time is a **placeholder**: `LEAD_TIME_DAYS` (default 7), overridable per product with a `lead_time_days` column in the stock snapshot or per request with `?lead_time_days=`. Every product row says where its lead time came from (`lead_time_source`). Order quantities are not rounded to pack sizes / minimum order quantities. Replace the placeholder when a supplier table exists.
* **Stock on hand** is not part of the forecast data. The service reads a snapshot file, `STOCK_PATH` (default `current_stock.csv`): columns `MSKU, stock_on_hand` (+ optional `on_order, lead_time_days, snapshot_at`). Create it with `python export_stock_snapshot.py` (check the SQL against your schema first) or write it from the PHP app. Without the file the endpoint answers **503**. The response reports `stock_as_of` and `stock_stale` (older than `STOCK_MAX_AGE_DAYS`, default 2): a stale snapshot gives wrong alerts.
* `current_stock.sample.csv` is **synthetic** (made-up quantities, for trying the endpoint). Use it with `STOCK_PATH=current_stock.sample.csv`; never treat its alerts as real.
* Other settings: `REVIEW_DAYS` (30, days of demand one order covers), `OVERSTOCK_DAYS` (90). Service level by ABC class: A 98%, B 95%, C 90%.
* Product demand is the category forecast allocated by 12-month share, so slow sellers get a rough rate; their safety stock uses `max(std, sqrt(mean))` of monthly units because sparse sales behave like counts.

## Date-range filter (optional `from` / `to`)
`/api/forecast/demand` and `/api/forecast/revenue` accept `from=` and `to=` (`YYYY-MM` or `YYYY-MM-DD`). They only choose which **actual** sales months are returned (revenue `history`; demand `range_units` / `range_revenue`). The forecast itself always starts after the latest month in the data and is 1-3 months long.

## Demo: move the forecast window
By default forecasts start the month after the last month in the history. To demonstrate output against months whose real sales are known, set `HISTORY_CUTOFF=YYYY-MM` (Render dashboard -> Environment, or locally) and restart/reload:

```
HISTORY_CUTOFF=2026-05   ->  horizon=3 forecasts 2026-06, 2026-07, 2026-08  (compare with the real months)
```
Unset it to return to normal. `GET /api/metrics` reports the active cutoff under `data.history_cutoff`.

## Monthly retrain (automatic)
`.github/workflows/monthly_retrain.yml` runs on the 2nd of every month (or manually: Actions -> Monthly retrain -> Run workflow, with an optional dry run):

1. `rebuild_features_from_db.py` appends **closed months only** from the live database to both history files. It counts online orders in status `to_ship / to_receive / delivered / completed` plus completed cashier POS sales (voided, cancelled, unpaid and pending are ignored), buckets months in Asia/Manila, skips product categories the model has no code for (use `--allow-new-categories` to keep them), and opens the database connection **read-only**.
2. `rebuild_models_v2.py` retrains on the chronological 80/20 split. The **regression gate is on by default**: if the new models are worse than the deployed ones nothing is overwritten and the job fails (exit 2). `--no-enforce-gate` overrides.
3. Unit tests run (`python -m unittest discover -s tests -v`).
4. The job opens a pull request `retrain/YYYY-MM` containing the new data + models and the gate report. **Merging it makes Render redeploy** with the new files - Render's disk is ephemeral, so models must live in git, not be retrained on the running service. Git history is the rollback (`git revert` the merge).

One-time setup: GitHub repo -> Settings -> Secrets and variables -> Actions: `DB_HOST DB_PORT DB_NAME DB_USER DB_PASS` (a read-only database user - see the security guide); Settings -> Actions -> General -> "Allow GitHub Actions to create and approve pull requests".

Manual run (same steps):
```
pip install -r requirements-retrain.txt
python rebuild_features_from_db.py --dry-run     # preview
python rebuild_features_from_db.py
python rebuild_models_v2.py
```
Previous artifacts are also copied to `artifacts_backup/<timestamp>/` locally before every overwrite.

## Requirements files
- `requirements.txt` - what the Render web service installs (runtime only).
- `requirements-retrain.txt` - adds the database driver and Excel reader for the retrain job. Not installed on Render.

## Tests
```
pip install -r requirements.txt
python -m unittest discover -s tests -v
```
Covers the 1-3 month window, the history cutoff, the 80/20 chronological split, API-key / lockout / reload auth, security headers, and the live-data query rules.
