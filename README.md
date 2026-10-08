# TechPrime AI Forecast service

Flask API (XGBoost demand + revenue models) called server-to-server by the PHP app through `forecast_api.php`.
Deployed on Render as a web service (see `render.yaml`). Security setup: [`docs/RENDER_SECURITY_GUIDE.md`](docs/RENDER_SECURITY_GUIDE.md).

## Rules the service enforces
- **Forecast window: 1-3 months ahead** (`horizon` 1, 2 or 3; anything else returns HTTP 400). The PHP proxy and both forecast screens enforce the same window (a 1 / 2 / 3 month dropdown).
- Every route except `/health` needs the `X-API-Key` header (`FORECAST_API_KEY`). Repeated bad keys lock the caller out (429); all callers are rate limited.
- `POST /api/reload` needs `X-Admin-Key` (`FORECAST_ADMIN_KEY`); it is disabled when that variable is unset.

## Data and the 80/20 split
- Initial input: the historical POS export (2024-01 to 2026-08), already built into `preprocessed_monthly_features.csv` (per category, used for training) and `preprocessed_monthly_product_features.csv` (per product, read by the running service).
- `rebuild_models_v2.py` splits **chronologically by month, 80% train / 20% test**, redrawn each run. Early stopping uses only the last train months, so the test months are never used for fitting or tuning.
- Test results feed `category_confidence.csv`, `revenue_category_confidence.csv`, `monthly_predictions.csv`, `revenue_predictions.csv` and `GET /api/metrics`.

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
