# TechPrime AI Forecast service

Flask API (XGBoost demand + revenue models) called server-to-server by the PHP app through `forecast_api.php`.

## Rules the service enforces
- Forecast window: **1-3 months ahead** (`horizon` 1, 2 or 3; anything else returns HTTP 400).
- Every route except `/health` needs the `X-API-Key` header (`FORECAST_API_KEY`).
- `POST /api/reload` needs `X-Admin-Key` (`FORECAST_ADMIN_KEY`); it is disabled when that variable is unset.

## Monthly retrain (run on the 1st, after the previous month closes)
```
python rebuild_features_from_db.py          # append live sales to preprocessed_monthly_features.csv (--dry-run to preview)
python rebuild_models_v2.py --enforce-gate  # chronological 80/20 split, retrain, refuse to overwrite if accuracy regressed
curl -X POST -H "X-API-Key: ..." -H "X-Admin-Key: ..." $FORECAST_URL/api/reload
```
- 80/20 split is by month and redrawn each run; the 20% test months are never used for fitting or early stopping.
- Previous artifacts are copied to `artifacts_backup/<timestamp>/` before every overwrite (roll back by copying them back and reloading).
- Env vars for the DB step: `DB_HOST DB_PORT DB_NAME DB_USER DB_PASS` (use a read-only database user).
