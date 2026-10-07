"""TechPrime AI forecast service (Flask). Runs on Render; called server-to-server by the PHP app."""
import hmac
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv(Path(__file__).parent / ".env")
import inventory_forecasting as forecast  # noqa: E402  (after load_dotenv so HISTORY_PATH etc. are read)

API_KEY = os.environ.get("FORECAST_API_KEY", "")
if not API_KEY:
    raise RuntimeError("FORECAST_API_KEY is not set - refusing to start an unauthenticated forecast service.")

# Separate secret for the maintenance endpoint (/api/reload). Unset = the endpoint is disabled.
ADMIN_KEY = os.environ.get("FORECAST_ADMIN_KEY", "")

app = Flask(__name__)


@app.before_request
def check_api_key():
    if request.endpoint == "health":
        return None
    if not hmac.compare_digest(request.headers.get("X-API-Key", ""), API_KEY):
        return {"error": "unauthorized"}, 401
    return None


class BadRequest(ValueError):
    """Invalid query parameter (reported to the caller as HTTP 400)."""


def _int_arg(name, default):
    raw = request.args.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise BadRequest(f"{name} must be a whole number") from None


def _horizon(default=forecast.MAX_HORIZON):
    """Months ahead, 1..MAX_HORIZON (3). Out-of-range values are rejected, not silently clamped."""
    return forecast.check_horizon(_int_arg("horizon", default))


def _top(default=50):
    return max(1, min(_int_arg("top", default), 300))


def _category():
    c = (request.args.get("category") or "").strip()
    if len(c) > 80:
        raise BadRequest("category is too long")
    return c or None


def _safe(fn):
    try:
        return jsonify(fn())
    except ValueError as e:                      # bad parameter / category / product, not enough history
        return {"error": str(e)}, 400
    except Exception:                            # noqa: BLE001
        app.logger.exception("forecast error")   # details go to the server log only, never to the caller
        return {"error": "internal error"}, 500


@app.route("/health")
def health():
    return {"status": "ok"}, 200


@app.route("/api/forecast/demand")            # Product Demand Forecast screen
def demand():
    return _safe(lambda: forecast.get_product_demand(
        horizon=_horizon(), category=_category(), top=_top()))


@app.route("/api/forecast/revenue")           # Sales / Revenue Forecast screen
def revenue():
    return _safe(lambda: forecast.get_revenue_forecast(horizon=_horizon(), category=_category()))


@app.route("/api/forecast")                   # backward compatible with the existing forecast_api.php
def legacy():
    return _safe(lambda: forecast.get_forecast(
        category=_category() or (None if request.args.get("product") else "MEMORY"),
        product=request.args.get("product"), horizon=_horizon()))


@app.route("/api/metrics")
def metrics():
    return _safe(forecast.get_model_metrics)


@app.route("/api/reload", methods=["POST"])   # call after a retrain has replaced the CSV/model files
def reload_data():
    if not ADMIN_KEY:
        return {"error": "reload is disabled (FORECAST_ADMIN_KEY is not set)"}, 403
    if not hmac.compare_digest(request.headers.get("X-Admin-Key", ""), ADMIN_KEY):
        return {"error": "unauthorized"}, 401
    try:
        forecast.reload_history()                # reloads models + confidence tables + history, clears caches
    except Exception:                            # noqa: BLE001  (old artifacts stay active: reload_models swaps atomically)
        app.logger.exception("reload failed")
        return {"error": "reload failed; previous model is still active"}, 500
    return {"status": "reloaded"}



@app.route("/api/categories")
def categories():
    return _safe(forecast.get_categories)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
