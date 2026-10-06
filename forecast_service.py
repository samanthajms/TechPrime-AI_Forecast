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

app = Flask(__name__)


@app.before_request
def check_api_key():
    if request.endpoint == "health":
        return None
    if not hmac.compare_digest(request.headers.get("X-API-Key", ""), API_KEY):
        return {"error": "unauthorized"}, 401
    return None


def _horizon(default=3):
    return max(1, min(int(request.args.get("horizon", default)), 6))


def _safe(fn):
    try:
        return jsonify(fn())
    except ValueError as e:                      # bad category/product, not enough history
        return {"error": str(e)}, 400
    except Exception as e:                       # noqa: BLE001
        app.logger.exception("forecast error")
        return {"error": "internal error", "detail": str(e)}, 500


@app.route("/health")
def health():
    return {"status": "ok"}, 200


@app.route("/api/forecast/demand")            # Product Demand Forecast screen
def demand():
    return _safe(lambda: forecast.get_product_demand(
        horizon=_horizon(), category=request.args.get("category"), top=int(request.args.get("top", 50))))


@app.route("/api/forecast/revenue")           # Sales / Revenue Forecast screen
def revenue():
    return _safe(lambda: forecast.get_revenue_forecast(horizon=_horizon(), category=request.args.get("category")))


@app.route("/api/forecast")                   # backward compatible with the existing forecast_api.php
def legacy():
    return _safe(lambda: forecast.get_forecast(
        category=request.args.get("category") or (None if request.args.get("product") else "MEMORY"),
        product=request.args.get("product"), horizon=_horizon()))


@app.route("/api/metrics")
def metrics():
    return _safe(forecast.get_model_metrics)


@app.route("/api/reload", methods=["POST"])   # call after a retrain has replaced the CSV/model files
def reload_data():
    forecast.reload_history()
    return {"status": "reloaded"}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
