"""TechPrime AI forecast service (Flask). Runs on Render; called server-to-server by the PHP app."""
import hmac
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix

load_dotenv(Path(__file__).parent / ".env")
import inventory_forecasting as forecast  # noqa: E402  (after load_dotenv so HISTORY_PATH etc. are read)

API_KEY = os.environ.get("FORECAST_API_KEY", "")
if not API_KEY:
    raise RuntimeError("FORECAST_API_KEY is not set - refusing to start an unauthenticated forecast service.")

# Separate secret for the maintenance endpoint (/api/reload). Unset = the endpoint is disabled.
ADMIN_KEY = os.environ.get("FORECAST_ADMIN_KEY", "")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024            # this API takes no request bodies
# Behind Render's proxy the real client address is in X-Forwarded-For. Trust only the hops we actually have
# (TRUSTED_PROXY_HOPS, default 1) so a caller cannot spoof its IP to dodge the limits below.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=int(os.environ.get("TRUSTED_PROXY_HOPS", "1")), x_proto=1)

RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "120"))      # requests / minute / client IP
RELOAD_LIMIT_PER_MIN = int(os.environ.get("RELOAD_LIMIT_PER_MIN", "5"))    # /api/reload attempts / minute / client IP
AUTH_FAIL_LIMIT = int(os.environ.get("AUTH_FAIL_LIMIT", "10"))             # bad keys allowed ...
AUTH_FAIL_WINDOW = int(os.environ.get("AUTH_FAIL_WINDOW_SECONDS", "300"))  # ... per this many seconds, then 429


class _SlidingWindow:
    """Tiny in-process sliding-window counter (no extra dependency). State is per gunicorn worker, which is why
    render.yaml runs one worker with several threads; use a shared store (Redis) before adding workers."""

    def __init__(self):
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, q, now, window):
        while q and now - q[0] > window:
            q.popleft()

    def count(self, key, window):
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            self._prune(q, now, window)
            return len(q)

    def hit(self, key, window):
        """Record a hit and return how many hits are now inside the window."""
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > 10000:                  # bound memory under a spray of spoofed addresses
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] > window]:
                    del self._hits[k]
            q = self._hits[key]
            self._prune(q, now, window)
            q.append(now)
            return len(q)


_limiter = _SlidingWindow()


def _too_many(window):
    return {"error": "too many requests"}, 429, {"Retry-After": str(window)}


@app.before_request
def check_api_key():
    if request.endpoint == "health":
        return None
    ip = request.remote_addr or "unknown"

    # Locked out after repeated bad keys: refuse before even looking at the key, so guessing gets no feedback.
    if _limiter.count(("fail", ip), AUTH_FAIL_WINDOW) >= AUTH_FAIL_LIMIT:
        return _too_many(AUTH_FAIL_WINDOW)

    if not hmac.compare_digest(request.headers.get("X-API-Key", "").encode(), API_KEY.encode()):
        _limiter.hit(("fail", ip), AUTH_FAIL_WINDOW)
        app.logger.warning("rejected request: bad API key from %s on %s", ip, request.path)
        return {"error": "unauthorized"}, 401

    if _limiter.hit(("req", ip), 60) > RATE_LIMIT_PER_MIN:
        return _too_many(60)
    return None


@app.after_request
def security_headers(resp):
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return resp


@app.errorhandler(404)
def not_found(_e):
    return {"error": "not found"}, 404


@app.errorhandler(405)
def method_not_allowed(_e):
    return {"error": "method not allowed"}, 405


@app.errorhandler(413)
def too_large(_e):
    return {"error": "request too large"}, 413


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


def _month_arg(name):
    """Optional ?from= / ?to= as YYYY-MM or YYYY-MM-DD -> 'YYYY-MM' (or None)."""
    raw = (request.args.get(name) or "").strip()
    if not raw:
        return None
    try:
        return str(pd.Period(raw[:7], freq="M"))
    except Exception:  # noqa: BLE001
        raise BadRequest(f"{name} must be a date like 2026-01 or 2026-01-31") from None


def _range():
    start, end = _month_arg("from"), _month_arg("to")
    if start and end and start > end:
        raise BadRequest("from must not be after to")
    return start, end


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
        horizon=_horizon(), category=_category(), top=_top(), start=_range()[0], end=_range()[1]))


@app.route("/api/forecast/revenue")           # Sales / Revenue Forecast screen
def revenue():
    return _safe(lambda: forecast.get_revenue_forecast(
        horizon=_horizon(), category=_category(), start=_range()[0], end=_range()[1]))


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
    ip = request.remote_addr or "unknown"
    if not ADMIN_KEY:
        return {"error": "reload is disabled (FORECAST_ADMIN_KEY is not set)"}, 403
    if _limiter.hit(("reload", ip), 60) > RELOAD_LIMIT_PER_MIN:
        return _too_many(60)
    if not hmac.compare_digest(request.headers.get("X-Admin-Key", "").encode(), ADMIN_KEY.encode()):
        _limiter.hit(("fail", ip), AUTH_FAIL_WINDOW)
        app.logger.warning("rejected reload: bad admin key from %s", ip)
        return {"error": "unauthorized"}, 401
    try:
        forecast.reload_history()                # reloads models + confidence tables + history, clears caches
    except Exception:                            # noqa: BLE001  (old artifacts stay active: reload_models swaps atomically)
        app.logger.exception("reload failed")
        return {"error": "reload failed; previous model is still active"}, 500
    app.logger.info("models reloaded by %s", ip)
    return {"status": "reloaded"}


@app.route("/api/categories")
def categories():
    return _safe(forecast.get_categories)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
