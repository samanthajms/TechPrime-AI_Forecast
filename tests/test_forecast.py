"""Run: python -m unittest discover -s tests -v   (from the repo root)"""
import importlib
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FORECAST_API_KEY", "test-key")
os.environ["FORECAST_ADMIN_KEY"] = "admin-key"
os.environ["RATE_LIMIT_PER_MIN"] = "1000"
os.environ["AUTH_FAIL_LIMIT"] = "3"


class Horizon(unittest.TestCase):
    def test_only_1_to_3(self):
        import inventory_forecasting as f
        for ok in (1, 2, 3):
            self.assertEqual(f.check_horizon(ok), ok)
        for bad in (0, 4, 6, -1, "2", 2.0, True, None):
            with self.assertRaises(ValueError):
                f.check_horizon(bad)

    def test_forecast_months_follow_history_cutoff(self):
        import inventory_forecasting as f
        try:
            os.environ["HISTORY_CUTOFF"] = "2026-05"
            f.reload_history()
            rows = f.get_forecast(category="MEMORY", horizon=3)
            self.assertEqual([r["month"] for r in rows], ["2026-06", "2026-07", "2026-08"])
            os.environ.pop("HISTORY_CUTOFF")
            f.reload_history()
            rows = f.get_forecast(category="MEMORY", horizon=1)
            self.assertEqual(rows[0]["month"], "2026-09")
        finally:
            os.environ.pop("HISTORY_CUTOFF", None)
            f.reload_history()


class Split(unittest.TestCase):
    def test_80_20_chronological_by_month(self):
        import pandas as pd
        import rebuild_models_v2 as r
        months = pd.period_range("2024-01", periods=25, freq="M")
        df = pd.DataFrame({"Month": list(months) * 2, "Category": ["A"] * 25 + ["B"] * 25})
        train, test, tm, sm = r._chronological_split(df)
        self.assertEqual((len(tm), len(sm)), (20, 5))
        self.assertLess(max(tm), min(sm))                       # no test month is earlier than a train month
        self.assertFalse(set(tm) & set(sm))


class Service(unittest.TestCase):
    def setUp(self):
        import forecast_service
        importlib.reload(forecast_service)
        self.svc = forecast_service
        self.c = forecast_service.app.test_client()
        self.h = {"X-API-Key": "test-key"}

    def test_health_open_everything_else_needs_key(self):
        self.assertEqual(self.c.get("/health").status_code, 200)
        self.assertEqual(self.c.get("/api/categories").status_code, 401)
        self.assertEqual(self.c.get("/api/categories", headers=self.h).status_code, 200)

    def test_horizon_window_enforced_over_http(self):
        for h, code in ((1, 200), (3, 200), (0, 400), (4, 400), (6, 400), ("x", 400)):
            r = self.c.get(f"/api/forecast/revenue?horizon={h}", headers=self.h)
            self.assertEqual(r.status_code, code, h)

    def test_date_range_filters_history_not_forecast(self):
        full = self.c.get("/api/forecast/revenue?horizon=2", headers=self.h).get_json()
        part = self.c.get("/api/forecast/revenue?horizon=2&from=2026-06-01&to=2026-08-31", headers=self.h).get_json()
        self.assertEqual([r["month"] for r in part["history"]], ["2026-06", "2026-07", "2026-08"])
        self.assertGreater(len(full["history"]), 3)
        self.assertEqual(full["total"], part["total"])               # forecast untouched
        d = self.c.get("/api/forecast/demand?horizon=1&top=3&from=2026-06&to=2026-08", headers=self.h).get_json()
        self.assertIn("range_units", d["products"][0])
        self.assertEqual(d["range"], {"from": "2026-06", "to": "2026-08"})

    def test_bad_date_range_rejected(self):
        for q in ("from=nope", "from=2026-08&to=2026-01"):
            r = self.c.get(f"/api/forecast/revenue?horizon=1&{q}", headers=self.h)
            self.assertEqual(r.status_code, 400, q)

    def test_security_headers_and_json_404(self):
        r = self.c.get("/api/nope", headers=self.h)
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(r.headers["Cache-Control"], "no-store")

    def test_lockout_after_repeated_bad_keys(self):
        bad = {"X-API-Key": "wrong"}
        codes = [self.c.get("/api/categories", headers=bad).status_code for _ in range(5)]
        self.assertEqual(codes[:3], [401, 401, 401])
        self.assertEqual(codes[3:], [429, 429])
        # locked out even with the right key until the window passes
        self.assertEqual(self.c.get("/api/categories", headers=self.h).status_code, 429)
        self.assertEqual(self.c.get("/health").status_code, 200)

    def test_reload_needs_admin_key(self):
        self.assertEqual(self.c.post("/api/reload", headers=self.h).status_code, 401)
        ok = self.c.post("/api/reload", headers={**self.h, "X-Admin-Key": "admin-key"})
        self.assertEqual(ok.status_code, 200)

    def test_non_ascii_key_does_not_crash(self):
        r = self.c.get("/api/categories", headers={"X-API-Key": "kéy"})
        self.assertEqual(r.status_code, 401)


class Pipeline(unittest.TestCase):
    def test_product_history_only_appends_after_last_month(self):
        import pandas as pd
        import rebuild_features_from_db as r
        existing = pd.DataFrame([{"Month": "2026-08", "MSKU": "1", "Product Name": "x", "Category": "MEMORY",
                                  "units_sold": 1.0, "revenue": 10.0, "returns_units": 0.0, "returns_amount": 0.0,
                                  "net_revenue": 10.0, "avg_price": 10.0}])
        live = pd.DataFrame({"month": pd.PeriodIndex(["2026-08", "2026-09", "2026-09"], freq="M"),
                             "msku": ["1", "1", None], "product_name": ["x", "x", "y"],
                             "category": ["MEMORY"] * 3, "units_sold": [5.0, 2.0, 3.0], "revenue": [50.0, 20.0, 30.0]})
        out = r.merge_product_history(existing, live)
        self.assertEqual(sorted(out["Month"]), ["2026-08", "2026-09"])   # Aug kept as-is, null-SKU row dropped
        self.assertEqual(float(out[out.Month == "2026-08"].units_sold.iloc[0]), 1.0)

    def test_sales_query_counts_only_closed_months_and_valid_statuses(self):
        import rebuild_features_from_db as r
        sql = r._sales_lines_sql("month", "month")
        self.assertIn("month < date_trunc('month', now() AT TIME ZONE 'Asia/Manila')", sql)
        self.assertIn("pos_sale_items", sql)
        self.assertIn("s.status = 'completed'", sql)
        self.assertNotIn("cancelled", r.COUNTED_ORDER_STATUSES)
        self.assertNotIn("to_pay", r.COUNTED_ORDER_STATUSES)


if __name__ == "__main__":
    unittest.main()
