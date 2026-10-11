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


class LiveAppend(unittest.TestCase):
    """The monthly top-up: existing rows untouched, new months appended with past-only features, open month excluded."""
    FEATS = ["lag_1", "lag_2", "lag_3", "lag_12", "roll_mean_3", "roll_mean_6", "roll_std_3", "avg_price_lag_1",
             "avg_price_roll_mean_3", "price_trend", "revenue_lag_1", "revenue_lag_12", "revenue_roll_mean_3",
             "revenue_roll_mean_6"]

    def test_features_match_the_committed_csv_convention(self):
        """next_month_features on the months before a row must reproduce that row's stored features (>=12 months in)."""
        import pandas as pd
        import rebuild_features_from_db as r
        f = pd.read_csv(ROOT / "preprocessed_monthly_features.csv")
        f = f[f.Category != "All"]
        checked = 0
        for _, g in f.groupby("Category"):
            g = g.sort_values("Month").reset_index(drop=True)
            for i in range(12, len(g)):
                got = r.next_month_features(list(g.units_sold[:i]), list(g.revenue[:i]), list(g.avg_price[:i]))
                for k in self.FEATS:
                    self.assertAlmostEqual(got[k], g.loc[i, k], delta=0.02 + abs(g.loc[i, k]) * 1e-3, msg=f"{k} {g.Category[i]} {g.Month[i]}")
                checked += 1
        self.assertGreater(checked, 300)

    def _existing(self):
        import pandas as pd
        import rebuild_features_from_db as r
        return pd.read_csv(ROOT / "preprocessed_monthly_features.csv")[r.OUTPUT_COLUMNS]

    def test_append_keeps_old_rows_and_only_adds_later_months(self):
        import pandas as pd
        import rebuild_features_from_db as r
        existing = self._existing()
        last = existing[existing.Category != "All"].Month.max()
        live = pd.DataFrame({"category": ["MEMORY", "MEMORY", "MOUSE"],
                             "month": pd.PeriodIndex([last, "2026-09", "2026-09"], freq="M"),
                             "units_sold": [999.0, 10.0, 4.0], "revenue": [999.0, 1000.0, 400.0]})
        new = r.append_live_months(existing, live)
        self.assertEqual(set(new.Month), {"2026-09"})                             # the clashing old month is not re-added
        self.assertEqual(set(new.Category), set(existing[existing.Category != "All"].Category))   # zeros for the rest
        mem = new[new.Category == "MEMORY"].iloc[0]
        self.assertEqual((mem.units_sold, mem.revenue), (10, 1000.0))
        prev = existing[(existing.Category == "MEMORY")].sort_values("Month")
        self.assertEqual(mem.lag_1, prev.units_sold.iloc[-1])                      # lags come from the real last month
        self.assertEqual(new[new.Category == "ACCESSORIES"].iloc[0].units_sold, 0)

    def test_features_ignore_the_month_they_describe(self):
        import pandas as pd
        import rebuild_features_from_db as r
        existing = self._existing()
        def run(units):
            live = pd.DataFrame({"category": ["MEMORY"], "month": pd.PeriodIndex(["2026-09"], freq="M"),
                                 "units_sold": [units], "revenue": [units * 100.0]})
            return r.append_live_months(existing, live).query("Category == 'MEMORY'").iloc[0]
        a, b = run(1.0), run(500.0)
        for k in self.FEATS:
            self.assertEqual(a[k], b[k], k)                                       # no leakage of the month's own sales

    def test_open_month_is_dropped_after_the_append(self):
        import pandas as pd
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from month_guard import drop_open_months
        now = datetime(2026, 10, 11, 9, 0, tzinfo=ZoneInfo("Asia/Manila"))
        df = pd.DataFrame({"Month": ["2026-08", "2026-09", "2026-10", "2026-11"], "v": [1, 2, 3, 4]})
        self.assertEqual(list(drop_open_months(df, now=now).Month), ["2026-08", "2026-09"])
        # 2026-09-30 17:00 UTC is already 2026-10-01 in Manila -> September is closed there
        edge = datetime(2026, 9, 30, 17, 0, tzinfo=ZoneInfo("UTC"))
        self.assertEqual(list(drop_open_months(df, now=edge).Month), ["2026-08", "2026-09"])


class TrainingFrame(unittest.TestCase):
    def test_open_month_junk_category_and_short_history_are_not_trained_on(self):
        import pandas as pd
        from datetime import datetime
        from zoneinfo import ZoneInfo
        import rebuild_models_v2 as m
        months = [str(p) for p in pd.period_range("2025-01", "2026-10", freq="M")]
        rows = [{"Month": mo, "Category": "MEMORY"} for mo in months]
        rows += [{"Month": mo, "Category": "All"} for mo in months]
        rows += [{"Month": mo, "Category": "NEWCAT"} for mo in months[3:]]          # starts mid-file
        now = datetime(2026, 10, 11, tzinfo=ZoneInfo("Asia/Manila"))
        out = m.prepare_training_frame(pd.DataFrame(rows), now=now)
        self.assertNotIn("All", set(out.Category))
        self.assertEqual(str(out.Month.max()), "2026-09")                            # October is in progress
        self.assertEqual(len(out[out.Category == "MEMORY"]), 21)                     # present from the file's first month: all closed months kept
        self.assertEqual(str(out[out.Category == "NEWCAT"].Month.min()), "2025-10")  # began 2025-04: its 6 zero-padded months are skipped


if __name__ == "__main__":
    unittest.main()
