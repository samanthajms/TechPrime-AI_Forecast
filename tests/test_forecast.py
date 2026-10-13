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
            last = f._history()["Month"].max()                       # forecast starts the month after the latest actual
            self.assertEqual(rows[0]["month"], str(__import__("pandas").Period(last, freq="M") + 1))
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


class AccuracyMetrics(unittest.TestCase):
    def test_known_values(self):
        import accuracy_metrics as am
        actual, pred = [100, 0, 50], [90, 10, 60]
        self.assertAlmostEqual(am.mae(actual, pred), 10.0)
        self.assertAlmostEqual(am.rmse(actual, pred), 10.0)
        self.assertAlmostEqual(am.wape(actual, pred), 30 / 150)             # sum|e| / sum(actual); no divide-by-zero on the 0
        self.assertAlmostEqual(am.nrmse(actual, pred), 10 / 50)
        self.assertAlmostEqual(am.bias(actual, pred), 160 / 150 - 1)

    def test_total_level_error_uses_summed_forecasts(self):
        import pandas as pd
        import accuracy_metrics as am
        log = pd.DataFrame({"Month": ["m1", "m1", "m2", "m2"], "Category": list("ABAB"),
                            "actual": [100, 100, 50, 50], "predicted": [150, 50, 60, 60]})
        tl = am.total_level_ape(log)
        self.assertAlmostEqual(tl["m1"], 0.0)                               # +50 and -50 cancel in the total
        self.assertAlmostEqual(tl["m2"], 0.2)
        s = am.summarize(log)
        self.assertEqual(s["total_level_mape_pct"], 10.0)
        self.assertGreater(s["wape_pct"], s["total_level_mape_pct"])        # category errors are larger than the total's

    def test_wape_empty_actuals_is_nan_not_error(self):
        import math
        import accuracy_metrics as am
        self.assertTrue(math.isnan(am.wape([0, 0], [1, 2])))


class FeatureBuilder(unittest.TestCase):
    def _grid(self, units):
        import pandas as pd
        n = len(units)
        return pd.DataFrame({"category": ["Memory"] * n, "month": pd.period_range("2025-01", periods=n, freq="M"),
                             "units_sold": units, "revenue": [u * 100.0 for u in units], "avg_price": [100.0] * n})

    def test_rolling_features_never_use_the_month_being_predicted(self):
        import rebuild_features_from_db as r
        base = r.add_features(self._grid([10, 20, 30, 40, 50, 60, 70, 80]))
        spiked = r.add_features(self._grid([10, 20, 30, 40, 50, 60, 70, 99999]))   # only the LAST (predicted) month changes
        cols = ["lag_1", "lag_2", "lag_3", "lag_12", "roll_mean_3", "roll_mean_6", "roll_std_3",
                "revenue_lag_1", "revenue_roll_mean_3", "revenue_roll_mean_6", "avg_price_roll_mean_3"]
        self.assertEqual(base.iloc[-1][cols].tolist(), spiked.iloc[-1][cols].tolist())
        self.assertEqual(base.iloc[-1]["roll_mean_3"], (50 + 60 + 70) / 3)

    def test_training_features_match_inference_features(self):
        import pandas as pd
        import inventory_forecasting as f
        import rebuild_features_from_db as r
        units = [12, 7, 30, 18, 25, 9, 41, 22, 15]
        feats = r.add_features(self._grid(units)).iloc[-1]                  # row for the 9th month
        series = self._grid(units[:-1]).rename(columns={"month": "Month"})  # what inference sees: months BEFORE it
        live, _ = f._build_demand_features(series)
        for k in ("lag_1", "lag_2", "lag_3", "lag_12", "roll_mean_3", "roll_mean_6", "roll_std_3"):
            self.assertAlmostEqual(float(feats[k]), float(live[k]), places=6, msg=k)

    def test_no_synthetic_all_category(self):
        import rebuild_features_from_db as r
        grid = r.build_full_grid(self._grid([1, 2, 3]).assign(category="Memory"))
        self.assertEqual(set(grid["category"]), {"Memory"})


class SystemCategories(unittest.TestCase):
    """The history, code map and forecasts all use the system's own category labels (category_alignment.taxonomy())."""

    def test_csv_and_code_map_use_only_system_labels(self):
        import pandas as pd
        import category_alignment as ca
        labels = {l for ls in ca.taxonomy()["groups"].values() for l in ls}
        for name in ("preprocessed_monthly_features.csv", "preprocessed_monthly_product_features.csv"):
            cats = set(pd.read_csv(ROOT / name)["Category"])
            self.assertFalse(cats - labels, f"{name}: not system labels: {sorted(cats - labels)}")
        import json
        with open(ROOT / "category_code_map.json") as fh:
            self.assertEqual(set(json.load(fh)), labels)

    def test_group_column_matches_taxonomy(self):
        import pandas as pd
        import category_alignment as ca
        f = pd.read_csv(ROOT / "preprocessed_monthly_features.csv")
        self.assertTrue((f["Category"].map(ca.group_of) == f["Category Group"]).all())

    def test_forecast_by_label_group_and_any_case(self):
        import inventory_forecasting as f
        a = f.get_forecast(category="Memory", horizon=2)
        self.assertEqual(len(a), 2)
        self.assertEqual(f.get_forecast(category="MEMORY", horizon=2), a)       # case-insensitive
        self.assertEqual(f.get_forecast(category="model:Memory", horizon=1)[0]["month"], a[0]["month"])
        grp = f.get_forecast(category="Component", horizon=1)                   # a whole category group
        self.assertGreater(grp[0]["predicted_units"], a[0]["predicted_units"])

    def test_demand_screen_has_total_that_matches_category_sum(self):
        import inventory_forecasting as f
        d = f.get_product_demand(horizon=2, top=100000)
        self.assertEqual(len(d["total"]), 2)
        total_units = sum(m["predicted_units"] for m in d["total"])
        self.assertAlmostEqual(total_units, sum(p["forecast_units"] for p in d["products"]), delta=total_units * 0.01)
        mem = f.get_product_demand(horizon=1, category="Memory", top=500)
        self.assertAlmostEqual(mem["total"][0]["predicted_units"], f.get_forecast(category="Memory", horizon=1)[0]["predicted_units"], places=0)

    def test_metrics_endpoint_has_study_metrics(self):
        import inventory_forecasting as f
        m = f.get_model_metrics()
        for target in ("demand", "revenue"):
            for k in ("mae", "rmse", "wape_pct", "nrmse_pct", "bias_pct", "total_level_mape_pct",
                      "naive_last_month_wape_pct", "moving_avg_6m_wape_pct"):
                self.assertIn(k, m[target], f"{target}.{k}")


if __name__ == "__main__":
    unittest.main()
