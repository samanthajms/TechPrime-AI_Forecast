"""Tests for the fine-tuning changes. Run: python -m pytest tests   (from the repo root; no database needed)."""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FORECAST_API_KEY", "test-key")


class Retrain(unittest.TestCase):
    def test_finished_orders_are_counted(self):
        import rebuild_features_from_db as r
        # orders.status CHECK: to_pay | to_ship | to_receive | to_review | cancelled  (to_review = received = completed sale)
        for ok in ("to_ship", "to_receive", "to_review"):
            self.assertIn(ok, r.COUNTED_ORDER_STATUSES)
        for bad in ("to_pay", "cancelled"):
            self.assertNotIn(bad, r.COUNTED_ORDER_STATUSES)

    def test_rolling_features_use_previous_months_only(self):
        import pandas as pd
        import rebuild_features_from_db as r
        units = [10, 20, 30, 40, 50, 60, 70, 80]
        full = pd.DataFrame({"category": "MEMORY", "month": pd.period_range("2025-01", periods=8, freq="M"),
                             "units_sold": units, "revenue": [u * 100.0 for u in units], "avg_price": 100.0})
        out = r.add_features(full).sort_values("Month").reset_index(drop=True)
        row = out.iloc[5]                                    # month 6 (units 60): previous three are 50, 40, 30
        self.assertEqual(row["lag_1"], 50)
        self.assertAlmostEqual(row["roll_mean_3"], 40.0)      # NOT (60+50+40)/3 -- that would include the target
        self.assertAlmostEqual(row["roll_mean_6"], (50 + 40 + 30 + 20 + 10 + 0) / 6)
        self.assertAlmostEqual(row["revenue_roll_mean_3"], 4000.0)

    def test_products_map_to_model_buckets(self):
        import rebuild_features_from_db as r
        known = {"DISPLAY", "MEMORY", "COOLING", "LAPTOP PR2", "AUDIO"}
        lookup = {"900": "LAPTOP PR2"}
        self.assertEqual(r.resolve_bucket("900", "Laptops", known, lookup), "LAPTOP PR2")     # SKU history wins
        self.assertEqual(r.resolve_bucket("1", "Memory", known, {}), "MEMORY")                # label == bucket
        self.assertEqual(r.resolve_bucket("2", "Monitor", known, {}), "DISPLAY")              # label map
        self.assertEqual(r.resolve_bucket("3", "Display", known, {}), "DISPLAY")              # legacy value
        self.assertEqual(r.resolve_bucket("4", "CPU Cooling", known, {}), "COOLING")
        self.assertIsNone(r.resolve_bucket("5", "Operating System", known, {}))               # no bucket -> reported, not guessed
        self.assertIsNone(r.resolve_bucket("6", "Laptops", known, {}))                        # 7 buckets: needs SKU history


class ZeroRule(unittest.TestCase):
    def setUp(self):
        for k in ("ZERO_RULE_MONTHS", "SPARSE_METHOD", "SPARSE_UNITS_PER_MONTH"):
            os.environ.pop(k, None)

    def test_method_selection(self):
        import intermittent as i
        self.assertEqual(i.method_for([5, 0, 0, 0]), "zero_rule")           # last 3 months all zero
        self.assertEqual(i.method_for([5, 4, 0, 0]), "xgboost")             # only 2 -> model still runs
        self.assertEqual(i.method_for([0, 0]), "xgboost")                   # fewer than N months of history: rule needs N
        os.environ["ZERO_RULE_MONTHS"] = "0"
        self.assertEqual(i.method_for([5, 0, 0, 0]), "xgboost")             # switched off
        os.environ.pop("ZERO_RULE_MONTHS")
        os.environ["SPARSE_METHOD"] = "croston"
        self.assertEqual(i.method_for([0, 1, 0, 2, 0, 1]), "croston")
        self.assertEqual(i.method_for([200, 210, 190, 205]), "xgboost")     # dense stays on XGBoost

    def test_croston_is_zero_for_no_sales_and_positive_otherwise(self):
        import intermittent as i
        self.assertEqual(i.croston_sba([0, 0, 0, 0]), 0.0)
        self.assertGreater(i.croston_sba([0, 2, 0, 0, 3, 0]), 0.0)

    def test_dead_category_forecasts_zero_and_live_one_does_not(self):
        import inventory_forecasting as f
        for row in f.get_forecast(category="model:LAPTOP NO2", horizon=3):     # no sales for 24 months
            self.assertEqual((row["predicted_units"], row["predicted_revenue"]), (0.0, 0.0))
        self.assertGreater(f.get_forecast(category="model:MEMORY", horizon=1)[0]["predicted_units"], 0)


class ForecastTable(unittest.TestCase):
    def test_labels_roll_up_to_groups_and_total(self):
        import inventory_forecasting as f
        t = f.get_forecast_table(horizon=3)
        self.assertEqual(len(t["rows"]), 39)                                    # every system category is listed
        self.assertEqual(t["months"], sorted(t["months"]))
        for g in t["groups"]:
            labels = sum(r["forecast_units_total"] or 0 for r in t["rows"] if r["group"] == g["group"])
            self.assertAlmostEqual(labels, g["forecast_units_total"], delta=0.6)
        self.assertAlmostEqual(sum(x["predicted_units"] for x in t["total"]),
                               sum(g["forecast_units_total"] for g in t["groups"]), delta=1.0)
        empty = [r for r in t["rows"] if r["status"] != "forecast"]
        self.assertTrue(empty and all(r["forecast"] == [] for r in empty))

    def test_filters_and_http_route(self):
        import importlib
        import forecast_service
        importlib.reload(forecast_service)
        c = forecast_service.app.test_client()
        h = {"X-API-Key": os.environ["FORECAST_API_KEY"]}
        r = c.get("/api/forecast/table?horizon=1&category=Component", headers=h)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(all(x["group"] == "Component" for x in r.get_json()["rows"]))
        self.assertEqual(c.get("/api/forecast/table?horizon=4", headers=h).status_code, 400)
        self.assertEqual(c.get("/api/forecast/table").status_code, 401)


if __name__ == "__main__":
    unittest.main()
