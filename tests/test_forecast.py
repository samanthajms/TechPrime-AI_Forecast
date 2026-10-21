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


class LiveCategoryNames(unittest.TestCase):
    def test_upper_cased_db_categories_are_restored_before_the_code_map_check(self):
        import pandas as pd
        import rebuild_features_from_db as r
        live = pd.DataFrame({"category": ["PROCESSOR INTEL", "MEMORY", "NOT A CATEGORY"], "units_sold": [1, 2, 3]})
        out = r.filter_known_categories(live, "category", allow_new=False)
        self.assertEqual(sorted(out["category"]), ["Memory", "Processor INTEL"])      # system spelling, unknown one skipped


class RevenueFromUnits(unittest.TestCase):
    def test_revenue_is_units_times_flat_expected_price(self):
        import inventory_forecasting as f
        rows = f.get_forecast(category="model:Memory", horizon=3)
        series = f._monthly_series("Memory")
        price = f.expected_unit_price(series)
        self.assertAlmostEqual(price, float(series["avg_price"].tail(3).mean()), places=6)     # mean of the last 3 monthly prices
        for r in rows:                                                                         # same price every step: nothing compounds
            self.assertAlmostEqual(r["predicted_revenue"], r["predicted_units"] * price, delta=price * 0.06 + 0.01)

    def test_no_revenue_model_file_or_feedback(self):
        import inventory_forecasting as f
        self.assertFalse(hasattr(f, "_revenue_model"))
        self.assertFalse((ROOT / "xgb_monthly_revenue_model.json").exists())

    def test_small_categories_stay_near_their_real_revenue(self):
        import inventory_forecasting as f
        h = f._history()
        last3 = h[h["Month"].isin(sorted(h["Month"].unique())[-3:])]
        actual = last3.groupby("Shop Category")["revenue"].sum() / 3                          # typical monthly revenue
        units = last3.groupby("Shop Category")["units_sold"].sum() / 3
        r = f.get_revenue_forecast(horizon=3)
        for row in r["by_category"]:
            base = actual.get(row["category"], 0)
            if base > 0 and units.get(row["category"], 0) >= 5:       # skip near-dormant categories: their error is the unit forecast's
                self.assertLess(row["predicted_revenue"], base * 5, row["category"])           # the old model forecast CCTV at 15x

    def test_metrics_say_how_revenue_is_made(self):
        import inventory_forecasting as f
        self.assertIn("units", f.get_model_metrics()["revenue"]["method"])


class MonthOverMonth(unittest.TestCase):
    def test_pct_and_chain(self):
        import inventory_forecasting as f
        self.assertEqual(f._pct(110, 100), 10.0)
        self.assertEqual(f._pct(50, 100), -50.0)
        self.assertIsNone(f._pct(5, 0))                                  # growth from nothing is undefined, not infinite
        rows = [{"month": "2026-10", "predicted_units": 120.0, "predicted_revenue": 1000.0},
                {"month": "2026-11", "predicted_units": 90.0, "predicted_revenue": 1100.0}]
        out = f._with_mom(rows, (100.0, 800.0))                          # first month vs latest ACTUAL, then vs previous forecast
        self.assertEqual([r["units_mom_pct"] for r in out], [20.0, -25.0])
        self.assertEqual([r["revenue_mom_pct"] for r in out], [25.0, 10.0])

    def test_revenue_forecast_growth_matches_history_and_ignores_display_range(self):
        import inventory_forecasting as f
        full = f.get_revenue_forecast(horizon=2)
        ranged = f.get_revenue_forecast(horizon=2, start="2026-01", end="2026-03")
        self.assertEqual(full["growth"], ranged["growth"])               # from/to only limits the chart history
        g = full["growth"]
        self.assertEqual(g["last_actual_month"], f._history()["Month"].max())
        self.assertAlmostEqual(g["revenue_mom_pct"], (g["last_actual_revenue"] / g["previous_revenue"] - 1) * 100, delta=0.1)
        first = full["total"][0]
        self.assertAlmostEqual(first["units_mom_pct"], (first["predicted_units"] / g["last_actual_units"] - 1) * 100, delta=0.1)
        for row in full["by_category"]:
            self.assertIn("units_mom_pct", row); self.assertIn("revenue_mom_pct", row)

    def test_demand_screen_total_has_mom(self):
        import inventory_forecasting as f
        t = f.get_product_demand(horizon=2, top=1)["total"]
        self.assertTrue(all("units_mom_pct" in m for m in t))


class AbcClassification(unittest.TestCase):
    def test_pareto_cutoffs(self):
        import pandas as pd
        import abc_classification as abc
        rev = {"a": 700, "b": 200, "c": 60, "d": 40, "zero": 0, "neg": -5}
        h = pd.DataFrame({"Month": "2026-09", "MSKU": list(rev), "Product Name": list(rev), "Shop Category": "Memory",
                          "Shop Group": "Component", "units_sold": 1, "revenue": list(rev.values())})
        t = abc.classify(h)
        self.assertEqual(t["abc_class"].to_dict(), {"a": "A", "b": "A", "c": "B", "d": "C"})   # shares before: 0 / .70 / .90 / .96
        self.assertNotIn("zero", t.index); self.assertNotIn("neg", t.index)                  # no positive revenue -> no class

    def test_real_history_split_and_demand_rows_carry_class(self):
        import inventory_forecasting as f
        a = f.get_abc(top=5)
        shares = {s["class"]: s["revenue_share_pct"] for s in a["summary"]}
        self.assertGreaterEqual(shares["A"], 80.0); self.assertAlmostEqual(sum(shares.values()), 100.0, delta=0.2)
        self.assertEqual(a["products"][0]["abc_class"], "A")
        self.assertEqual({r["abc_class"] for r in f.get_abc(abc_class="b", top=500)["products"]}, {"B"})
        with self.assertRaises(ValueError):
            f.get_abc(abc_class="Z")
        rows = f.get_product_demand(horizon=1, top=20)["products"]
        self.assertTrue(all(r["abc_class"] in ("A", "B", "C", None) for r in rows))


class InventoryActions(unittest.TestCase):
    def test_plan_row_math(self):
        import inventory_planning as ip
        # d = 2/day, sigma = 6/month, L = 7, 95%: SS = ceil(1.645 x 6 x sqrt(7/30.4)) = 5, ROP = 14 + 5 = 19, order-up-to = 19 + 60 = 79
        r = ip.plan_row(10, 0, 2, 6, 7, 0.95)
        self.assertEqual((r["safety_stock"], r["reorder_point"], r["order_up_to"]), (5, 19, 79))
        self.assertEqual((r["stockout_risk"], r["action"], r["recommended_order_qty"], r["days_of_inventory"]), ("high", "order_now", 69, 5.0))
        r = ip.plan_row(18, 0, 2, 6, 7, 0.95)                             # 9 days of cover (>= lead time) but below the ROP
        self.assertEqual((r["stockout_risk"], r["action"], r["recommended_order_qty"]), ("medium", "reorder", 61))
        r = ip.plan_row(100, 0, 2, 6, 7, 0.95)
        self.assertEqual((r["stockout_risk"], r["action"], r["recommended_order_qty"], r["overstock"]), ("low", "ok", 0, False))
        r = ip.plan_row(500, 0, 2, 6, 7, 0.95)                            # 250 days of cover
        self.assertEqual((r["action"], r["overstock"], r["recommended_order_qty"], r["excess_units"]), ("overstocked", True, 0, 320))
        r = ip.plan_row(0, 40, 2, 6, 7, 0.95)                             # nothing on hand but 40 already ordered
        self.assertEqual((r["stockout_risk"], r["action"], r["recommended_order_qty"]), ("high", "awaiting_delivery", 0))
        r = ip.plan_row(5, 0, 0, 3, 7, 0.95)                              # stock but no forecast demand
        self.assertEqual((r["action"], r["overstock"], r["days_of_inventory"], r["recommended_order_qty"]), ("no_demand", True, None, 0))
        self.assertGreater(ip.plan_row(10, 0, 2, 6, 7, 0.98)["safety_stock"], ip.plan_row(10, 0, 2, 6, 7, 0.90)["safety_stock"])

    def _stock_file(self, text):
        import tempfile
        fh = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        fh.write(text); fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_snapshot_cleaning_and_missing_file(self):
        import inventory_forecasting as f
        import inventory_planning as ip
        top = f.get_abc(top=3)["products"]
        a, b, c = (p["msku"] for p in top)
        path = self._stock_file(f"MSKU,stock_on_hand,on_order,lead_time_days,snapshot_at\n{a},10,0,,2026-10-01T00:00:00+00:00\n"
                                f"{b},-4,0,,\n{c},abc,0,,\n{a},12,3,14,\nNOPE-999,5,0,,\n")
        stock, as_of, q = ip.load_stock(path)
        self.assertEqual(q["rows_in_file"], 5); self.assertEqual(q["duplicate_msku_rows"], 1)
        self.assertEqual(q["rows_skipped_invalid"], 2)                    # negative and non-numeric stock are dropped, never guessed
        self.assertEqual(stock.loc[a, "stock_on_hand"], 12)               # the later duplicate wins
        self.assertEqual(stock.loc[a, "lead_time_days"], 14)
        with self.assertRaises(ip.StockUnavailable):
            ip.load_stock("/no/such/stock.csv")

    def test_actions_use_snapshot_lead_time_then_request_override_and_flag_staleness(self):
        import inventory_forecasting as f
        import inventory_planning as ip
        a = f.get_product_demand(horizon=1, top=1)["products"][0]["msku"]
        path = self._stock_file(f"MSKU,stock_on_hand,on_order,lead_time_days,snapshot_at\n{a},40,0,21,2020-01-01T00:00:00+00:00\nNOPE-999,5,0,,\n")
        old = ip.STOCK_PATH
        ip.STOCK_PATH = __import__("pathlib").Path(path)
        try:
            r = ip.get_inventory_actions(horizon=1)
            self.assertEqual([p["msku"] for p in r["products"]], [a])     # a stock row for an unknown MSKU is ignored
            p = r["products"][0]
            self.assertEqual((p["lead_time_days"], p["lead_time_source"]), (21, "stock snapshot"))
            self.assertTrue(r["stock_stale"])
            self.assertIn("PLACEHOLDER", r["assumptions"]["lead_time_note"])
            self.assertEqual(ip.get_inventory_actions(horizon=1, lead_time_days=3)["products"][0]["lead_time_source"], "request")
            with self.assertRaises(ValueError):
                ip.get_inventory_actions(horizon=1, lead_time_days=0)
            with self.assertRaises(ValueError):
                ip.get_inventory_actions(horizon=1, action="bogus")
        finally:
            ip.STOCK_PATH = old

    def test_live_stock_from_db_is_cached_and_survives_a_failed_refresh(self):
        import pandas as pd
        import inventory_forecasting as f
        import inventory_planning as ip
        a = f.get_product_demand(horizon=1, top=1)["products"][0]["msku"]
        calls, db_down = [], [False]

        def fake_query():
            calls.append(1)
            if db_down[0]:
                raise OSError("connection refused")
            return pd.DataFrame({"MSKU": [a, None, "X"], "stock_on_hand": [7, 3, -1]})

        saved = (ip.STOCK_SOURCE, ip._query_stock_table, dict(ip._live_cache))
        ip.STOCK_SOURCE, ip._query_stock_table = "db", fake_query
        ip._live_cache.update(data=None, next_read=0.0)
        try:
            stock, as_of, q = ip.load_stock()
            self.assertEqual(stock.loc[a, "stock_on_hand"], 7)
            self.assertEqual(q["rows_skipped_invalid"], 2)                # a NULL sku and negative stock are dropped
            ip.load_stock()
            self.assertEqual(len(calls), 1)                               # second call within the cache window: no query
            ip._live_cache["next_read"] = 0.0                             # cache expired, refresh fails -> last good read
            db_down[0] = True
            with self.assertLogs("inventory_planning", "ERROR"):
                again = ip.load_stock()
            self.assertIs(again, ip._live_cache["data"]); self.assertEqual(len(calls), 2)
            r = ip.get_inventory_actions(horizon=1)
            self.assertEqual(r["stock_source"], "database (live)"); self.assertFalse(r["stock_stale"])
            ip._live_cache.update(data=None, next_read=0.0)               # failure with nothing cached -> 503-style error
            with self.assertLogs("inventory_planning", "ERROR"), self.assertRaises(ip.StockUnavailable) as e:
                ip.load_stock()
            self.assertNotIn("refused", str(e.exception))                 # connection details never reach the caller
            self.assertGreater(ip.load_stock(str(ROOT / "current_stock.sample.csv"))[2]["rows_used"], 0)  # an explicit path still reads the file
        finally:
            ip.STOCK_SOURCE, ip._query_stock_table = saved[0], saved[1]
            ip._live_cache.clear(); ip._live_cache.update(saved[2])

    def test_sample_snapshot_and_endpoints(self):
        import forecast_service
        import inventory_planning as ip
        importlib.reload(forecast_service)
        c, h = forecast_service.app.test_client(), {"X-API-Key": "test-key"}
        old = ip.STOCK_PATH
        try:
            ip.STOCK_PATH = ROOT / "no-such-stock.csv"
            r = c.get("/api/inventory/actions", headers=h)
            self.assertEqual(r.status_code, 503); self.assertIn("stock snapshot", r.get_json()["error"])
            ip.STOCK_PATH = ROOT / "current_stock.sample.csv"
            r = c.get("/api/inventory/actions?action=order_now&top=5", headers=h)
            self.assertEqual(r.status_code, 200)
            body = r.get_json()
            self.assertTrue(all(p["action"] == "order_now" for p in body["products"]))
            self.assertEqual(sum(body["summary"].values()), body["total_products"])
            for bad in ("class=Z", "action=zzz", "lead_time_days=abc", "horizon=9"):
                self.assertEqual(c.get("/api/inventory/actions?" + bad, headers=h).status_code, 400, bad)
            self.assertEqual(c.get("/api/abc?class=A&top=3", headers=h).status_code, 200)
            self.assertEqual(c.get("/api/abc?class=Q", headers=h).status_code, 400)
            self.assertEqual(c.get("/api/inventory/actions").status_code, 401)             # still needs the API key
        finally:
            ip.STOCK_PATH = old


if __name__ == "__main__":
    unittest.main()
