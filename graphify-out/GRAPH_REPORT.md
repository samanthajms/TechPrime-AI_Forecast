# Graph Report - TechPrime-AI_Forecast  (2026-10-29)

## Corpus Check
- Corpus is ~21,978 words - fits in a single context window. You may not need a graph.

## Summary
- 368 nodes · 695 edges · 21 communities (10 shown, 11 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 19 edges (avg confidence: 0.87)
- Token cost: 100,034 input · 0 output

## Community Hubs (Navigation)
- Demand Forecasting Engine
- Deployment Security & Secrets
- Flask Forecast API
- DB Data Refresh Scripts
- Retrain & Accuracy Gate
- ABC Classes & Accuracy Metrics
- Inventory Planning & Stock
- Category Taxonomy Alignment
- API Security Tests
- Test Harness & Split
- Inventory Action Tests
- Rate-Limit Sliding Window
- System Category Tests
- Feature Builder Tests
- Revenue-from-Units Tests
- Accuracy Metric Tests
- Month-over-Month Tests
- ABC Classification Tests
- Forecast Horizon Tests
- Sales Pipeline Tests
- Serena Editor Config

## God Nodes (most connected - your core abstractions)
1. `Render Security Configuration Guide` - 27 edges
2. `TechPrime AI Forecast Service README` - 23 edges
3. `_history()` - 18 edges
4. `get_revenue_forecast()` - 17 edges
5. `techprime-forecast Render Web Service` - 14 edges
6. `get_product_demand()` - 13 edges
7. `requirements.txt (runtime)` - 12 edges
8. `get_forecast()` - 11 edges
9. `get_inventory_actions()` - 11 edges
10. `main()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `1-3 Month Forecast Window` --implements--> `check_horizon()`  [INFERRED]
  README.md → inventory_forecasting.py
- `Render Security Configuration Guide` --references--> `check_horizon()`  [EXTRACTED]
  docs/RENDER_SECURITY_GUIDE.md → inventory_forecasting.py
- `TechPrime AI Forecast Service README` --references--> `expected_unit_price()`  [EXTRACTED]
  README.md → inventory_forecasting.py
- `Revenue = Predicted Units x Expected Unit Price` --references--> `expected_unit_price()`  [EXTRACTED]
  README.md → inventory_forecasting.py
- `psycopg2-binary (retrain)` --semantically_similar_to--> `psycopg2-binary`  [INFERRED] [semantically similar]
  requirements-retrain.txt → requirements.txt

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Monthly Retrain-to-Deploy Pipeline** — _github_workflows_monthly_retrain, rebuild_features_from_db, rebuild_models_v2, readme_regression_gate, _github_workflows_monthly_retrain_retrain_pr, render_techprime_forecast [EXTRACTED 1.00]
- **Service-to-Service Trust Chain** — docs_render_security_guide_techprime_ai, docs_render_security_guide_forecast_api_php, docs_render_security_guide_forecast_api_key, render_techprime_forecast, docs_render_security_guide_rate_limit_lockout [EXTRACTED 1.00]
- **Demand and Revenue Forecast Method** — readme_xgboost_demand_model, readme_recursive_forecast, readme_product_share_allocation, readme_revenue_units_times_price, inventory_forecasting_expected_unit_price [INFERRED 0.85]

## Communities (21 total, 11 thin omitted)

### Community 0 - "Demand Forecasting Engine"
Cohesion: 0.06
Nodes (69): Turn a user-supplied category into ('label'|'group'|'model', canonical_value)., resolve_filter(), _abc(), _actual_months(), _aggregate_forecast(), _build_demand_features(), check_horizon(), compare_actual_vs_forecast() (+61 more)

### Community 1 - "Deployment Security & Secrets"
Cohesion: 0.06
Nodes (56): DB_* GitHub Actions Secrets, retrain/YYYY-MM Pull Request, taxonomy(), Render Security Configuration Guide, GitHub Branch Protection and Secret Scanning, Models Travel Through Git (Ephemeral Render Disk), FORECAST_ADMIN_KEY, FORECAST_API_KEY (+48 more)

### Community 2 - "Flask Forecast API"
Cohesion: 0.11
Nodes (38): after_request, before_request, collections, errorhandler, flask, abc_classes(), BadRequest, categories() (+30 more)

### Community 3 - "DB Data Refresh Scripts"
Cohesion: 0.11
Nodes (29): argparse, datetime, diagnose_db.py ============== Standalone sanity-check for why…, dotenv, export_stock_snapshot.py ======================== Writes current_stock.csv (the…, os, Path, pathlib (+21 more)

### Community 4 - "Retrain & Accuracy Gate"
Cohesion: 0.09
Nodes (30): Monthly Retrain Workflow (GitHub Actions), Cron 02:00 UTC on the 2nd, Dry-run Data Refresh Preview, Known Limits, Accuracy Metrics (MAE, RMSE, WAPE, NRMSE, bias, total-level error), Chronological 80/20 Train/Test Split, Likely Range (forecast +/- held-out RMSE), model_accuracy.json (+22 more)

### Community 5 - "ABC Classes & Accuracy Metrics"
Cohesion: 0.13
Nodes (21): classify(), abc_classification.py ===================== ABC (Pareto) classification of…, history: product-month frame (Month, MSKU, Product Name, Shop Category, Shop…, _arr(), bias(), mae(), nrmse(), per_category_wape() (+13 more)

### Community 6 - "Inventory Planning & Stock"
Cohesion: 0.13
Nodes (21): _clean_stock(), _load_live_stock(), load_stock(), _message(), plan_row(), _query_stock_table(), inventory_planning.py ===================== Inventory actions for the Product…, Run STOCK_SQL on the database (read-only session) and return the rows as a… (+13 more)

### Community 7 - "Category Taxonomy Alignment"
Cohesion: 0.13
Nodes (18): canonical_label(), classify_product(), group_of(), _label_index(), _model_categories(), _place(), Bridges the two category vocabularies used by TechPrime AI: MODEL categories -…, The system label spelled exactly as in the taxonomy for a case-insensitive… (+10 more)

### Community 9 - "Test Harness & Split"
Cohesion: 0.25
Nodes (5): importlib, LiveCategoryNames, Run: python -m unittest discover -s tests -v (from the repo root), Split, unittest

### Community 11 - "Rate-Limit Sliding Window"
Cohesion: 0.38
Nodes (3): Tiny in-process sliding-window counter (no extra dependency). State is per…, Record a hit and return how many hits are now inside the window., _SlidingWindow

## Knowledge Gaps
- **17 isolated node(s):** `Cron 02:00 UTC on the 2nd`, `Serena Project Config`, `Python Language Server (Serena)`, `WAPE`, `DEMAND_FEATURES (lags, rolling stats, month, quarter, category_code)` (+12 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 151 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Render Security Configuration Guide` connect `Deployment Security & Secrets` to `Demand Forecasting Engine`, `Flask Forecast API`, `DB Data Refresh Scripts`, `Retrain & Accuracy Gate`?**
  _High betweenness centrality (0.124) - this node is a cross-community bridge._
- **Why does `TechPrime AI Forecast Service README` connect `Deployment Security & Secrets` to `Demand Forecasting Engine`, `DB Data Refresh Scripts`, `Retrain & Accuracy Gate`, `ABC Classes & Accuracy Metrics`, `Inventory Planning & Stock`?**
  _High betweenness centrality (0.118) - this node is a cross-community bridge._
- **Why does `Service` connect `API Security Tests` to `Test Harness & Split`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **What connects `Cron 02:00 UTC on the 2nd`, `Serena Project Config`, `Python Language Server (Serena)` to the rest of the system?**
  _17 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Demand Forecasting Engine` be split into smaller, more focused modules?**
  _Cohesion score 0.055130784708249496 - nodes in this community are weakly interconnected._
- **Should `Deployment Security & Secrets` be split into smaller, more focused modules?**
  _Cohesion score 0.05764411027568922 - nodes in this community are weakly interconnected._
- **Should `Flask Forecast API` be split into smaller, more focused modules?**
  _Cohesion score 0.1064102564102564 - nodes in this community are weakly interconnected._