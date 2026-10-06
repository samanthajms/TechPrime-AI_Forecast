"""
build_product_history.py
Builds preprocessed_monthly_product_features.csv (one row per product per month, WITH product name)
from the legacy POS export (For-sample.xlsx) or, later, from any dataframe with the same columns.

Rules (edit here, not in the model code):
  * Non-product rows are dropped: categories Customer Advances / All / CUSTOMIZATION, "Customer Down Payment".
  * Returns (Quantity <= 0) are NOT mixed into demand: they go to returns_units / returns_amount.
  * Timestamps are grouped by calendar month (Asia/Manila local time in the export).
Usage:  python build_product_history.py For-sample.xlsx
"""
import sys
import pandas as pd

EXCLUDE_CATEGORIES = {"CUSTOMER ADVANCES", "ALL", "CUSTOMIZATION"}
EXCLUDE_PRODUCTS = {"CUSTOMER DOWN PAYMENT"}

def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Category"] = df["Category"].astype(str).str.strip().str.upper()
    df["Product Name"] = df["Product Name"].astype(str).str.strip()
    df = df[~df["Category"].isin(EXCLUDE_CATEGORIES)]
    df = df[~df["Product Name"].str.upper().isin(EXCLUDE_PRODUCTS)]
    df["Month"] = pd.to_datetime(df["POS Order Date"]).dt.to_period("M").astype(str)
    df["MSKU"] = df["MSKU"].astype(str)
    sales, rets = df[df["Quantity"] > 0], df[df["Quantity"] <= 0]
    key = ["Month", "MSKU"]
    out = sales.groupby(key).agg(**{"Product Name": ("Product Name", "last"), "Category": ("Category", "last"),
                                    "units_sold": ("Quantity", "sum"),
                                    "revenue": ("Total Sales VAT Inclusive", "sum")}).reset_index()
    r = rets.groupby(key).agg(returns_units=("Quantity", lambda s: -s.sum()),
                              returns_amount=("Total Sales VAT Inclusive", lambda s: -s.sum())).reset_index()
    out = out.merge(r, on=key, how="outer")
    meta = df.sort_values("POS Order Date").groupby("MSKU")[["Product Name", "Category"]].last()
    for c in ["Product Name", "Category"]:
        out[c] = out[c].fillna(out["MSKU"].map(meta[c]))
    out[["units_sold", "revenue", "returns_units", "returns_amount"]] = out[["units_sold", "revenue", "returns_units", "returns_amount"]].fillna(0)
    out["avg_price"] = (out["revenue"] / out["units_sold"].where(out["units_sold"] > 0)).round(2)
    out["net_revenue"] = (out["revenue"] - out["returns_amount"]).round(2)
    return out[["Month", "MSKU", "Product Name", "Category", "units_sold", "revenue", "returns_units",
                "returns_amount", "net_revenue", "avg_price"]].sort_values(["Month", "Category", "MSKU"])

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "For-sample.xlsx"
    res = build(pd.read_excel(src))
    res.to_csv("preprocessed_monthly_product_features.csv", index=False)
    print(len(res), "rows,", res.MSKU.nunique(), "products,", res.Month.min(), "to", res.Month.max())
