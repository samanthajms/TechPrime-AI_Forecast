"""
abc_classification.py
=====================
ABC (Pareto) classification of products by sales value, used by the Product Demand screen and the inventory actions.

  A  the products that together make up the first 80% of revenue          (few items, most of the money: forecast and
                                                                           stock them most carefully)
  B  the next 15% (cumulative 80% - 95%)
  C  the remaining 5% (the long tail)

Revenue is net of returns over the last `window` months of the product history. A product that crosses a cut-off stays
in the higher class. Products with no positive net revenue in the window get no class (None).
"""
import pandas as pd

A_CUT, B_CUT, WINDOW_MONTHS = 0.80, 0.95, 12


def classify(history, window=WINDOW_MONTHS, a_cut=A_CUT, b_cut=B_CUT):
    """history: product-month frame (Month, MSKU, Product Name, Shop Category, Shop Group, units_sold, revenue, net_revenue).
    Returns a DataFrame indexed by MSKU: product_name, category, group, units, revenue, revenue_share, cumulative_share,
    abc_class (best-selling first)."""
    months = sorted(history['Month'].unique())[-window:]
    h = history[history['Month'].isin(months)]
    value = 'net_revenue' if 'net_revenue' in h.columns else 'revenue'
    last = h.sort_values('Month').groupby('MSKU')[['Product Name', 'Shop Category', 'Shop Group']].last()
    g = h.groupby('MSKU').agg(units=('units_sold', 'sum'), revenue=(value, 'sum'))
    g = g[g['revenue'] > 0].sort_values('revenue', ascending=False)
    total = g['revenue'].sum()
    g['revenue_share'] = g['revenue'] / total if total > 0 else 0.0
    g['cumulative_share'] = g['revenue_share'].cumsum()
    before = g['cumulative_share'] - g['revenue_share']              # share already covered by better sellers
    g['abc_class'] = pd.Series(
        ['A' if b < a_cut else 'B' if b < b_cut else 'C' for b in before], index=g.index)
    g['product_name'] = last['Product Name']
    g['category'] = last['Shop Category']
    g['group'] = last['Shop Group']
    return g[['product_name', 'category', 'group', 'units', 'revenue', 'revenue_share', 'cumulative_share', 'abc_class']]
