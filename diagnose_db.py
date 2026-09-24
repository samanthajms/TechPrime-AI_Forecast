"""
diagnose_db.py
==============
Standalone sanity-check for why rebuild_features_from_db.py's query returned
zero rows. Run this from the TechPrime-AI_Forecast folder (same place as
rebuild_features_from_db.py) -- it reuses the same .env.

    python diagnose_db.py
"""

import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / '.env')

DB_CONFIG = {
    'host': os.environ.get('DB_HOST'),
    'port': os.environ.get('DB_PORT'),
    'dbname': os.environ.get('DB_NAME'),
    'user': os.environ.get('DB_USER'),
    'password': os.environ.get('DB_PASS'),
    'sslmode': 'require',
}

CHECKS = [
    ("Which tables actually exist in the public schema",
     "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY 1;"),

    ("Row count: orders",
     "SELECT COUNT(*) FROM orders;"),

    ("Row count: order_items",
     "SELECT COUNT(*) FROM order_items;"),

    ("Row count: products",
     "SELECT COUNT(*) FROM products;"),

    ("Products with a non-empty category",
     "SELECT COUNT(*) FROM products WHERE category IS NOT NULL AND category <> '';"),

    ("Sample of distinct product categories (first 20)",
     "SELECT DISTINCT category FROM products ORDER BY category LIMIT 20;"),

    ("Does order_items.product_id actually match any products.id",
     "SELECT COUNT(*) FROM order_items oi JOIN products p ON p.id = oi.product_id;"),

    ("Does order_items.order_id actually match any orders.id",
     "SELECT COUNT(*) FROM order_items oi JOIN orders o ON o.id = oi.order_id;"),

    ("Sample of 5 raw order_items rows (to see real column values)",
     "SELECT * FROM order_items LIMIT 5;"),
]

with psycopg2.connect(**DB_CONFIG) as conn:
    with conn.cursor() as cur:
        for label, sql in CHECKS:
            print(f"\n--- {label} ---")
            try:
                cur.execute(sql)
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description] if cur.description else []
                if cols:
                    print(cols)
                for r in rows[:20]:
                    print(r)
            except Exception as e:
                print(f"FAILED: {e}")
                conn.rollback()  # so the next check can still run
