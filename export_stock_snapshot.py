"""
export_stock_snapshot.py
========================
Writes current_stock.csv (the stock-on-hand snapshot read by inventory_planning.py) from the live database.

    python export_stock_snapshot.py                 # writes current_stock.csv next to this file
    python export_stock_snapshot.py --out /path/x.csv
    python export_stock_snapshot.py --dry-run       # print the first rows, write nothing

The query is NOT verified against your schema: the forecast job's database role has never needed stock, and the default
below ASSUMES products.sku (the MSKU used in the sales history) and products.stock. Check the real column name, then either
edit DEFAULT_SQL or set STOCK_SQL. The query must return the columns  MSKU, stock_on_hand  and may return  on_order,
lead_time_days.  The role also needs SELECT on the table it reads (see docs/RENDER_SECURITY_GUIDE.md section 6).
Same DB_* settings and read-only session as rebuild_features_from_db.py.
"""
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / '.env')

DEFAULT_SQL = 'SELECT p.sku AS "MSKU", p.stock AS stock_on_hand FROM products p'   # ASSUMED column names: verify!


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(BASE_DIR / 'current_stock.csv'))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    cfg = {'host': os.environ.get('DB_HOST'), 'port': os.environ.get('DB_PORT'), 'dbname': os.environ.get('DB_NAME'),
           'user': os.environ.get('DB_USER'), 'password': os.environ.get('DB_PASS'), 'sslmode': 'require'}
    with psycopg2.connect(**cfg) as conn:
        conn.set_session(readonly=True)
        df = pd.read_sql(os.environ.get('STOCK_SQL', DEFAULT_SQL), conn)
    if not {'MSKU', 'stock_on_hand'} <= set(df.columns):
        sys.exit('The query must return the columns MSKU and stock_on_hand.')
    df['MSKU'] = df['MSKU'].astype(str).str.strip()
    df['snapshot_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    if args.dry_run:
        print(df.head(10).to_string(index=False)); print(len(df), 'rows (dry run, nothing written)')
        return
    tmp = Path(args.out + '.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(args.out)                                   # atomic: the service never reads a half-written file
    print(f'wrote {len(df)} rows to {args.out}')


if __name__ == '__main__':
    main()
