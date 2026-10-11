"""
month_guard.py
===============
One definition of "the month is closed" (Asia/Manila calendar, same as the SQL in rebuild_features_from_db.py), shared by
the data refresh, the retrain and the forecast service. A month that is still in progress has only part of its sales, so
it must never be a training row nor the "latest actual month" a forecast starts from.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

SHOP_TZ = ZoneInfo('Asia/Manila')


def current_month(now=None) -> pd.Period:
    """The month in progress right now in Manila (or at `now`, for tests)."""
    now = now or datetime.now(SHOP_TZ)
    return pd.Period(now.astimezone(SHOP_TZ).replace(tzinfo=None), freq='M')


def drop_open_months(df: pd.DataFrame, col: str = 'Month', now=None) -> pd.DataFrame:
    """Rows whose month is the current (unfinished) month or later are removed. `col` may hold 'YYYY-MM' strings or Periods."""
    if df.empty:
        return df
    closed = pd.PeriodIndex(df[col].astype(str), freq='M') < current_month(now)
    return df[closed]
