import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import *

# =========================================================
# RISK / PERFORMANCE DIAGNOSTICS HELPERS
# =========================================================
def max_drawdown(cum_pnl: pd.Series):
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    return float(drawdown.min()), drawdown



def annualized_sharpe(pnl_series: pd.Series):
    pnl = pnl_series.dropna()
    if len(pnl) < 2 or pnl.std(ddof=0) == 0:
        return np.nan
    return float(np.sqrt(ANNUALIZATION_MINUTES) * pnl.mean() / pnl.std(ddof=0))



def summarize_side(bt: pd.DataFrame, side_value: int, label: str):
    side = bt.loc[bt["shares"].apply(np.sign) == side_value].copy()
    return {
        f"{label}_gross_pnl": float(side["gross_pnl"].sum()),
        f"{label}_net_pnl": float(side["net_pnl"].sum()),
        f"{label}_avg_gross_notional": float(side["gross_abs_notional"].mean()) if not side.empty else 0.0,
        f"{label}_share_minutes": int((side["shares"] != 0).sum()),
    }