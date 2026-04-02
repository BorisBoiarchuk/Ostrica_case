import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import *

# =========================================================
# EXECUTION HELPERS
# =========================================================

def execute_targets(group: pd.DataFrame) -> pd.DataFrame:
    lot = group["lot_size"].to_numpy(dtype=np.int64)
    target = group["target_shares"].to_numpy(dtype=float)
    volume = group["volume"].to_numpy(dtype=float)
    part = group["max_participation_rate"].to_numpy(dtype=float)
    abnormal = group["is_abnormal_zone"].to_numpy(dtype=bool)
    event = group["is_event_window"].to_numpy(dtype=bool)
    locate = group["locate_limit_shares"].to_numpy(dtype=float)

    n = len(group)
    shares = np.zeros(n, dtype=float)
    prev_shares = np.zeros(n, dtype=float)
    shares_traded = np.zeros(n, dtype=float)
    participation_used = np.zeros(n, dtype=float)
    forced_exit_flag = np.zeros(n, dtype=bool)
    participation_cap_shares = np.zeros(n, dtype=float)

    current = 0.0

    for i in range(n):
        li = max(int(lot[i]), 1)
        max_trade = volume[i] * part[i]
        max_trade = np.floor(max_trade / li) * li
        max_trade = max(max_trade, 0.0)
        participation_cap_shares[i] = max_trade

        desired_change = target[i] - current

        if FORCE_EXIT_IGNORES_PARTICIPATION and (event[i] or abnormal[i]):
            executed_change = -current
            forced_exit_flag[i] = current != 0
        else:
            executed_change = np.clip(desired_change, -max_trade, max_trade)

        executed_change = np.sign(executed_change) * (np.floor(np.abs(executed_change) / li) * li)

        prev = current
        current = current + executed_change

        if current < 0:
            current = -min(abs(current), locate[i])
            current = -(np.floor(abs(current) / li) * li)

        prev_shares[i] = prev
        shares[i] = current
        shares_traded[i] = abs(current - prev)
        if max_trade > 0:
            participation_used[i] = abs(current - prev) / max_trade
        else:
            participation_used[i] = 0.0

    out = group.copy()
    out["prev_shares"] = prev_shares
    out["shares"] = shares
    out["shares_traded"] = shares_traded
    out["participation_cap_shares"] = participation_cap_shares
    out["participation_used"] = participation_used
    out["forced_exit_flag"] = forced_exit_flag
    return out