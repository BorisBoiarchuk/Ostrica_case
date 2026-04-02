
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import *
from src.validation import *
from src.execution import *
from src.diagnostics import *

# =========================================================
# REVIEWER NOTES
# =========================================================
# This script is written for a reproducible case-study submission.
# Design goals:
#   1) validate the supplied synthetic data carefully,
#   2) implement a transparent, stock-specific intraday mean-reversion signal,
#   3) simulate constrained execution and transaction costs,
#   4) evaluate both raw and hedged performance with risk diagnostics,
#   5) save reproducible outputs to disk.
#
# Signal rationale:
#   The rolling, per-ticker thresholds, such that
#   the decision rule adapts to each name's
#   own price level and recent dispersion.
#
#   Long entry candidate: ask_t <= mean(mid_{t-100:t-1}) - 1.0 * std(mid_{t-100:t-1})
#   Short entry candidate: bid_t >= mean(mid_{t-100:t-1}) + 1.0 * std(mid_{t-100:t-1})
#
#   "Abnormal" zones are treated as forced-flat / do-not-initiate regions:
#   abnormal low:  ask_t <= mean - 1.5 * std
#   abnormal high: bid_t >= mean + 2.0 * std
#
# Information timing:
#   All rolling statistics are shifted by one bar to avoid look-ahead bias.
#   A minute-t decision uses only information available through t-1 for the
#   rolling thresholds and the minute-t quote/volume for execution sizing.
#
# Hedge rationale:
#   The hedge uses index-weighted dollar exposure from supplied constituent
#   weights. This is not a true beta hedge; it is an exposure-proxy hedge that
#   is unit-consistent with the data provided in the case.
# =========================================================


# =========================================================
# LOAD DATA
# =========================================================
intraday_prices = pd.read_csv("data/synthetic_intraday_prices_500tickers_v2.csv")
ticker_metadata = pd.read_csv("data/ticker_metadata_500tickers_v2.csv")
index_constituents = pd.read_csv("data/index_constituents_500tickers_v2.csv")
scheduled_events = pd.read_csv("data/scheduled_events_calendar_500tickers_v2.csv")
fee_schedule = pd.read_csv("data/fee_schedule_500tickers_v2.csv")

intraday_prices["timestamp"] = pd.to_datetime(intraday_prices["timestamp"], errors="coerce")
scheduled_events["timestamp"] = pd.to_datetime(scheduled_events["timestamp"], errors="coerce")


# =========================================================
# RUN VALIDATION / CLEANING
# =========================================================
validation_report = build_validation_report(
    intraday_prices=intraday_prices,
    ticker_metadata=ticker_metadata,
    index_constituents=index_constituents,
    scheduled_events=scheduled_events,
    fee_schedule=fee_schedule,
)
validation_report.to_csv("outputs/validation_report.csv", index=False)

intraday_prices, ticker_metadata, index_constituents, scheduled_events, fee_schedule, duplicate_summary = clean_validated_inputs(
    intraday_prices=intraday_prices,
    ticker_metadata=ticker_metadata,
    index_constituents=index_constituents,
    scheduled_events=scheduled_events,
    fee_schedule=fee_schedule,
)

if PRINT_VALIDATION_SUMMARY:
    print_validation_summary(validation_report, duplicate_summary)

# =========================================================
# FEES
# =========================================================
fee_map = dict(zip(fee_schedule["fee_component"], fee_schedule["value"]))
COMMISSION_BPS = float(fee_map["commission_bps_notional"])
EXCHANGE_BPS = float(fee_map["exchange_fee_bps_notional"])
SEC_SELL_BPS = float(fee_map["sec_fee_bps_sell_notional"])
LONG_FINANCING_BPS_ANNUAL = float(fee_map["financing_bps_annual_long_cash"])
LONG_FIN_PER_MIN = (LONG_FINANCING_BPS_ANNUAL / 10000) / MINUTES_PER_YEAR

# =========================================================
# SPLIT STOCKS / INDEX
# =========================================================
index_df = intraday_prices.loc[intraday_prices["ticker"] == "INDEX"].copy()
stocks_df = intraday_prices.loc[intraday_prices["ticker"] != "INDEX"].copy()

if index_df.empty:
    raise ValueError("INDEX ticker not found in intraday_prices after cleaning")

for df in (stocks_df, index_df):
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]

if (stocks_df["spread"] < 0).any() or (index_df["spread"] < 0).any():
    raise ValueError("Negative spread remains after cleaning")

# =========================================================
# MERGE METADATA
# =========================================================
stocks_df = stocks_df.merge(ticker_metadata, on="ticker", how="left", validate="many_to_one")
stocks_df = stocks_df.merge(index_constituents, on="ticker", how="left", validate="many_to_one")

missing_meta_after_merge = stocks_df["sector"].isna().sum()
if missing_meta_after_merge > 0:
    raise ValueError(f"{missing_meta_after_merge} stock rows still missing metadata after merge")

stocks_df["index_weight"] = stocks_df["index_weight"].fillna(0.0)
stocks_df["borrow_fee_bps_annual"] = stocks_df["borrow_fee_bps_annual"].fillna(0.0)
stocks_df["borrow_per_min"] = (stocks_df["borrow_fee_bps_annual"] / 10000) / MINUTES_PER_YEAR
stocks_df["lot_size"] = stocks_df["lot_size"].fillna(1).astype(int)
stocks_df["locate_limit_shares"] = stocks_df["locate_limit_shares"].fillna(0.0)
stocks_df["max_participation_rate"] = stocks_df["max_participation_rate"].fillna(0.0)

# =========================================================
# EVENT WINDOWS
# =========================================================
event_minutes = pd.DataFrame({"event_time": scheduled_events["timestamp"].drop_duplicates()})
event_offsets = pd.DataFrame({"offset_min": np.arange(-EVENT_WINDOW_MINUTES, EVENT_WINDOW_MINUTES + 1, dtype=int)})
event_grid = event_minutes.merge(event_offsets, how="cross")
event_grid["timestamp"] = event_grid["event_time"] + pd.to_timedelta(event_grid["offset_min"], unit="m")
event_flags = event_grid[["timestamp"]].drop_duplicates()
event_flags["is_event_window"] = True

stocks_df = stocks_df.merge(event_flags, on="timestamp", how="left")
index_df = index_df.merge(event_flags, on="timestamp", how="left")
stocks_df["is_event_window"] = stocks_df["is_event_window"].fillna(False).astype(bool)
index_df["is_event_window"] = index_df["is_event_window"].fillna(False).astype(bool)

# =========================================================
# SIGNAL REGIME FLAGS (ROLLING MEAN REVERSION)
# =========================================================
stocks_df = stocks_df.sort_values(["ticker", "timestamp"]).copy()

stocks_df["rolling_mean"] = (
    stocks_df.groupby("ticker", sort=False)["mid"]
    .transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean().shift(1))
)
stocks_df["rolling_std"] = (
    stocks_df.groupby("ticker", sort=False)["mid"]
    .transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std(ddof=0).shift(1))
)

stocks_df["buy_threshold"] = stocks_df["rolling_mean"] - BUY_Z * stocks_df["rolling_std"]
stocks_df["short_threshold"] = stocks_df["rolling_mean"] + SHORT_Z * stocks_df["rolling_std"]
stocks_df["abnormal_low_threshold"] = stocks_df["rolling_mean"] - ABNORMAL_LOW_Z * stocks_df["rolling_std"]
stocks_df["abnormal_high_threshold"] = stocks_df["rolling_mean"] + ABNORMAL_HIGH_Z * stocks_df["rolling_std"]

valid_stats = (
    stocks_df["rolling_mean"].notna()
    & stocks_df["rolling_std"].notna()
    & (stocks_df["rolling_std"] > 0)
)

stocks_df["is_abnormal_zone"] = False
stocks_df.loc[valid_stats, "is_abnormal_zone"] = (
    (stocks_df.loc[valid_stats, "ask"] <= stocks_df.loc[valid_stats, "abnormal_low_threshold"])
    | (stocks_df.loc[valid_stats, "bid"] >= stocks_df.loc[valid_stats, "abnormal_high_threshold"])
)

stocks_df["signal_now"] = np.nan
stocks_df.loc[
    valid_stats
    & ~stocks_df["is_event_window"]
    & ~stocks_df["is_abnormal_zone"]
    & (stocks_df["ask"] <= stocks_df["buy_threshold"]),
    "signal_now",
] = 1
stocks_df.loc[
    valid_stats
    & ~stocks_df["is_event_window"]
    & ~stocks_df["is_abnormal_zone"]
    & (stocks_df["bid"] >= stocks_df["short_threshold"]),
    "signal_now",
] = -1
stocks_df.loc[stocks_df["is_event_window"] | stocks_df["is_abnormal_zone"], "signal_now"] = 0

# useful diagnostics inputs
stocks_df["zscore_mid"] = (stocks_df["mid"] - stocks_df["rolling_mean"]) / stocks_df["rolling_std"]

# =========================================================
# BUILD DIRECTIONAL POSITION
# =========================================================
def build_position_from_signal(group: pd.DataFrame) -> pd.DataFrame:
    sig = group["signal_now"].to_numpy()
    pos = np.zeros(len(group), dtype=np.int8)
    current = 0
    for i in range(len(sig)):
        if not np.isnan(sig[i]):
            current = int(sig[i])
        pos[i] = current
    out = group.copy()
    out["position"] = pos
    return out


bt = (
    stocks_df.groupby("ticker", group_keys=False, sort=False)
    .apply(build_position_from_signal)
    .reset_index(drop=True)
)

# =========================================================
# TARGET NOTIONAL BY TIMESTAMP
# =========================================================
long_counts = bt.groupby("timestamp")["position"].transform(lambda s: (s > 0).sum())
short_counts = bt.groupby("timestamp")["position"].transform(lambda s: (s < 0).sum())

long_notional_each = np.where(long_counts > 0, np.minimum(LONG_BOOK_CAPITAL / long_counts, PER_NAME_CAPITAL), 0.0)
short_notional_each = np.where(short_counts > 0, np.minimum(SHORT_BOOK_CAPITAL / short_counts, PER_NAME_CAPITAL), 0.0)

bt["target_notional"] = 0.0
bt.loc[bt["position"] > 0, "target_notional"] = long_notional_each[bt["position"] > 0]
bt.loc[bt["position"] < 0, "target_notional"] = -short_notional_each[bt["position"] < 0]

bt["raw_target_shares"] = bt["target_notional"] / bt["mid"]
abs_target = np.floor(np.abs(bt["raw_target_shares"]) / bt["lot_size"]) * bt["lot_size"]
bt["target_shares"] = np.sign(bt["raw_target_shares"]) * abs_target

short_mask = bt["target_shares"] < 0
bt.loc[short_mask, "target_shares"] = -np.minimum(
    np.abs(bt.loc[short_mask, "target_shares"]),
    bt.loc[short_mask, "locate_limit_shares"],
)
abs_target = np.floor(np.abs(bt["target_shares"]) / bt["lot_size"]) * bt["lot_size"]
bt["target_shares"] = np.sign(bt["target_shares"]) * abs_target

# =========================================================
# EXECUTION
# =========================================================

bt = (
    bt.groupby("ticker", group_keys=False, sort=False)
    .apply(execute_targets)
    .reset_index(drop=True)
)

# =========================================================
# PNL
# =========================================================
bt["next_mid"] = bt.groupby("ticker", sort=False)["mid"].shift(-1)
bt["mid_change"] = bt["next_mid"] - bt["mid"]
bt["gross_pnl"] = bt["shares"] * bt["mid_change"]

bt["trade_notional"] = bt["shares_traded"] * bt["mid"]
bt["explicit_fee_cost"] = bt["trade_notional"] * ((COMMISSION_BPS + EXCHANGE_BPS) / 10000)

position_change = bt["shares"] - bt["prev_shares"]
bt["sell_shares"] = np.where(position_change < 0, -position_change, 0.0)
bt["sell_notional"] = bt["sell_shares"] * bt["mid"]
bt["explicit_fee_cost"] += bt["sell_notional"] * (SEC_SELL_BPS / 10000)

bt["slippage_cost"] = bt["shares_traded"] * SLIPPAGE_SPREAD_FRAC * bt["spread"]

bt["holding_cost"] = 0.0
long_mask = bt["shares"] > 0
short_mask = bt["shares"] < 0
bt.loc[long_mask, "holding_cost"] = bt.loc[long_mask, "shares"] * bt.loc[long_mask, "mid"] * LONG_FIN_PER_MIN
bt.loc[short_mask, "holding_cost"] = (
    np.abs(bt.loc[short_mask, "shares"]) * bt.loc[short_mask, "mid"] * bt.loc[short_mask, "borrow_per_min"]
)

bt["trading_cost"] = bt["explicit_fee_cost"] + bt["slippage_cost"]
bt["net_pnl"] = bt["gross_pnl"] - bt["trading_cost"] - bt["holding_cost"]

# execution quality metrics before dropping final next_mid NaNs
bt["trade_side"] = np.select(
    [position_change > 0, position_change < 0],
    ["buy", "sell"],
    default="none",
)

bt = bt.dropna(subset=["next_mid"]).copy()

# =========================================================
# INDEX EXPOSURE / HEDGE
# =========================================================
bt["position_notional"] = bt["shares"] * bt["mid"]
bt["gross_abs_notional"] = np.abs(bt["position_notional"])
bt["index_weighted_dollar_exposure"] = bt["position_notional"] * bt["index_weight"]

index_exposure_by_time = (
    bt.groupby("timestamp", as_index=False)["index_weighted_dollar_exposure"]
    .sum()
    .rename(columns={"index_weighted_dollar_exposure": "net_index_exposure_dollars"})
)

index_df = index_df.sort_values("timestamp").copy()
index_df["next_mid"] = index_df["mid"].shift(-1)
index_df["index_mid_change"] = index_df["next_mid"] - index_df["mid"]

hedge_df = index_df.merge(index_exposure_by_time, on="timestamp", how="left")
hedge_df["net_index_exposure_dollars"] = hedge_df["net_index_exposure_dollars"].fillna(0.0)
hedge_df["index_target_shares_raw"] = -hedge_df["net_index_exposure_dollars"] / hedge_df["mid"]
hedge_df["index_target_shares"] = np.trunc(hedge_df["index_target_shares_raw"])

if not REBALANCE_HEDGE_EVERY_MINUTE:
    hedge_df["index_target_shares"] = hedge_df["index_target_shares"].shift(1).fillna(0.0)

hedge_df["index_prev_shares"] = hedge_df["index_target_shares"].shift(1).fillna(0.0)
hedge_df["index_shares_traded"] = (hedge_df["index_target_shares"] - hedge_df["index_prev_shares"]).abs()
hedge_df["hedge_pnl_gross"] = hedge_df["index_target_shares"] * hedge_df["index_mid_change"]

hedge_df["trade_notional"] = hedge_df["index_shares_traded"] * hedge_df["mid"]
hedge_df["hedge_explicit_fee_cost"] = hedge_df["trade_notional"] * ((COMMISSION_BPS + EXCHANGE_BPS) / 10000)
hedge_position_change = hedge_df["index_target_shares"] - hedge_df["index_prev_shares"]
hedge_df["hedge_sell_shares"] = np.where(hedge_position_change < 0, -hedge_position_change, 0.0)
hedge_df["hedge_sell_notional"] = hedge_df["hedge_sell_shares"] * hedge_df["mid"]
hedge_df["hedge_explicit_fee_cost"] += hedge_df["hedge_sell_notional"] * (SEC_SELL_BPS / 10000)
hedge_df["hedge_slippage_cost"] = hedge_df["index_shares_traded"] * SLIPPAGE_SPREAD_FRAC * hedge_df["spread"]

hedge_df["hedge_holding_cost"] = 0.0
hedge_long_mask = hedge_df["index_target_shares"] > 0
hedge_df.loc[hedge_long_mask, "hedge_holding_cost"] = (
    hedge_df.loc[hedge_long_mask, "index_target_shares"] * hedge_df.loc[hedge_long_mask, "mid"] * LONG_FIN_PER_MIN
)
# Reviewer note: no explicit borrow schedule is provided for the INDEX instrument,
# so short index borrow is set to zero rather than imputed.

hedge_df["hedge_trading_cost"] = hedge_df["hedge_explicit_fee_cost"] + hedge_df["hedge_slippage_cost"]
hedge_df["hedge_pnl_net"] = hedge_df["hedge_pnl_gross"] - hedge_df["hedge_trading_cost"] - hedge_df["hedge_holding_cost"]
hedge_df = hedge_df.dropna(subset=["next_mid"]).copy()

# =========================================================
# AGGREGATE RESULTS
# =========================================================
portfolio_pnl = (
    bt.groupby("timestamp", as_index=False)[[
        "gross_pnl", "explicit_fee_cost", "slippage_cost", "trading_cost", "holding_cost", "net_pnl",
        "shares_traded", "position_notional", "gross_abs_notional", "index_weighted_dollar_exposure",
        "trade_notional", "participation_used", "forced_exit_flag", "is_event_window", "is_abnormal_zone",
    ]]
    .sum()
    .rename(columns={
        "position_notional": "net_stock_notional",
        "gross_abs_notional": "gross_stock_notional",
        "index_weighted_dollar_exposure": "net_index_exposure_dollars_from_stocks",
    })
)

breadth = (
    bt.groupby("timestamp")
    .agg(
        long_names=("shares", lambda s: int((s > 0).sum())),
        short_names=("shares", lambda s: int((s < 0).sum())),
        active_names=("shares", lambda s: int((s != 0).sum())),
    )
    .reset_index()
)
portfolio_pnl = portfolio_pnl.merge(breadth, on="timestamp", how="left")

portfolio_pnl = portfolio_pnl.merge(
    hedge_df[[
        "timestamp", "index_target_shares", "index_shares_traded", "net_index_exposure_dollars",
        "hedge_pnl_gross", "hedge_explicit_fee_cost", "hedge_slippage_cost", "hedge_trading_cost",
        "hedge_holding_cost", "hedge_pnl_net",
    ]],
    on="timestamp",
    how="left",
)

fill_zero_cols = [
    "index_target_shares", "index_shares_traded", "net_index_exposure_dollars", "hedge_pnl_gross",
    "hedge_explicit_fee_cost", "hedge_slippage_cost", "hedge_trading_cost", "hedge_holding_cost", "hedge_pnl_net",
]
portfolio_pnl[fill_zero_cols] = portfolio_pnl[fill_zero_cols].fillna(0.0)

portfolio_pnl["net_pnl_hedged_gross_of_hedge_costs"] = portfolio_pnl["net_pnl"] + portfolio_pnl["hedge_pnl_gross"]
portfolio_pnl["net_pnl_hedged"] = portfolio_pnl["net_pnl"] + portfolio_pnl["hedge_pnl_net"]
portfolio_pnl["cum_gross_pnl"] = portfolio_pnl["gross_pnl"].cumsum()
portfolio_pnl["cum_net_pnl"] = portfolio_pnl["net_pnl"].cumsum()
portfolio_pnl["cum_net_pnl_hedged_gross_of_hedge_costs"] = portfolio_pnl["net_pnl_hedged_gross_of_hedge_costs"].cumsum()
portfolio_pnl["cum_net_pnl_hedged"] = portfolio_pnl["net_pnl_hedged"].cumsum()

# =========================================================
# RISK / PERFORMANCE DIAGNOSTICS
# =========================================================

# Per-minute portfolio diagnostics
portfolio_pnl["cost_to_gross_pnl_ratio"] = np.where(
    portfolio_pnl["gross_pnl"] != 0,
    portfolio_pnl["trading_cost"] / np.abs(portfolio_pnl["gross_pnl"]),
    np.nan,
)
portfolio_pnl["hedge_cost_to_hedge_gross_pnl_ratio"] = np.where(
    portfolio_pnl["hedge_pnl_gross"] != 0,
    portfolio_pnl["hedge_trading_cost"] / np.abs(portfolio_pnl["hedge_pnl_gross"]),
    np.nan,
)

unhedged_mdd, unhedged_drawdown = max_drawdown(portfolio_pnl["cum_net_pnl"])
hedged_mdd, hedged_drawdown = max_drawdown(portfolio_pnl["cum_net_pnl_hedged"])
portfolio_pnl["drawdown_unhedged"] = unhedged_drawdown
portfolio_pnl["drawdown_hedged"] = hedged_drawdown

# trade-level diagnostics using the minute after execution as the next realized mark-to-market period
trade_rows = bt.loc[bt["shares_traded"] > 0].copy()
trade_rows["next_bar_edge_after_costs"] = trade_rows["net_pnl"]
trade_rows["next_bar_edge_before_costs"] = trade_rows["gross_pnl"]
trade_rows["won_next_bar_after_costs"] = trade_rows["next_bar_edge_after_costs"] > 0

# signal quality buckets
signal_quality = bt.loc[bt["signal_now"].isin([1, -1]) & bt["rolling_std"].gt(0)].copy()
if not signal_quality.empty:
    signal_quality["abs_z_bucket"] = pd.cut(
        signal_quality["zscore_mid"].abs(),
        bins=[0, 1, 1.5, 2, 3, np.inf],
        right=False,
        labels=["[0,1)", "[1,1.5)", "[1.5,2)", "[2,3)", "[3,+)"],
    )
    signal_bucket_summary = (
        signal_quality.groupby("abs_z_bucket", observed=False)
        .agg(
            observations=("ticker", "size"),
            avg_next_bar_gross_pnl=("gross_pnl", "mean"),
            avg_next_bar_net_pnl=("net_pnl", "mean"),
            avg_spread=("spread", "mean"),
            avg_trade_notional=("trade_notional", "mean"),
        )
        .reset_index()
    )
else:
    signal_bucket_summary = pd.DataFrame(columns=["abs_z_bucket", "observations", "avg_next_bar_gross_pnl", "avg_next_bar_net_pnl", "avg_spread", "avg_trade_notional"])

# sector concentration diagnostics
sector_exposure = (
    bt.groupby(["timestamp", "sector"], as_index=False)["position_notional"]
    .sum()
)
sector_concentration = (
    sector_exposure.assign(abs_position_notional=lambda d: d["position_notional"].abs())
    .groupby("timestamp", as_index=False)["abs_position_notional"]
    .max()
    .rename(columns={"abs_position_notional": "largest_abs_sector_notional"})
)
portfolio_pnl = portfolio_pnl.merge(sector_concentration, on="timestamp", how="left")

# event / abnormal regime analysis
regime_summary = pd.DataFrame({
    "regime": ["event_window", "abnormal_zone", "normal_minutes"],
    "stock_rows": [
        int(bt["is_event_window"].sum()),
        int(bt["is_abnormal_zone"].sum()),
        int((~bt["is_event_window"] & ~bt["is_abnormal_zone"]).sum()),
    ],
    "gross_pnl": [
        float(bt.loc[bt["is_event_window"], "gross_pnl"].sum()),
        float(bt.loc[bt["is_abnormal_zone"], "gross_pnl"].sum()),
        float(bt.loc[~bt["is_event_window"] & ~bt["is_abnormal_zone"], "gross_pnl"].sum()),
    ],
    "net_pnl": [
        float(bt.loc[bt["is_event_window"], "net_pnl"].sum()),
        float(bt.loc[bt["is_abnormal_zone"], "net_pnl"].sum()),
        float(bt.loc[~bt["is_event_window"] & ~bt["is_abnormal_zone"], "net_pnl"].sum()),
    ],
})

# portfolio summary table
summary = {
    "total_capital": TOTAL_CAPITAL,
    "rolling_window": ROLLING_WINDOW,
    "buy_z": BUY_Z,
    "short_z": SHORT_Z,
    "abnormal_low_z": ABNORMAL_LOW_Z,
    "abnormal_high_z": ABNORMAL_HIGH_Z,
    "event_window_minutes_each_side": EVENT_WINDOW_MINUTES,
    "gross_stock_pnl": float(bt["gross_pnl"].sum()),
    "stock_explicit_fees": float(bt["explicit_fee_cost"].sum()),
    "stock_slippage_cost": float(bt["slippage_cost"].sum()),
    "stock_trading_costs": float(bt["trading_cost"].sum()),
    "stock_holding_costs": float(bt["holding_cost"].sum()),
    "stock_net_pnl": float(bt["net_pnl"].sum()),
    "hedge_gross_pnl": float(hedge_df["hedge_pnl_gross"].sum()),
    "hedge_explicit_fees": float(hedge_df["hedge_explicit_fee_cost"].sum()),
    "hedge_slippage_cost": float(hedge_df["hedge_slippage_cost"].sum()),
    "hedge_trading_costs": float(hedge_df["hedge_trading_cost"].sum()),
    "hedge_holding_costs": float(hedge_df["hedge_holding_cost"].sum()),
    "hedge_net_pnl": float(hedge_df["hedge_pnl_net"].sum()),
    "unhedged_net_pnl": float(portfolio_pnl["net_pnl"].sum()),
    "hedged_net_pnl": float(portfolio_pnl["net_pnl_hedged"].sum()),
    "total_stock_shares_traded": float(bt["shares_traded"].sum()),
    "total_stock_trade_notional": float(bt["trade_notional"].sum()),
    "total_hedge_shares_traded": float(hedge_df["index_shares_traded"].sum()),
    "avg_stock_index_exposure": float(portfolio_pnl["net_index_exposure_dollars_from_stocks"].mean()),
    "max_abs_stock_index_exposure": float(portfolio_pnl["net_index_exposure_dollars_from_stocks"].abs().max()),
    "avg_hedge_shares": float(portfolio_pnl["index_target_shares"].mean()),
    "max_abs_hedge_shares": float(portfolio_pnl["index_target_shares"].abs().max()),
    "avg_gross_stock_notional": float(portfolio_pnl["gross_stock_notional"].mean()),
    "max_gross_stock_notional": float(portfolio_pnl["gross_stock_notional"].max()),
    "avg_active_names": float(portfolio_pnl["active_names"].mean()),
    "max_active_names": int(portfolio_pnl["active_names"].max()),
    "avg_long_names": float(portfolio_pnl["long_names"].mean()),
    "avg_short_names": float(portfolio_pnl["short_names"].mean()),
    "unhedged_mean_minute_pnl": float(portfolio_pnl["net_pnl"].mean()),
    "unhedged_minute_pnl_std": float(portfolio_pnl["net_pnl"].std(ddof=0)),
    "hedged_mean_minute_pnl": float(portfolio_pnl["net_pnl_hedged"].mean()),
    "hedged_minute_pnl_std": float(portfolio_pnl["net_pnl_hedged"].std(ddof=0)),
    "unhedged_annualized_sharpe_like": annualized_sharpe(portfolio_pnl["net_pnl"]),
    "hedged_annualized_sharpe_like": annualized_sharpe(portfolio_pnl["net_pnl_hedged"]),
    "unhedged_max_drawdown": unhedged_mdd,
    "hedged_max_drawdown": hedged_mdd,
    "fraction_profitable_minutes_unhedged": float((portfolio_pnl["net_pnl"] > 0).mean()),
    "fraction_profitable_minutes_hedged": float((portfolio_pnl["net_pnl_hedged"] > 0).mean()),
    "trade_count": int((bt["shares_traded"] > 0).sum()),
    "forced_exit_count": int(bt["forced_exit_flag"].sum()),
    "avg_participation_used_when_trading": float(bt.loc[bt["shares_traded"] > 0, "participation_used"].mean()),
    "max_participation_used_when_trading": float(bt.loc[bt["shares_traded"] > 0, "participation_used"].max()),
    "avg_largest_abs_sector_notional": float(portfolio_pnl["largest_abs_sector_notional"].mean()),
    "max_largest_abs_sector_notional": float(portfolio_pnl["largest_abs_sector_notional"].max()),
    "next_bar_trade_win_rate_after_costs": float(trade_rows["won_next_bar_after_costs"].mean()) if not trade_rows.empty else np.nan,
    "avg_next_bar_edge_after_costs_per_trade": float(trade_rows["next_bar_edge_after_costs"].mean()) if not trade_rows.empty else np.nan,
    "avg_next_bar_edge_before_costs_per_trade": float(trade_rows["next_bar_edge_before_costs"].mean()) if not trade_rows.empty else np.nan,
}
summary.update(summarize_side(bt, 1, "long_book"))
summary.update(summarize_side(bt, -1, "short_book"))
summary_df = pd.DataFrame([summary])

# additional daily-style summary by date, although data may span multiple dates
by_date = (
    portfolio_pnl.assign(date=lambda d: d["timestamp"].dt.date)
    .groupby("date", as_index=False)
    .agg(
        gross_pnl=("gross_pnl", "sum"),
        net_pnl=("net_pnl", "sum"),
        net_pnl_hedged=("net_pnl_hedged", "sum"),
        stock_trading_cost=("trading_cost", "sum"),
        hedge_trading_cost=("hedge_trading_cost", "sum"),
        avg_active_names=("active_names", "mean"),
        max_gross_stock_notional=("gross_stock_notional", "max"),
    )
)

# save outputs
summary_df.to_csv("outputs/summary_metrics.csv", index=False)
portfolio_pnl.to_csv("outputs/portfolio_pnl_timeseries.csv", index=False)
by_date.to_csv("outputs/performance_by_date.csv", index=False)
regime_summary.to_csv("outputs/regime_summary.csv", index=False)
signal_bucket_summary.to_csv("outputs/signal_bucket_summary.csv", index=False)
trade_rows.to_csv("outputs/trade_level_diagnostics.csv", index=False)

# =========================================================
# PRINT SUMMARY
# =========================================================
print("\n============== STRATEGY SUMMARY ==============")
print("Signal parameters:")
print(f"  Rolling window:               {ROLLING_WINDOW}")
print(f"  Buy threshold:                mean - {BUY_Z:.1f} std")
print(f"  Short threshold:              mean + {SHORT_Z:.1f} std")
print(f"  Abnormal low threshold:       mean - {ABNORMAL_LOW_Z:.1f} std")
print(f"  Abnormal high threshold:      mean + {ABNORMAL_HIGH_Z:.1f} std")
print(f"  Event window (+/- min):       {EVENT_WINDOW_MINUTES}")
print()
print(f"Gross stock PnL:                {bt['gross_pnl'].sum():.4f}")
print(f"Stock explicit fees:            {bt['explicit_fee_cost'].sum():.4f}")
print(f"Stock slippage cost:            {bt['slippage_cost'].sum():.4f}")
print(f"Stock trading costs:            {bt['trading_cost'].sum():.4f}")
print(f"Stock holding costs:            {bt['holding_cost'].sum():.4f}")
print(f"Stock net PnL:                  {bt['net_pnl'].sum():.4f}")
print()
print(f"Hedge gross PnL:                {hedge_df['hedge_pnl_gross'].sum():.4f}")
print(f"Hedge explicit fees:            {hedge_df['hedge_explicit_fee_cost'].sum():.4f}")
print(f"Hedge slippage cost:            {hedge_df['hedge_slippage_cost'].sum():.4f}")
print(f"Hedge trading costs:            {hedge_df['hedge_trading_cost'].sum():.4f}")
print(f"Hedge holding costs:            {hedge_df['hedge_holding_cost'].sum():.4f}")
print(f"Hedge net PnL:                  {hedge_df['hedge_pnl_net'].sum():.4f}")
print()
print(f"Unhedged net PnL:               {portfolio_pnl['net_pnl'].sum():.4f}")
print(f"Hedged net PnL:                 {portfolio_pnl['net_pnl_hedged'].sum():.4f}")
print(f"Unhedged Sharpe-like:           {summary_df.loc[0, 'unhedged_annualized_sharpe_like']:.4f}")
print(f"Hedged Sharpe-like:             {summary_df.loc[0, 'hedged_annualized_sharpe_like']:.4f}")
print(f"Unhedged max drawdown:          {summary_df.loc[0, 'unhedged_max_drawdown']:.4f}")
print(f"Hedged max drawdown:            {summary_df.loc[0, 'hedged_max_drawdown']:.4f}")
print()
print(f"Total stock shares traded:      {bt['shares_traded'].sum():.0f}")
print(f"Total stock trade notional:     {bt['trade_notional'].sum():.4f}")
print(f"Total hedge shares traded:      {hedge_df['index_shares_traded'].sum():.0f}")
print(f"Trade count:                    {summary_df.loc[0, 'trade_count']:.0f}")
print(f"Forced exits:                   {summary_df.loc[0, 'forced_exit_count']:.0f}")
print(f"Next-bar win rate after costs:  {summary_df.loc[0, 'next_bar_trade_win_rate_after_costs']:.4f}")
print()
print(f"Average active names:           {summary_df.loc[0, 'avg_active_names']:.4f}")
print(f"Max active names:               {summary_df.loc[0, 'max_active_names']:.0f}")
print(f"Average stock index exposure:   {portfolio_pnl['net_index_exposure_dollars_from_stocks'].mean():.4f}")
print(f"Max abs stock idx exposure:     {portfolio_pnl['net_index_exposure_dollars_from_stocks'].abs().max():.4f}")
print(f"Average hedge shares:           {portfolio_pnl['index_target_shares'].mean():.4f}")
print(f"Max abs hedge shares:           {portfolio_pnl['index_target_shares'].abs().max():.4f}")
print(f"Average largest sector notional:{summary_df.loc[0, 'avg_largest_abs_sector_notional']:.4f}")
print()
print(f"Outputs written to:             outputs/")

# =========================================================
# PLOTS
# =========================================================
plt.figure(figsize=PLOT_FIGSIZE)
plt.plot(portfolio_pnl["timestamp"], portfolio_pnl["cum_gross_pnl"], label="Gross Stock PnL")
plt.plot(portfolio_pnl["timestamp"], portfolio_pnl["cum_net_pnl"], label="Net Stock PnL")
plt.plot(portfolio_pnl["timestamp"], portfolio_pnl["cum_net_pnl_hedged_gross_of_hedge_costs"], label="Hedged PnL (Gross Hedge)")
plt.plot(portfolio_pnl["timestamp"], portfolio_pnl["cum_net_pnl_hedged"], label="Hedged Net PnL")
plt.title("Cumulative PnL")
plt.xlabel("Time")
plt.ylabel("PnL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/cumulative_pnl.png", dpi=160)
plt.close()

plt.figure(figsize=PLOT_FIGSIZE)
plt.plot(portfolio_pnl["timestamp"], portfolio_pnl["net_index_exposure_dollars_from_stocks"])
plt.title("Index-Weighted Dollar Exposure From Stock Book")
plt.xlabel("Time")
plt.ylabel("Dollar Exposure")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/stock_index_exposure.png", dpi=160)
plt.close()

plt.figure(figsize=PLOT_FIGSIZE)
plt.plot(portfolio_pnl["timestamp"], portfolio_pnl["index_target_shares"])
plt.title("Index Hedge Shares Over Time")
plt.xlabel("Time")
plt.ylabel("Index Shares")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/index_hedge_shares.png", dpi=160)
plt.close()

plt.figure(figsize=PLOT_FIGSIZE)
plt.plot(portfolio_pnl["timestamp"], portfolio_pnl["drawdown_unhedged"], label="Unhedged drawdown")
plt.plot(portfolio_pnl["timestamp"], portfolio_pnl["drawdown_hedged"], label="Hedged drawdown")
plt.title("Drawdown")
plt.xlabel("Time")
plt.ylabel("PnL drawdown")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/drawdown.png", dpi=160)
plt.close()

plt.figure(figsize=PLOT_FIGSIZE)
plt.plot(portfolio_pnl["timestamp"], portfolio_pnl["active_names"])
plt.title("Active Names Over Time")
plt.xlabel("Time")
plt.ylabel("Active names")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/active_names.png", dpi=160)
plt.close()

print("Generated figures:")
for name in [
    "cumulative_pnl.png",
    "stock_index_exposure.png",
    "index_hedge_shares.png",
    "drawdown.png",
    "active_names.png",
]:
    print(f"  - outputs/{name}")
