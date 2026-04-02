import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import *

# =========================================================
# VALIDATION HELPERS
# =========================================================
def add_check(rows, dataset, check_name, severity, value, status, action):
    rows.append({
        "dataset": dataset,
        "check_name": check_name,
        "severity": severity,
        "value": value,
        "status": status,
        "action": action,
    })



def resolve_intraday_duplicates(df: pd.DataFrame):
    before_rows = len(df)

    exact_dup_count = int(df.duplicated().sum())
    df_no_exact = df.drop_duplicates().copy()

    dup_key_mask = df_no_exact.duplicated(subset=["ticker", "timestamp"], keep=False)
    dup_key_rows = int(dup_key_mask.sum())
    dup_key_groups = int(
        df_no_exact.loc[dup_key_mask, ["ticker", "timestamp"]].drop_duplicates().shape[0]
    )

    if dup_key_rows == 0:
        duplicate_summary = {
            "rows_before": before_rows,
            "exact_duplicates_dropped": exact_dup_count,
            "duplicate_key_rows_resolved": 0,
            "duplicate_key_groups_resolved": 0,
            "rows_after": len(df_no_exact),
        }
        return df_no_exact.sort_values(["ticker", "timestamp"]).reset_index(drop=True), duplicate_summary

    non_dup = df_no_exact.loc[~dup_key_mask].copy()
    dup_df = df_no_exact.loc[dup_key_mask].copy()

    agg_df = (
        dup_df.sort_values(["ticker", "timestamp"])
        .groupby(["ticker", "timestamp"], as_index=False)
        .agg({
            "date": "first",
            "minute": "first",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "adj_close": "last",
            "bid": "last",
            "ask": "last",
            "volume": "sum",
        })
    )

    out = pd.concat([non_dup, agg_df], ignore_index=True)
    out = out.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    duplicate_summary = {
        "rows_before": before_rows,
        "exact_duplicates_dropped": exact_dup_count,
        "duplicate_key_rows_resolved": dup_key_rows,
        "duplicate_key_groups_resolved": dup_key_groups,
        "rows_after": len(out),
    }
    return out, duplicate_summary



def build_validation_report(
    intraday_prices: pd.DataFrame,
    ticker_metadata: pd.DataFrame,
    index_constituents: pd.DataFrame,
    scheduled_events: pd.DataFrame,
    fee_schedule: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    required_cols = {
        "intraday_prices": ["timestamp", "date", "minute", "ticker", "open", "high", "low", "close", "adj_close", "bid", "ask", "volume"],
        "ticker_metadata": ["ticker", "sector", "liquidity_score_1to5", "hard_to_borrow", "borrow_fee_bps_annual", "locate_limit_shares", "lot_size", "tick_size", "max_participation_rate"],
        "index_constituents": ["ticker", "index_weight"],
        "scheduled_events": ["timestamp", "event_type", "notes"],
        "fee_schedule": ["fee_component", "value"],
    }
    dfs = {
        "intraday_prices": intraday_prices,
        "ticker_metadata": ticker_metadata,
        "index_constituents": index_constituents,
        "scheduled_events": scheduled_events,
        "fee_schedule": fee_schedule,
    }

    for name, req in required_cols.items():
        missing = [c for c in req if c not in dfs[name].columns]
        add_check(rows, name, "required_columns", "error", 0 if len(missing) == 0 else len(missing),
                  "PASS" if len(missing) == 0 else f"FAIL: missing {missing}", "Raise error if missing")

    add_check(rows, "intraday_prices", "null_timestamps", "error", int(intraday_prices["timestamp"].isna().sum()),
              "PASS" if intraday_prices["timestamp"].isna().sum() == 0 else "FAIL", "Raise error if > 0")
    add_check(rows, "scheduled_events", "null_timestamps", "error", int(scheduled_events["timestamp"].isna().sum()),
              "PASS" if scheduled_events["timestamp"].isna().sum() == 0 else "FAIL", "Raise error if > 0")

    add_check(rows, "intraday_prices", "row_count", "info", int(len(intraday_prices)), "INFO", "Informational")
    add_check(rows, "intraday_prices", "ticker_count", "info", int(intraday_prices["ticker"].nunique()), "INFO", "Informational")
    add_check(rows, "ticker_metadata", "ticker_count", "info", int(ticker_metadata["ticker"].nunique()), "INFO", "Informational")
    add_check(rows, "index_constituents", "ticker_count", "info", int(index_constituents["ticker"].nunique()), "INFO", "Informational")
    add_check(rows, "scheduled_events", "event_count", "info", int(len(scheduled_events)), "INFO", "Informational")

    exact_dup_count = int(intraday_prices.duplicated().sum())
    dup_key_count = int(intraday_prices.duplicated(subset=["ticker", "timestamp"]).sum())
    add_check(rows, "intraday_prices", "exact_duplicate_rows", "warning", exact_dup_count,
              "PASS" if exact_dup_count == 0 else "WARN", "Drop exact duplicates")
    add_check(rows, "intraday_prices", "duplicate_(ticker,timestamp)_rows", "warning", dup_key_count,
              "PASS" if dup_key_count == 0 else "WARN", "Aggregate conflicting bars to one row per key")

    crossed_count = int((intraday_prices["bid"] > intraday_prices["ask"]).sum())
    locked_count = int((intraday_prices["bid"] == intraday_prices["ask"]).sum())
    non_positive_bid = int((intraday_prices["bid"] <= 0).sum())
    non_positive_ask = int((intraday_prices["ask"] <= 0).sum())
    negative_volume = int((intraday_prices["volume"] < 0).sum())

    add_check(rows, "intraday_prices", "crossed_markets_(bid>ask)", "error", crossed_count,
              "PASS" if crossed_count == 0 else "FAIL", "Drop crossed rows before backtest")
    add_check(rows, "intraday_prices", "locked_markets_(bid=ask)", "warning", locked_count,
              "PASS" if locked_count == 0 else "WARN", "Keep but track")
    add_check(rows, "intraday_prices", "non_positive_bid", "error", non_positive_bid,
              "PASS" if non_positive_bid == 0 else "FAIL", "Raise error if > 0")
    add_check(rows, "intraday_prices", "non_positive_ask", "error", non_positive_ask,
              "PASS" if non_positive_ask == 0 else "FAIL", "Raise error if > 0")
    add_check(rows, "intraday_prices", "negative_volume_rows", "error", negative_volume,
              "PASS" if negative_volume == 0 else "FAIL", "Drop if enabled, else raise")

    bad_minute = int((~intraday_prices["minute"].between(0, 389)).sum())
    date_mismatch = int((intraday_prices["timestamp"].dt.strftime("%Y-%m-%d") != intraday_prices["date"].astype(str)).sum())
    add_check(rows, "intraday_prices", "minute_outside_[0,389]", "error", bad_minute,
              "PASS" if bad_minute == 0 else "FAIL", "Raise error if > 0")
    add_check(rows, "intraday_prices", "date_timestamp_mismatch", "error", date_mismatch,
              "PASS" if date_mismatch == 0 else "FAIL", "Raise error if > 0")

    add_check(rows, "ticker_metadata", "duplicate_tickers", "error", int(ticker_metadata["ticker"].duplicated().sum()),
              "PASS" if ticker_metadata["ticker"].duplicated().sum() == 0 else "FAIL", "Raise error if > 0")
    add_check(rows, "index_constituents", "duplicate_tickers", "error", int(index_constituents["ticker"].duplicated().sum()),
              "PASS" if index_constituents["ticker"].duplicated().sum() == 0 else "FAIL", "Raise error if > 0")
    add_check(rows, "fee_schedule", "duplicate_fee_components", "error", int(fee_schedule["fee_component"].duplicated().sum()),
              "PASS" if fee_schedule["fee_component"].duplicated().sum() == 0 else "FAIL", "Raise error if > 0")

    invalid_liq = int((~ticker_metadata["liquidity_score_1to5"].isin([1, 2, 3, 4, 5])).sum())
    non_positive_lot = int((ticker_metadata["lot_size"] <= 0).sum())
    non_positive_tick = int((ticker_metadata["tick_size"] <= 0).sum())
    invalid_participation = int(((ticker_metadata["max_participation_rate"] <= 0) | (ticker_metadata["max_participation_rate"] > 1)).sum())
    negative_locate = int((ticker_metadata["locate_limit_shares"] < 0).sum())

    add_check(rows, "ticker_metadata", "invalid_liquidity_score", "error", invalid_liq, "PASS" if invalid_liq == 0 else "FAIL", "Raise error if > 0")
    add_check(rows, "ticker_metadata", "non_positive_lot_size", "error", non_positive_lot, "PASS" if non_positive_lot == 0 else "FAIL", "Raise error if > 0")
    add_check(rows, "ticker_metadata", "non_positive_tick_size", "error", non_positive_tick, "PASS" if non_positive_tick == 0 else "FAIL", "Raise error if > 0")
    add_check(rows, "ticker_metadata", "invalid_max_participation_rate", "error", invalid_participation, "PASS" if invalid_participation == 0 else "FAIL", "Raise error if > 0")
    add_check(rows, "ticker_metadata", "negative_locate_limit", "error", negative_locate, "PASS" if negative_locate == 0 else "FAIL", "Raise error if > 0")

    weight_sum = float(index_constituents["index_weight"].sum())
    bad_weights = int(((index_constituents["index_weight"] < 0) | (index_constituents["index_weight"] > 1)).sum())
    add_check(rows, "index_constituents", "weights_outside_[0,1]", "error", bad_weights,
              "PASS" if bad_weights == 0 else "FAIL", "Raise error if > 0")
    add_check(rows, "index_constituents", "weight_sum", "warning", round(weight_sum, 12),
              "PASS" if np.isclose(weight_sum, 1.0, atol=1e-6) else "WARN", "Raise or review if not ~1")

    intraday_tickers = set(intraday_prices["ticker"].unique())
    stock_tickers = {t for t in intraday_tickers if t != "INDEX"}
    meta_tickers = set(ticker_metadata["ticker"].unique())
    index_tickers = set(index_constituents["ticker"].unique())

    missing_in_meta = len(stock_tickers - meta_tickers)
    missing_in_index = len(stock_tickers - index_tickers)
    extra_meta = len(meta_tickers - stock_tickers)
    extra_index = len(index_tickers - stock_tickers)

    add_check(rows, "universe", "stock_tickers_missing_from_metadata", "error", missing_in_meta,
              "PASS" if missing_in_meta == 0 else "FAIL", "Raise error if > 0")
    add_check(rows, "universe", "stock_tickers_missing_from_index_file", "warning", missing_in_index,
              "PASS" if missing_in_index == 0 else "WARN", "Fill weight=0 or review")
    add_check(rows, "universe", "metadata_tickers_missing_from_intraday", "info", extra_meta, "INFO", "Informational")
    add_check(rows, "universe", "index_tickers_missing_from_intraday", "info", extra_index, "INFO", "Informational")

    required_fee_components = {
        "commission_bps_notional",
        "exchange_fee_bps_notional",
        "sec_fee_bps_sell_notional",
        "financing_bps_annual_long_cash",
        "borrow_fee_bps_annual_short",
    }
    found_fee_components = set(fee_schedule["fee_component"])
    missing_fee_components = required_fee_components - found_fee_components
    add_check(rows, "fee_schedule", "missing_required_fee_components", "error", len(missing_fee_components),
              "PASS" if len(missing_fee_components) == 0 else f"FAIL: {sorted(missing_fee_components)}", "Raise error if > 0")

    observed_clock = set(intraday_prices["timestamp"].dt.strftime("%H:%M:%S").unique())
    events_outside_clock = int((~scheduled_events["timestamp"].dt.strftime("%H:%M:%S").isin(observed_clock)).sum())
    add_check(rows, "scheduled_events", "events_outside_observed_intraday_clock", "warning", events_outside_clock,
              "PASS" if events_outside_clock == 0 else "WARN", "Review event alignment")

    return pd.DataFrame(rows)



def clean_validated_inputs(
    intraday_prices: pd.DataFrame,
    ticker_metadata: pd.DataFrame,
    index_constituents: pd.DataFrame,
    scheduled_events: pd.DataFrame,
    fee_schedule: pd.DataFrame,
):
    if intraday_prices["timestamp"].isna().any():
        raise ValueError("intraday_prices has unparsable timestamps")
    if scheduled_events["timestamp"].isna().any():
        raise ValueError("scheduled_events has unparsable timestamps")

    if (intraday_prices["bid"] <= 0).any():
        raise ValueError("intraday_prices has non-positive bid values")
    if (intraday_prices["ask"] <= 0).any():
        raise ValueError("intraday_prices has non-positive ask values")
    if (~intraday_prices["minute"].between(0, 389)).any():
        raise ValueError("intraday_prices has minute values outside [0,389]")
    if (intraday_prices["timestamp"].dt.strftime("%Y-%m-%d") != intraday_prices["date"].astype(str)).any():
        raise ValueError("intraday_prices has date/timestamp mismatch")

    if ticker_metadata["ticker"].duplicated().any():
        raise ValueError("ticker_metadata has duplicate tickers")
    if index_constituents["ticker"].duplicated().any():
        raise ValueError("index_constituents has duplicate tickers")
    if fee_schedule["fee_component"].duplicated().any():
        raise ValueError("fee_schedule has duplicate fee_component values")

    if (~ticker_metadata["liquidity_score_1to5"].isin([1, 2, 3, 4, 5])).any():
        raise ValueError("invalid liquidity_score_1to5 values found")
    if (ticker_metadata["lot_size"] <= 0).any():
        raise ValueError("non-positive lot_size found")
    if (ticker_metadata["tick_size"] <= 0).any():
        raise ValueError("non-positive tick_size found")
    if ((ticker_metadata["max_participation_rate"] <= 0) | (ticker_metadata["max_participation_rate"] > 1)).any():
        raise ValueError("invalid max_participation_rate found")
    if (ticker_metadata["locate_limit_shares"] < 0).any():
        raise ValueError("negative locate_limit_shares found")

    weight_sum = index_constituents["index_weight"].sum()
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(f"index weights do not sum to 1; found {weight_sum}")

    stock_tickers = set(intraday_prices.loc[intraday_prices["ticker"] != "INDEX", "ticker"].unique())
    meta_tickers = set(ticker_metadata["ticker"].unique())
    missing_in_meta = stock_tickers - meta_tickers
    if missing_in_meta:
        raise ValueError(f"stock tickers missing from metadata: {sorted(list(missing_in_meta))[:10]}")

    intraday_prices, duplicate_summary = resolve_intraday_duplicates(intraday_prices)

    if DROP_NEGATIVE_VOLUME:
        intraday_prices = intraday_prices.loc[intraday_prices["volume"] >= 0].copy()
    else:
        if (intraday_prices["volume"] < 0).any():
            raise ValueError("negative volume found")

    if DROP_CROSSED_MARKETS:
        intraday_prices = intraday_prices.loc[intraday_prices["bid"] <= intraday_prices["ask"]].copy()
    else:
        if (intraday_prices["bid"] > intraday_prices["ask"]).any():
            raise ValueError("crossed markets found")

    if not ALLOW_LOCKED_MARKETS:
        intraday_prices = intraday_prices.loc[intraday_prices["bid"] < intraday_prices["ask"]].copy()

    intraday_prices = intraday_prices.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    scheduled_events = scheduled_events.sort_values("timestamp").reset_index(drop=True)

    return intraday_prices, ticker_metadata, index_constituents, scheduled_events, fee_schedule, duplicate_summary



def print_validation_summary(validation_report: pd.DataFrame, duplicate_summary: dict | None = None):
    print("\n==================== DATA VALIDATION SUMMARY ====================")

    errors = validation_report.loc[validation_report["severity"] == "error"]
    warnings = validation_report.loc[validation_report["severity"] == "warning"]

    fail_errors = errors.loc[~errors["status"].astype(str).str.startswith("PASS")]
    warn_items = warnings.loc[~warnings["status"].astype(str).str.startswith("PASS")]

    print(f"Total checks:   {len(validation_report)}")
    print(f"Error checks:   {len(errors)}")
    print(f"Warning checks: {len(warnings)}")
    print(f"Failed errors:  {len(fail_errors)}")
    print(f"Warnings hit:   {len(warn_items)}")

    if duplicate_summary is not None:
        print("\nDuplicate handling:")
        print(f"  Exact duplicates dropped:         {duplicate_summary['exact_duplicates_dropped']:,}")
        print(f"  Duplicate key rows resolved:      {duplicate_summary['duplicate_key_rows_resolved']:,}")
        print(f"  Duplicate key groups resolved:    {duplicate_summary['duplicate_key_groups_resolved']:,}")
        print(f"  Rows before cleaning:             {duplicate_summary['rows_before']:,}")
        print(f"  Rows after duplicate handling:    {duplicate_summary['rows_after']:,}")

    print("\nTriggered issues:")
    triggered = validation_report.loc[
        ~validation_report["status"].astype(str).str.startswith("PASS")
        & (validation_report["status"] != "INFO"),
        ["dataset", "check_name", "value", "status", "action"],
    ]

    if triggered.empty:
        print("  None")
    else:
        print(triggered.to_string(index=False))