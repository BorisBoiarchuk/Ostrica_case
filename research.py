import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import *
from src.validation import clean_validated_inputs

# =========================================================
# RESEARCH
# Here we just do data exploration looking for some patterns
# to come up with some sensible strategy
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

intraday_prices, ticker_metadata, index_constituents, scheduled_events, fee_schedule, duplicate_summary = clean_validated_inputs(
    intraday_prices=intraday_prices,
    ticker_metadata=ticker_metadata,
    index_constituents=index_constituents,
    scheduled_events=scheduled_events,
    fee_schedule=fee_schedule,
)

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

# plot 16 random tickers
fig, axs = plt.subplots(4, 4, figsize=(18, 14), constrained_layout=True)

trial_tickers_1 = stocks_df["ticker"].drop_duplicates().sample(16, random_state=123)

for ax, ticker in zip(axs.flat, trial_tickers_1):
    df_t = stocks_df.loc[stocks_df["ticker"] == ticker]
    ax.plot(df_t["timestamp"], df_t["mid"])
    ax.set_title(ticker)
    ax.tick_params(axis="x", rotation=45)
plt.suptitle("Random 16 tickers: Mid price", fontsize=16)
plt.savefig("outputs/research_outputs/16_rnd_tickers.png", dpi=160)

# it looks like in last 4 minutes of the trading session a sort structural break happens
# let's try to see how do the series look like before that

fig, axs = plt.subplots(4, 4, figsize=(18, 14), constrained_layout=True)

trial_tickers_2 = stocks_df["ticker"].drop_duplicates().sample(16, random_state=123)

for ax, ticker in zip(axs.flat, trial_tickers_2):
    df_t = stocks_df.loc[stocks_df["ticker"] == ticker]
    ax.plot(df_t["timestamp"][:-4], df_t["mid"][:-4])
    ax.set_title(ticker)
    ax.tick_params(axis="x", rotation=45)
plt.suptitle("Random 16 tickers: Mid price (without last 4 minutes)", fontsize=16)
plt.savefig("outputs/research_outputs/16_rnd_tickers_no_break.png", dpi=160)

# now it looks like the price for every ticker is reverting around 200 with no visible trends,
# no seasonality, no heteroskedasticity, so it makes sense to try a mean reversion strategy
time_mask_mean_hist = stocks_df["timestamp"] < "2025-11-14 15:56:00" # time of structural break in the end
plt.figure(figsize = (12, 6))
plt.hist(stocks_df[time_mask_mean_hist].groupby("ticker")["mid"].mean())
plt.title("Distribution of means among tickers")
plt.savefig("outputs/research_outputs/dstr_means.png", dpi=160)
# from the histogram we see that the mean seem to be the same, centered around 200 for all tickers

# similar the std seems to be the same for all tickers
time_mask_std_hist = stocks_df["timestamp"] < "2025-11-14 15:56:00"
plt.figure(figsize = (12, 6))
plt.hist(stocks_df[time_mask_std_hist].groupby("ticker")["mid"].std())
plt.title("Distribution of standard deviations among tickers")
plt.savefig("outputs/research_outputs/dstr_stds.png", dpi=160)

# overall it seems that tickers have very similar properties
# let's look at some the time series of a single day more closely
time_mask_single_day = (
    (stocks_df["timestamp"] >= "2025-11-10 09:30:00") &
    (stocks_df["timestamp"] < "2025-11-10 16:00:00")
)
trial_tickers_3 = stocks_df["ticker"].drop_duplicates().sample(16, random_state=123)  


fig, axs = plt.subplots(4, 4, figsize=(18, 14), constrained_layout=True)                    
for ax, ticker in zip(axs.flat, trial_tickers_3):
    df_t = stocks_df[time_mask_single_day].loc[stocks_df["ticker"] == ticker]
    ax.plot(df_t["timestamp"], df_t["mid"])
    ax.set_title(ticker)
    ax.tick_params(axis="x", rotation=45)
plt.suptitle("Random 16 tickers: Mid price during 2025-11-10", fontsize=16)
plt.savefig("outputs/research_outputs/20251110_16_rnd_tickers.png", dpi=160)

# Judging from the plots there are indeed no visible trends, no seasonality, 
# no heteroskedasticity, so the exploiting mean reversion property seems to be the only sensible idea.
#
# Setting the mean reversion level to 200 directly would be cheating due to look ahead bias,
# so we set to be the average of last 100 observations (so it would be close to the true mean).
#
# We also observed structural breaks in the end, so our strategy should be also ready for them.
# We can do that by not allowing to trade and flatten the position when the price becomes abnormally high or
# low, signalling the possible regime break
#
# Also we don't trade near the events, because we don't know how these will affect the price.