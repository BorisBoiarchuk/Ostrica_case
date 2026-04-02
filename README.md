# Intraday Trading Strategy Case Study

## Overview

This project implements a systematic intraday trading strategy across a universe of ~500 equities.

The goal is to:
- design a data-driven trading signal
- simulate realistic execution
- incorporate transaction costs and constraints
- evaluate portfolio performance and risk
- apply a portfolio hedge using index exposure

The project is fully reproducible and organized as a modular pipeline.

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

From the project root:

```bash
python main.py
```

This will:
- validate and clean data
- generate signals
- simulate execution
- compute PnL and risk metrics
- produce plots and output files in `/outputs`

---

## Project Structure

```text
project/
├── src/
│   ├── config.py
│   ├── validation.py
│   ├── execution.py
│   ├── diagnostics.py
├── main.py
├── research.py
├── data/
├── outputs/
├── requirements.txt
└── README.md
```

---

## Strategy Description (Details can be found in research.py)

### Signal: Rolling Mean Reversion

For each stock:
- compute rolling mean and standard deviation over the last 100 observations
- shift by one period to avoid look-ahead bias

Trading rules:
- Long if: ask ≤ mean − 1.0 × std  
- Short if: bid ≥ mean + 1.0 × std  

---

### Risk Filters

Trading is disabled in two regimes:

- Event window: ±10 minutes around scheduled events  
- Abnormal price region:
  - ask ≤ mean − 1.5 × std  
  - bid ≥ mean + 2.0 × std  

---

## Portfolio Construction

- Total capital: $10M
- Long book: 50%
- Short book: 50%
- Max per-name allocation: $25k

---

## Execution Model

- volume-based participation limits
- lot-size rounding
- locate constraints for shorting
- forced exits in risk regimes

---

## Transaction Costs

- commissions, exchange fees, SEC fees
- spread-based slippage
- financing and borrow costs

---

## Hedging

Uses an index-weighted exposure proxy:
- position × index weight
- offset via index position

---

## Outputs

- cumulative PnL plots
- exposure plots
- hedge position plots
- CSV summary tables

---

## Performance Metrics

- net and gross PnL
- Sharpe-like metric
- max drawdown
- trade count and win rate
- exposure and concentration metrics

---

## Limitations

- no beta hedge (weight-based proxy only)
- simplified slippage model
- no order book dynamics
- no latency modeling

---

## Summary

This project demonstrates a full intraday trading pipeline with:
- robust validation
- realistic execution
- cost-aware backtesting
- structured diagnostics

---

## Reproducibility

1. install dependencies  
2. run `python main.py`

---

## Author

Boris Boiarchuk
