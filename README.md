# Volatility Smile Relative-Value Straddle

A fully self-contained equity options backtesting framework that trades straddles based on cross-sectional z-scores of the **IV Risk Premium (IVRP)** and **smile shape** (put skew, call skew, curvature) across ~460 S&P 500 names.

No external backtest framework dependencies. All simulation logic, capital accounting, signal generation, and reporting are written from scratch.

---

## Strategy Overview

Each month, on the third Friday, the strategy:

1. **Scores** every ticker by `IVRP = ATM IV − Realized Vol (21-day)`, z-scored against each ticker's own trailing 12-month history.
2. **Filters** via a smile-shape eligibility gate — shorts excluded when curvature/skew z-scores signal real tail risk; longs excluded when the smile is anomalously flat.
3. **Ranks** within each eligible pool into quintiles. Q5 (most expensive vol) → short straddle. Q1 (cheapest vol) → long straddle.
4. **Monitors** daily: exits on convergence (IVRP reverts within ±0.1σ), divergence (smile deteriorates past 2σ), or forced close at DTE ≤ 5.

---

## Repository Structure

```
├── config.py               # All parameters (signal, eligibility, exit rules, data paths)
├── signal_generation.py    # Data loading, smile snapshot, z-scoring, quintile ranking
├── backtest.py             # Walk-forward simulation, capital tracking, reporting
├── backtest_v25.py         # Experimental variant: no smile filter + no divergence exit
└── results/                # Output CSVs and JSON summaries (git-ignored)
```

---

## Modules

### `config.py`
Central parameter store. Key knobs:

| Parameter | Default | Description |
|---|---|---|
| `RV_WINDOW` | 21 | Trailing trading days for realized vol |
| `Z_LOOKBACK_MONTHS` | 12 | Trailing months for z-score baseline |
| `MIN_HISTORY` | 8 | Minimum valid months before a ticker is scored |
| `Z_EXTREME_HIGH` | 1.5 | Smile gate: short excluded above this curvature/skew z |
| `CONVERGENCE_Z` | 0.1 | Take-profit: exit when IVRP z reverts inside ±this |
| `DIVERGENCE_Z` | 2.0 | Stop-loss: exit when smile worsens past this |
| `FORCE_EXIT_DTE` | 5 | Forced close below this many days to expiry |

### `signal_generation.py`
- Loads OptionMetrics daily parquets and CRSP price data
- Computes ATM straddle (call delta closest to 0.50 fixes strike; put taken at same strike)
- Computes 25-delta wings independently for skew/curvature (never traded)
- Builds rolling `IVRP + smile shape` history per ticker-month
- Z-scores each field against the ticker's own trailing window (no look-ahead — current month excluded from its own baseline)
- Applies joint eligibility gate and ranks into quintiles within each pool

### `backtest.py`
- Walk-forward simulation: 2015-01 → 2022-11 (CRSP ends 2022-12-31)
- Pre-2015 OM data skipped due to `exdate` encoding inconsistency (Saturday vs Friday)
- History bootstrap reaches 13 months before `BACKTEST_START`
- Positions priced via live bid/ask on held contract symbols; falls back to intrinsic payoff if no quote
- Capital tracked daily: long premiums + short Reg T margin (20% × underlying × 100)
- Reports: Sharpe, CAGR, max drawdown, win rate, exit reason breakdown, monthly IC (Spearman), quintile monotonicity

### `backtest_v25.py`
Experimental variant combining two ablations:
- **v23** (no divergence exit): Sharpe 0.80 → 0.90, but max drawdown worsened
- **v24** (no smile entry filter): Sharpe 0.80 → 0.85, max drawdown improved significantly (-20.8% → -8.3%), more trades (161 → 199)
- **v25** tests whether removing both compounds the gains, since v24's newly-enterable extreme-smile SHORT names were disproportionately cut by divergence exits

Monkey-patches `sg.eligibility` and `bt.check_exit` in memory only — does not touch source files on disk.

---

## Data Requirements

- **OptionMetrics**: daily parquets with columns `secid, date, symbol, exdate, cp_flag, strike_price, best_bid, best_offer, delta, impl_volatility, open_interest`
- **CRSP**: daily stock prices (`stock_price__1996_2022.parquet`)
- **Link table**: `secid → permno` mapping

Configure paths in `config.py` under `OPTIONMETRICS_BASE`, `CRSP_PATH`, `LINK_TABLE_PATH`.

---

## Running

```bash
# Build signal history and output current portfolio target
python3 signal_generation.py

# Full walk-forward backtest (2015–2022)
python3 backtest.py

# Experimental v25 variant
python3 backtest_v25.py
```

Outputs are written to `results/`:
- `trade_log_{RUN_LABEL}.csv` — every trade with entry/exit signals and P&L
- `returns_{RUN_LABEL}.csv` — monthly return series
- `performance_summary_{RUN_LABEL}.json` — full metrics including IC, quintile tables, year-by-year breakdown

---

## Baseline Results (v12, 2015–2022)

| Metric | Overall | Long Book | Short Book |
|---|---|---|---|
| Sharpe | 0.80 | — | — |
| Max Drawdown | −20.8% | — | — |
| N Trades | 161 | — | — |
| Short IC (mean) | — | — | ~0.05 |

*Full results in `results/performance_summary_v12_rerun_confirm.json` after running.*

---

## Design Notes

- **No look-ahead**: z-score baselines are frozen at entry time and stored per-position; daily monitoring re-standardizes live smile values against the same frozen baseline, not a rolling one.
- **Fixed-strike P&L**: exit prices mark the actual held call/put symbols (fixed strike), not a dynamically re-selected ATM option.
- **Quintile separation**: short and long pools are ranked independently, so a name can be in neither pool (filtered out), one, or both.
- **No external alpha**: signal is purely derived from OptionMetrics implied vols and CRSP realized vol — no earnings, macro, or sentiment data.
