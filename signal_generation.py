"""
signal_generation.py — Volatility Smile Relative-Value Straddle: signal engine
================================================================================
Pure signal engine — no exit logic, no P&L, no capital tracking (those live in
backtest.py). See README.md for the full design.

Builds a monthly history of IVRP (IV risk premium) + smile shape (skew,
curvature) per ticker from OptionMetrics (options) + CRSP (stock prices) +
link table (secid->permno), z-scores each against the ticker's own trailing
history, applies the joint smile-eligibility gate, and ranks into quintiles
within each eligible pool.

Edit START_DATE, END_DATE below, then: python3 signal_generation.py
"""

import json
import os
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    UNIVERSE, RV_WINDOW, Z_LOOKBACK_MONTHS, MIN_HISTORY, STD_FLOOR,
    Z_EXTREME_HIGH, Z_EXTREME_LOW, N_QUINTILES, LONG_QUINTILE, SHORT_QUINTILE,
    DELTA_MIN, DELTA_MAX, WING_DELTA_MIN, WING_DELTA_MAX, MAX_SPREAD_PCT_ENTRY,
    OPTIONMETRICS_BASE, CRSP_PATH, LINK_TABLE_PATH, OM_COLUMNS,
    SIGNAL_HISTORY_FILE, PORTFOLIO_TARGET, RESULTS_DIR,
)

UNIVERSE_SET = set(UNIVERSE)
TRADING_DAYS_PER_YEAR = 252


# ── Date helpers (same monthly 3rd-Friday cycle as straddle_momentum) ─────────

def get_third_friday(year: int, month: int) -> date:
    first     = date(year, month, 1)
    to_friday = (4 - first.weekday()) % 7
    return first + timedelta(days=to_friday) + timedelta(weeks=2)


def next_expiry(year: int, month: int) -> date:
    if month == 12:
        return get_third_friday(year + 1, 1)
    return get_third_friday(year, month + 1)


def months_back(n: int, from_date: date) -> tuple:
    """Return (year, month) n months before from_date."""
    y, m = from_date.year, from_date.month
    for _ in range(n):
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return y, m


def month_key(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


def months_range(start: date, end: date):
    y, m = start.year, start.month
    while date(y, m, 1) <= date(end.year, end.month, 1):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


# ── OptionMetrics helpers ───────────────────────────────────────────────────────

def om_path(d: date) -> str:
    subdir = os.path.join("1996-2015", "1996-2015") if d.year <= 2015 else str(d.year)
    return os.path.join(
        OPTIONMETRICS_BASE, subdir,
        f"optionmetrics_{d.isoformat()}_nonzeroOI.parquet",
    )


def load_om_day(d: date) -> Optional[pd.DataFrame]:
    """Load one day's OM file, filtered to universe tickers. Returns None if missing."""
    path = om_path(d)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path, columns=OM_COLUMNS)
    df = df.copy()
    df['ticker']     = df['symbol'].str.split(' ').str[0]
    df               = df[df['ticker'].isin(UNIVERSE_SET)]
    if df.empty:
        return None
    df['strike']     = df['strike_price'] / 1000.0
    df['mid']        = (df['best_bid'] + df['best_offer']) / 2.0
    df['spread_pct'] = np.where(df['mid'] > 0,
                           (df['best_offer'] - df['best_bid']) / df['mid'], np.nan)
    df['date']   = pd.to_datetime(df['date']).dt.date
    df['exdate'] = pd.to_datetime(df['exdate']).dt.date
    return df.reset_index(drop=True)


def _liquid_chain(om_day: pd.DataFrame, ticker: str, expiry: date) -> pd.DataFrame:
    """Common liquidity filter chain for a ticker/expiry slice of one day's OM file."""
    return om_day[
        (om_day['ticker']     == ticker) &
        (om_day['exdate']     == expiry) &
        (om_day['best_bid']   >  0) &
        (om_day['best_offer'] >  0) &
        (om_day['mid']        >= 0.05) &
        (om_day['spread_pct'] <= MAX_SPREAD_PCT_ENTRY)
    ]


def find_atm_straddle(om_day: pd.DataFrame, ticker: str, expiry: date) -> Optional[dict]:
    """
    Find the ATM straddle: call with delta closest to +0.50 fixes the strike,
    put is taken at that SAME strike (true straddle, not a strangle) regardless
    of its own delta — the deviation from -0.50 IS the skew signal.
    """
    sub = _liquid_chain(om_day, ticker, expiry)

    calls = sub[
        (sub['cp_flag'] == 'C') &
        sub['delta'].between(DELTA_MIN, DELTA_MAX)
    ].dropna(subset=['delta', 'impl_volatility']).copy()

    puts_all = sub[sub['cp_flag'] == 'P'].dropna(subset=['delta', 'impl_volatility']).copy()

    if calls.empty or puts_all.empty:
        return None

    calls['d_diff'] = (calls['delta'] - 0.50).abs()
    best_call = calls.nsmallest(1, 'd_diff').iloc[0]
    strike    = best_call['strike']

    exact_puts = puts_all[puts_all['strike'] == strike]
    if exact_puts.empty:
        puts_all['s_diff'] = (puts_all['strike'] - strike).abs()
        best_put = puts_all.nsmallest(1, 's_diff').iloc[0]
    else:
        best_put = exact_puts.iloc[0]

    return {
        'ticker':       ticker,
        'expiry':       expiry,
        'strike':       float(strike),
        'call_symbol':  str(best_call['symbol']),
        'put_symbol':   str(best_put['symbol']),
        'secid':        float(best_call['secid']) if 'secid' in best_call.index else None,
        'call_delta':   float(best_call['delta']),
        'put_delta':    float(best_put['delta']),
        'call_iv':      float(best_call['impl_volatility']),
        'put_iv':       float(best_put['impl_volatility']),
        'atm_iv':       float((best_call['impl_volatility'] + best_put['impl_volatility']) / 2.0),
        'call_bid':     float(best_call['best_bid']),
        'call_ask':     float(best_call['best_offer']),
        'put_bid':      float(best_put['best_bid']),
        'put_ask':      float(best_put['best_offer']),
    }


def find_wing(om_day: pd.DataFrame, ticker: str, expiry: date, side: str) -> Optional[dict]:
    """
    Find the 25-delta wing option on one side ('call' or 'put'), independent
    of the ATM strike — used only for skew/curvature, never traded.
    """
    sub = _liquid_chain(om_day, ticker, expiry)
    flag = 'C' if side == 'call' else 'P'
    target = WING_DELTA_MIN + (WING_DELTA_MAX - WING_DELTA_MIN) / 2.0   # ~0.25
    if side == 'put':
        band = sub[(sub['cp_flag'] == 'P') &
                    sub['delta'].between(-WING_DELTA_MAX, -WING_DELTA_MIN)].dropna(
                        subset=['delta', 'impl_volatility']).copy()
        if band.empty:
            return None
        band['d_diff'] = (band['delta'] - (-target)).abs()
    else:
        band = sub[(sub['cp_flag'] == 'C') &
                    sub['delta'].between(WING_DELTA_MIN, WING_DELTA_MAX)].dropna(
                        subset=['delta', 'impl_volatility']).copy()
        if band.empty:
            return None
        band['d_diff'] = (band['delta'] - target).abs()

    best = band.nsmallest(1, 'd_diff').iloc[0]
    return {'delta': float(best['delta']), 'iv': float(best['impl_volatility'])}


def compute_smile_snapshot(om_day: pd.DataFrame, ticker: str, expiry: date) -> Optional[dict]:
    """
    One day's full smile snapshot for a ticker: ATM straddle (tradeable) +
    put_skew/call_skew/curvature (diagnostic only, never traded).
    """
    atm = find_atm_straddle(om_day, ticker, expiry)
    if atm is None:
        return None
    put_wing  = find_wing(om_day, ticker, expiry, 'put')
    call_wing = find_wing(om_day, ticker, expiry, 'call')
    if put_wing is None or call_wing is None:
        return None

    atm_iv    = atm['atm_iv']
    put_skew  = put_wing['iv']  - atm_iv
    call_skew = call_wing['iv'] - atm_iv
    curvature = (put_wing['iv'] + call_wing['iv']) / 2.0 - atm_iv

    return {
        **atm,
        'put_25d_iv':  put_wing['iv'],
        'call_25d_iv': call_wing['iv'],
        'put_skew':    put_skew,
        'call_skew':   call_skew,
        'curvature':   curvature,
    }


# ── CRSP + link table helpers (same as straddle_momentum) ─────────────────────

def load_crsp() -> dict:
    """Load CRSP daily prices, return {permno: pd.Series(price, index=date)}."""
    print("  Loading CRSP prices...")
    df = pd.read_parquet(CRSP_PATH, columns=['permno', 'date', 'prc'])
    df['date']  = pd.to_datetime(df['date']).dt.date
    df['price'] = df['prc'].abs()
    df          = df[df['price'] > 0].copy()
    crsp_idx    = {
        int(permno): grp.set_index('date')['price'].sort_index()
        for permno, grp in df.groupby('permno')
    }
    print(f"  CRSP loaded: {len(crsp_idx):,} permnos")
    return crsp_idx


def load_link_table() -> dict:
    """Returns {secid: [(sdate, edate, permno), ...]} for fast date-ranged lookup."""
    from collections import defaultdict
    link = pd.read_parquet(LINK_TABLE_PATH)
    link = link.dropna(subset=['permno']).copy()
    idx  = defaultdict(list)
    for _, row in link.iterrows():
        idx[float(row['secid'])].append(
            (row['sdate'], row['edate'], int(row['permno']))
        )
    return dict(idx)


def secid_to_permno(secid: float, as_of: date, link: dict) -> Optional[int]:
    ts = pd.Timestamp(as_of)
    for sdate, edate, permno in link.get(secid, []):
        if sdate <= ts <= edate:
            return permno
    return None


def realized_vol(crsp_idx: dict, permno: int, as_of: date,
                  window: int = RV_WINDOW) -> Optional[float]:
    """Annualized close-to-close realized vol over the trailing `window` trading days."""
    series = crsp_idx.get(permno)
    if series is None:
        return None
    avail = series[series.index <= as_of]
    if len(avail) < window + 1:
        return None
    prices   = avail.iloc[-(window + 1):].values
    log_rets = np.diff(np.log(prices))
    return float(np.std(log_rets, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


# ── Monthly history build ───────────────────────────────────────────────────────

def build_history(start: date, end: date, crsp_idx: dict, link: dict) -> dict:
    """
    Walk forward month by month, building
    {ticker: {YYYY-MM: {atm_iv, rv, ivrp, put_skew, call_skew, curvature}}}.
    """
    history = {t: {} for t in UNIVERSE}

    for year, month in months_range(start, end):
        entry_date  = get_third_friday(year, month)
        expiry_date = next_expiry(year, month)
        key         = month_key(year, month)

        om_entry = load_om_day(entry_date)
        if om_entry is None:
            print(f"  {key}  entry={entry_date}  OM file missing — skip")
            continue

        day_secids = om_entry[['ticker', 'secid']].drop_duplicates('ticker')
        day_secids = day_secids[day_secids['ticker'].isin(UNIVERSE_SET)]

        filled = 0
        for _, row in day_secids.iterrows():
            ticker = row['ticker']
            if key in history[ticker]:
                continue

            snap = compute_smile_snapshot(om_entry, ticker, expiry_date)
            if snap is None:
                continue

            permno = secid_to_permno(float(row['secid']), entry_date, link)
            if permno is None:
                continue

            rv = realized_vol(crsp_idx, permno, entry_date)
            if rv is None:
                continue

            history[ticker][key] = {
                'atm_iv':    round(snap['atm_iv'], 6),
                'rv':        round(rv, 6),
                'ivrp':      round(snap['atm_iv'] - rv, 6),
                'put_skew':  round(snap['put_skew'], 6),
                'call_skew': round(snap['call_skew'], 6),
                'curvature': round(snap['curvature'], 6),
            }
            filled += 1

        print(f"  {key}  entry={entry_date}  expiry={expiry_date}  filled={filled}")

    return history


# ── Z-scoring + eligibility + ranking ───────────────────────────────────────────

def _trailing_values(t_hist: dict, field: str, as_of: date) -> list:
    """Trailing Z_LOOKBACK_MONTHS values of `field` for lags t-1..t-Z_LOOKBACK_MONTHS
    (excludes the current month itself — no look-ahead, no self-inclusion)."""
    vals = []
    for lag in range(1, Z_LOOKBACK_MONTHS + 1):
        y, m = months_back(lag, as_of)
        rec  = t_hist.get(month_key(y, m))
        if rec is not None and np.isfinite(rec[field]):
            vals.append(rec[field])
    return vals


def compute_zscore(t_hist: dict, field: str, as_of: date, current_value: float) -> Optional[float]:
    trailing = _trailing_values(t_hist, field, as_of)
    if len(trailing) < MIN_HISTORY:
        return None
    std = max(float(np.std(trailing, ddof=1)), STD_FLOOR)
    return (current_value - float(np.mean(trailing))) / std


def compute_ticker_zscores(history: dict, ticker: str, as_of: date) -> Optional[dict]:
    """z-scores for one ticker as of `as_of` (the current month's own record must exist)."""
    t_hist = history.get(ticker, {})
    cur    = t_hist.get(month_key(as_of.year, as_of.month))
    if cur is None:
        return None

    ivrp_z      = compute_zscore(t_hist, 'ivrp',      as_of, cur['ivrp'])
    put_skew_z  = compute_zscore(t_hist, 'put_skew',  as_of, cur['put_skew'])
    call_skew_z = compute_zscore(t_hist, 'call_skew', as_of, cur['call_skew'])
    curvature_z = compute_zscore(t_hist, 'curvature', as_of, cur['curvature'])

    if None in (ivrp_z, put_skew_z, call_skew_z, curvature_z):
        return None

    return {
        'ivrp_z': ivrp_z, 'put_skew_z': put_skew_z,
        'call_skew_z': call_skew_z, 'curvature_z': curvature_z,
    }


def eligibility(z: dict) -> tuple:
    """
    (short_eligible, long_eligible) — see README §"Joint eligibility gate".
    Short-eligible: wings aren't pricing real tail risk (curvature and both
    skews below the extreme-high threshold).
    Long-eligible: excluded only if the smile is flat relative to its own norm
    (curvature extremely low AND both skews unremarkable) — a dead, un-traded name.
    """
    curv_z, put_z, call_z = z['curvature_z'], z['put_skew_z'], z['call_skew_z']

    short_eligible = (curv_z < Z_EXTREME_HIGH and
                      abs(put_z) < Z_EXTREME_HIGH and
                      abs(call_z) < Z_EXTREME_HIGH)

    flat = (curv_z < Z_EXTREME_LOW and
            abs(put_z) < Z_EXTREME_HIGH and
            abs(call_z) < Z_EXTREME_HIGH)
    long_eligible = not flat

    return short_eligible, long_eligible


def rank_quintiles(scores: dict) -> dict:
    """Returns {ticker: quintile 1-N_QUINTILES}, ranked ascending by score."""
    if not scores:
        return {}
    tickers = list(scores.keys())
    values  = np.array([scores[t] for t in tickers])
    ranks   = pd.Series(values).rank(pct=True).values
    return {t: int(np.ceil(r * N_QUINTILES)) for t, r in zip(tickers, ranks)}


def compute_monthly_signal(history: dict, as_of: date) -> dict:
    """
    For every ticker with enough history as of `as_of`, compute z-scores +
    eligibility, then rank ivrp_z within each eligible pool separately.
    Returns {ticker: {**zscores, short_eligible, long_eligible, short_quintile?, long_quintile?}}.
    """
    per_ticker = {}
    for ticker in UNIVERSE:
        z = compute_ticker_zscores(history, ticker, as_of)
        if z is None:
            continue
        short_elig, long_elig = eligibility(z)
        per_ticker[ticker] = {**z, 'short_eligible': short_elig, 'long_eligible': long_elig}

    short_pool = {t: v['ivrp_z'] for t, v in per_ticker.items() if v['short_eligible']}
    long_pool  = {t: v['ivrp_z'] for t, v in per_ticker.items() if v['long_eligible']}

    short_quintiles = rank_quintiles(short_pool)
    long_quintiles  = rank_quintiles(long_pool)

    for t, q in short_quintiles.items():
        per_ticker[t]['short_quintile'] = q
    for t, q in long_quintiles.items():
        per_ticker[t]['long_quintile'] = q

    return per_ticker


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 修改这里来控制历史构建范围 ────────────────────────────────────────────
    # Z-scoring needs Z_LOOKBACK_MONTHS (12) trailing months with >= MIN_HISTORY
    # (8) valid points, so history must start well before the desired signal
    # start date — same logic as straddle_momentum's bootstrap window.
    START_DATE = date(2009, 1, 1)
    END_DATE   = date(2023, 8, 31)
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("  Volatility Smile Relative-Value Straddle — Signal Generation")
    print(f"  History range : {START_DATE} → {END_DATE}")
    print(f"  Universe      : {len(UNIVERSE)} tickers")
    print(f"{'='*60}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    crsp_idx = load_crsp()
    link     = load_link_table()

    print("\n  Building IVRP + smile history...")
    history = build_history(START_DATE, END_DATE, crsp_idx, link)

    total = sum(len(v) for v in history.values())
    print(f"\n  History built: {total} ticker-months")

    with open(SIGNAL_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved → {SIGNAL_HISTORY_FILE}")

    # Compute signal as of the most recent available month
    last_year, last_month = END_DATE.year, END_DATE.month
    as_of = get_third_friday(last_year, last_month)
    signal = compute_monthly_signal(history, as_of)

    long_tickers  = sorted(
        [t for t, v in signal.items() if v.get('long_quintile') == LONG_QUINTILE],
        key=lambda t: signal[t]['ivrp_z'])
    short_tickers = sorted(
        [t for t, v in signal.items() if v.get('short_quintile') == SHORT_QUINTILE],
        key=lambda t: -signal[t]['ivrp_z'])

    print(f"\n  Scored tickers        : {len(signal)}")
    print(f"  Short-eligible pool   : {sum(1 for v in signal.values() if v['short_eligible'])}")
    print(f"  Long-eligible  pool   : {sum(1 for v in signal.values() if v['long_eligible'])}")

    print(f"\n  LONG  Q{LONG_QUINTILE} ({len(long_tickers)} tickers, cheapest vol):")
    for t in long_tickers:
        v = signal[t]
        print(f"    {t:<6}  ivrp_z={v['ivrp_z']:+.2f}  "
              f"put_skew_z={v['put_skew_z']:+.2f}  call_skew_z={v['call_skew_z']:+.2f}  "
              f"curvature_z={v['curvature_z']:+.2f}")

    print(f"\n  SHORT Q{SHORT_QUINTILE} ({len(short_tickers)} tickers, most expensive vol):")
    for t in short_tickers:
        v = signal[t]
        print(f"    {t:<6}  ivrp_z={v['ivrp_z']:+.2f}  "
              f"put_skew_z={v['put_skew_z']:+.2f}  call_skew_z={v['call_skew_z']:+.2f}  "
              f"curvature_z={v['curvature_z']:+.2f}")

    out = {
        "generated_at": as_of.isoformat(),
        "long":  [{"ticker": t, **{k: round(v, 4) for k, v in signal[t].items()
                                    if isinstance(v, float)}} for t in long_tickers],
        "short": [{"ticker": t, **{k: round(v, 4) for k, v in signal[t].items()
                                    if isinstance(v, float)}} for t in short_tickers],
    }
    with open(PORTFOLIO_TARGET, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Portfolio target → {PORTFOLIO_TARGET}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
