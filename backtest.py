"""
backtest.py — Volatility Smile Relative-Value Straddle: walk-forward backtest
================================================================================
See README.md for the full design. Reuses signal_generation.py's data loaders
and signal engine; this file adds simulation (entry/exit/capital) and
reporting (performance, IC, quintile monotonicity) — no signal logic lives here.

Edit the USER CONFIG block below, then: python3 backtest.py
"""

import json
import os
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

import signal_generation as sg
from config import (
    N_QUINTILES, LONG_QUINTILE, SHORT_QUINTILE,
    CONVERGENCE_Z, DIVERGENCE_Z, FORCE_EXIT_DTE,
    SHORT_MARGIN_PCT, RESULTS_DIR,
)

# ── USER CONFIG ──────────────────────────────────────────────────────────────
# CRSP coverage ends 2022-12-31; last entry month must leave room for an
# expiry-month CRSP price, so the practical ceiling is ~2022-11.
# 2010-2014 excluded: pre-2015 OM data in this dataset records standard
# monthly `exdate` as the Saturday after the 3rd Friday (not the Friday
# itself, as 2016+ per-year files do) — see README caveats. Not fixed, just
# skipped; history bootstrap still reaches back into 2014 so the first few
# months of 2015 will be thin (ramping up) until 8 valid trailing months exist.
BACKTEST_START = date(2015, 1, 1)
BACKTEST_END   = date(2022, 11, 30)
RUN_LABEL      = "v12_rerun_confirm"
RUN_QUINTILE_DIAGNOSTIC = True   # heavier pass: static returns for every eligible name, all quintiles
# ─────────────────────────────────────────────────────────────────────────────

HISTORY_BOOTSTRAP_MONTHS = 13   # history must start this many months before BACKTEST_START


def _subtract_months(d: date, n: int) -> date:
    y, m = d.year, d.month
    for _ in range(n):
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return date(y, m, 1)


def _crsp_price_on_or_before(crsp_idx: dict, permno: int, target: date):
    series = crsp_idx.get(permno)
    if series is None:
        return None
    avail = series[series.index <= target]
    return float(avail.iloc[-1]) if not avail.empty else None


def get_baselines(history: dict, ticker: str, as_of: date):
    """Frozen entry-time (mean, std) per field — same trailing window/floor as
    sg.compute_zscore, but returns the baseline itself so daily monitoring can
    re-standardize fresh raw values against it (see README §4)."""
    t_hist = history.get(ticker, {})
    out = {}
    for field in ('ivrp', 'put_skew', 'call_skew', 'curvature'):
        trailing = sg._trailing_values(t_hist, field, as_of)
        if len(trailing) < sg.MIN_HISTORY:
            return None
        std = max(float(np.std(trailing, ddof=1)), sg.STD_FLOOR)
        out[field] = (float(np.mean(trailing)), std)
    return out


def get_quote_by_symbol(om_day: pd.DataFrame, symbol: str):
    row = om_day[om_day['symbol'] == symbol]
    if row.empty:
        return None
    r = row.iloc[0]
    bid, ask = float(r['best_bid']), float(r['best_offer'])
    if bid <= 0 or ask <= 0:
        return None
    return {'bid': bid, 'ask': ask}


# ── Position lifecycle ──────────────────────────────────────────────────────────

def open_position(ticker: str, leg: str, om_entry: pd.DataFrame, entry_date: date,
                   expiry_date: date, signal_row: dict, baseline: dict, permno: int):
    strad = sg.find_atm_straddle(om_entry, ticker, expiry_date)
    if strad is None:
        return None
    entry_cost = (strad['call_ask'] + strad['put_ask']) if leg == 'long' \
        else (strad['call_bid'] + strad['put_bid'])
    if entry_cost <= 0:
        return None
    return {
        'ticker': ticker, 'leg': leg, 'permno': permno,
        'strike': strad['strike'],
        'call_symbol': strad['call_symbol'], 'put_symbol': strad['put_symbol'],
        'entry_date': entry_date, 'expiry_date': expiry_date,
        'entry_cost': entry_cost, 'entry_underlying': None,
        'entry_ivrp_z': signal_row['ivrp_z'],
        'entry_put_skew_z': signal_row['put_skew_z'],
        'entry_call_skew_z': signal_row['call_skew_z'],
        'entry_curvature_z': signal_row['curvature_z'],
        'baseline': baseline,
    }


def check_exit(pos: dict, om_day: pd.DataFrame, crsp_idx: dict, cur_date: date):
    """Returns an exit reason string or None. Re-derives ATM/wings fresh each
    day (delta-based, not the fixed held strike) — see README §4 for why."""
    snap = sg.compute_smile_snapshot(om_day, pos['ticker'], pos['expiry_date'])
    if snap is None:
        return None
    rv = sg.realized_vol(crsp_idx, pos['permno'], cur_date)
    if rv is None:
        return None

    m, s = pos['baseline']['ivrp'];      ivrp_z      = (snap['atm_iv'] - rv - m) / s
    m, s = pos['baseline']['put_skew'];  put_skew_z  = (snap['put_skew']  - m) / s
    m, s = pos['baseline']['call_skew']; call_skew_z = (snap['call_skew'] - m) / s
    m, s = pos['baseline']['curvature']; curvature_z = (snap['curvature'] - m) / s

    if pos['leg'] == 'short':
        if ivrp_z < CONVERGENCE_Z:
            return 'convergence'
        if curvature_z > DIVERGENCE_Z or abs(put_skew_z) > DIVERGENCE_Z or abs(call_skew_z) > DIVERGENCE_Z:
            return 'divergence'
    else:
        if ivrp_z > -CONVERGENCE_Z:
            return 'convergence'
        if curvature_z < -DIVERGENCE_Z and abs(put_skew_z) < DIVERGENCE_Z and abs(call_skew_z) < DIVERGENCE_Z:
            return 'divergence'
    return None


def close_position(pos: dict, exit_date: date, exit_reason: str,
                    om_day, crsp_idx: dict):
    """Marks the ACTUAL held contracts (fixed strike/symbols) via bid/ask;
    falls back to intrinsic payoff only if no quote exists that day."""
    call_q = get_quote_by_symbol(om_day, pos['call_symbol']) if om_day is not None else None
    put_q  = get_quote_by_symbol(om_day, pos['put_symbol'])  if om_day is not None else None

    if call_q and put_q:
        if pos['leg'] == 'long':
            exit_value = call_q['bid'] + put_q['bid']
            pnl_pct = (exit_value - pos['entry_cost']) / pos['entry_cost']
        else:
            exit_value = call_q['ask'] + put_q['ask']
            pnl_pct = (pos['entry_cost'] - exit_value) / pos['entry_cost']
        priced_via = 'quote'
    else:
        price = _crsp_price_on_or_before(crsp_idx, pos['permno'], exit_date)
        if price is None:
            return None
        payoff = max(price - pos['strike'], 0.0) + max(pos['strike'] - price, 0.0)
        if pos['leg'] == 'long':
            pnl_pct = (payoff - pos['entry_cost']) / pos['entry_cost']
        else:
            pnl_pct = (pos['entry_cost'] - payoff) / pos['entry_cost']
        exit_value = payoff
        priced_via = 'intrinsic_fallback'

    return {
        'ticker': pos['ticker'], 'leg': pos['leg'],
        'entry_date': pos['entry_date'].isoformat(),
        'exit_date': exit_date.isoformat(),
        'expiry_date': pos['expiry_date'].isoformat(),
        'exit_reason': exit_reason, 'priced_via': priced_via,
        'entry_ivrp_z': pos['entry_ivrp_z'],
        'entry_put_skew_z': pos['entry_put_skew_z'],
        'entry_call_skew_z': pos['entry_call_skew_z'],
        'entry_curvature_z': pos['entry_curvature_z'],
        'entry_cost': pos['entry_cost'], 'exit_value': exit_value,
        'pnl_pct': pnl_pct, 'entry_underlying': pos['entry_underlying'],
    }


# ── Monthly cohort simulation (the actual traded book: Q1 long / Q5 short) ────

def simulate_month(history: dict, year: int, month: int, crsp_idx: dict, link: dict):
    entry_date  = sg.get_third_friday(year, month)
    expiry_date = sg.next_expiry(year, month)

    om_entry = sg.load_om_day(entry_date)
    if om_entry is None:
        return []

    signal = sg.compute_monthly_signal(history, entry_date)
    long_tickers  = [t for t, v in signal.items() if v.get('long_quintile')  == LONG_QUINTILE]
    short_tickers = [t for t, v in signal.items() if v.get('short_quintile') == SHORT_QUINTILE]

    day_secids = (om_entry[['ticker', 'secid']].drop_duplicates('ticker')
                  .set_index('ticker')['secid'])

    positions = {}
    for ticker, leg in [(t, 'long') for t in long_tickers] + [(t, 'short') for t in short_tickers]:
        if ticker not in day_secids.index:
            continue
        permno = sg.secid_to_permno(float(day_secids[ticker]), entry_date, link)
        if permno is None:
            continue
        baseline = get_baselines(history, ticker, entry_date)
        if baseline is None:
            continue
        pos = open_position(ticker, leg, om_entry, entry_date, expiry_date,
                             signal[ticker], baseline, permno)
        if pos is None:
            continue
        entry_px = _crsp_price_on_or_before(crsp_idx, permno, entry_date)
        if entry_px is None:
            continue
        pos['entry_underlying'] = entry_px
        positions[ticker] = pos

    trades = []
    cur = entry_date + timedelta(days=1)
    while positions and cur <= expiry_date:
        om_day = sg.load_om_day(cur)
        dte = (expiry_date - cur).days
        if om_day is not None:
            for ticker in list(positions.keys()):
                pos = positions[ticker]
                exit_reason = check_exit(pos, om_day, crsp_idx, cur)
                if exit_reason is None and dte <= FORCE_EXIT_DTE:
                    exit_reason = 'forced_dte'
                if exit_reason:
                    closed = close_position(pos, cur, exit_reason, om_day, crsp_idx)
                    if closed is not None:
                        trades.append(closed)
                        del positions[ticker]
        else:
            for ticker in list(positions.keys()):
                if dte <= FORCE_EXIT_DTE:
                    closed = close_position(positions[ticker], cur, 'forced_dte', None, crsp_idx)
                    if closed is not None:
                        trades.append(closed)
                        del positions[ticker]
        cur += timedelta(days=1)

    if positions:
        om_expiry = sg.load_om_day(expiry_date)
        for ticker, pos in positions.items():
            closed = close_position(pos, expiry_date, 'held_to_expiry', om_expiry, crsp_idx)
            if closed is not None:
                trades.append(closed)

    return trades


# ── Quintile monotonicity diagnostic (static hold-to-expiry, all eligible names) ─

def simulate_month_quintiles(history: dict, year: int, month: int, crsp_idx: dict, link: dict):
    """Static (hold-to-expiry) returns for EVERY eligible name in each pool,
    bucketed by quintile — cheaper than the daily-monitored book above since
    it skips the day-by-day exit loop; used only for the monotonicity check."""
    entry_date  = sg.get_third_friday(year, month)
    expiry_date = sg.next_expiry(year, month)

    om_entry = sg.load_om_day(entry_date)
    if om_entry is None:
        return []
    om_expiry = sg.load_om_day(expiry_date)

    signal = sg.compute_monthly_signal(history, entry_date)
    day_secids = (om_entry[['ticker', 'secid']].drop_duplicates('ticker')
                  .set_index('ticker')['secid'])

    records = []
    for ticker, v in signal.items():
        if ticker not in day_secids.index:
            continue
        permno = sg.secid_to_permno(float(day_secids[ticker]), entry_date, link)
        if permno is None:
            continue
        strad = sg.find_atm_straddle(om_entry, ticker, expiry_date)
        if strad is None:
            continue

        for pool_flag, leg, qfield in [('short_eligible', 'short', 'short_quintile'),
                                        ('long_eligible',  'long',  'long_quintile')]:
            if not v.get(pool_flag) or qfield not in v:
                continue

            entry_cost = (strad['call_ask'] + strad['put_ask']) if leg == 'long' \
                else (strad['call_bid'] + strad['put_bid'])
            if entry_cost <= 0:
                continue

            call_q = get_quote_by_symbol(om_expiry, strad['call_symbol']) if om_expiry is not None else None
            put_q  = get_quote_by_symbol(om_expiry, strad['put_symbol'])  if om_expiry is not None else None
            if call_q and put_q:
                if leg == 'long':
                    exit_value = call_q['bid'] + put_q['bid']
                    pnl = (exit_value - entry_cost) / entry_cost
                else:
                    exit_value = call_q['ask'] + put_q['ask']
                    pnl = (entry_cost - exit_value) / entry_cost
            else:
                price = _crsp_price_on_or_before(crsp_idx, permno, expiry_date)
                if price is None:
                    continue
                payoff = max(price - strad['strike'], 0.0) + max(strad['strike'] - price, 0.0)
                pnl = (payoff - entry_cost) / entry_cost if leg == 'long' \
                    else (entry_cost - payoff) / entry_cost

            records.append({'ticker': ticker, 'leg': leg, 'quintile': v[qfield], 'pnl_pct': pnl})

    return records


def quintile_table(records: list, leg: str):
    sub = [r for r in records if r['leg'] == leg]
    out = {}
    for q in range(1, N_QUINTILES + 1):
        vals = [r['pnl_pct'] for r in sub if r['quintile'] == q]
        out[str(q)] = {'n': len(vals), 'avg_return': float(np.mean(vals)) if vals else None}
    return out


# ── Capital requirements (derived, peak-day methodology — README §5) ──────────

def track_capital(trades: list):
    cap_long, cap_short = defaultdict(float), defaultdict(float)
    for tr in trades:
        entry_d = date.fromisoformat(tr['entry_date'])
        exit_d  = date.fromisoformat(tr['exit_date'])
        cur = entry_d
        while cur <= exit_d:
            key = cur.isoformat()
            if tr['leg'] == 'long':
                cap_long[key] += tr['entry_cost'] * 100
            else:
                cap_short[key] += SHORT_MARGIN_PCT * tr['entry_underlying'] * 100
            cur += timedelta(days=1)

    all_dates = set(cap_long) | set(cap_short)
    if not all_dates:
        return {'peak_date': None, 'peak_total': 0.0, 'peak_long': 0.0,
                'peak_short': 0.0, 'p95_daily': 0.0, 'avg_daily': 0.0,
                'initial_capital': 0.0}

    totals = {d: cap_long.get(d, 0.0) + cap_short.get(d, 0.0) for d in all_dates}
    peak_date  = max(totals, key=totals.get)
    peak_total = totals[peak_date]
    vals = sorted(totals.values())
    p95  = vals[int(0.95 * len(vals))]
    avg  = sum(vals) / len(vals)

    return {
        'peak_date': peak_date, 'peak_total': peak_total,
        'peak_long': cap_long.get(peak_date, 0.0), 'peak_short': cap_short.get(peak_date, 0.0),
        'p95_daily': p95, 'avg_daily': avg, 'initial_capital': peak_total * 1.20,
    }


# ── Performance metrics ─────────────────────────────────────────────────────────

def build_return_series(trades: list, capital: float) -> pd.Series:
    """Monthly $ P&L (bucketed by entry month) / derived capital -> return series."""
    monthly = defaultdict(float)
    for tr in trades:
        entry_d = date.fromisoformat(tr['entry_date'])
        key = sg.month_key(entry_d.year, entry_d.month)
        monthly[key] += tr['pnl_pct'] * tr['entry_cost'] * 100
    if not monthly or capital <= 0:
        return pd.Series(dtype=float)
    months = sorted(monthly.keys())
    return pd.Series({m: monthly[m] / capital for m in months})


def performance_metrics(returns: pd.Series) -> dict:
    if returns.empty:
        return {'total_return': None, 'cagr': None, 'sharpe': None,
                'max_drawdown': None, 'n_months': 0}
    total_return = float((1 + returns).prod() - 1)
    n_years = len(returns) / 12.0
    cagr   = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else None
    std    = returns.std(ddof=1)
    sharpe = float(returns.mean() / std * np.sqrt(12)) if std and std > 0 else None
    equity = (1 + returns).cumprod()
    dd     = equity / equity.cummax() - 1
    return {'total_return': total_return, 'cagr': cagr, 'sharpe': sharpe,
            'max_drawdown': float(dd.min()), 'n_months': len(returns)}


def trade_stats(trades: list) -> dict:
    if not trades:
        return {'n_trades': 0, 'win_rate': None, 'avg_return': None}
    n = len(trades)
    wins = sum(1 for t in trades if t['pnl_pct'] > 0)
    return {'n_trades': n, 'win_rate': wins / n,
            'avg_return': float(np.mean([t['pnl_pct'] for t in trades]))}


def exit_reason_breakdown(trades: list) -> dict:
    out = {}
    for reason in ('convergence', 'divergence', 'forced_dte', 'held_to_expiry'):
        sub = [t for t in trades if t['exit_reason'] == reason]
        out[reason] = {
            'n': len(sub),
            'share': len(sub) / len(trades) if trades else 0.0,
            'avg_return': float(np.mean([t['pnl_pct'] for t in sub])) if sub else None,
        }
    return out


def _spearman(a, b) -> float:
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    return float(ra.corr(rb))


def compute_ic(trades: list, leg: str) -> dict:
    """Monthly Spearman IC between a conviction score and realized return.
    Score = ivrp_z for short (higher = more expensive = expect more +return),
    -ivrp_z for long (so higher score = higher expected return on both legs)."""
    sub = [t for t in trades if t['leg'] == leg]
    by_month = defaultdict(list)
    for t in sub:
        entry_d = date.fromisoformat(t['entry_date'])
        key = sg.month_key(entry_d.year, entry_d.month)
        score = t['entry_ivrp_z'] if leg == 'short' else -t['entry_ivrp_z']
        by_month[key].append((score, t['pnl_pct']))

    ics = []
    for pairs in by_month.values():
        if len(pairs) < 3:
            continue
        scores, rets = zip(*pairs)
        if len(set(scores)) < 2:
            continue
        rho = _spearman(scores, rets)
        if np.isfinite(rho):
            ics.append(rho)

    if not ics:
        return {'ic_mean': None, 'ic_tstat': None, 'n_months': 0}
    ic_mean = float(np.mean(ics))
    ic_std  = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
    tstat   = ic_mean / (ic_std / np.sqrt(len(ics))) if ic_std > 0 else None
    return {'ic_mean': ic_mean, 'ic_tstat': tstat, 'n_months': len(ics)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*70}")
    print("  Volatility Smile Relative-Value Straddle — Backtest")
    print(f"  Range: {BACKTEST_START} → {BACKTEST_END}   Run label: {RUN_LABEL}")
    print(f"{'='*70}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    crsp_idx = sg.load_crsp()
    link     = sg.load_link_table()

    history_start = _subtract_months(BACKTEST_START, HISTORY_BOOTSTRAP_MONTHS)
    print(f"\n  Building signal history from {history_start}...")
    history = sg.build_history(history_start, BACKTEST_END, crsp_idx, link)

    print("\n  Simulating traded book (Q1 long / Q5 short, daily-monitored)...")
    all_trades = []
    for year, month in sg.months_range(BACKTEST_START, BACKTEST_END):
        trades = simulate_month(history, year, month, crsp_idx, link)
        all_trades.extend(trades)
        print(f"    {sg.month_key(year, month)}: {len(trades)} trades closed")

    capital_info = track_capital(all_trades)
    initial_capital = capital_info['initial_capital']

    long_trades  = [t for t in all_trades if t['leg'] == 'long']
    short_trades = [t for t in all_trades if t['leg'] == 'short']

    overall_returns = build_return_series(all_trades, initial_capital)
    long_returns    = build_return_series(long_trades, initial_capital)
    short_returns   = build_return_series(short_trades, initial_capital)

    overall_perf = {**performance_metrics(overall_returns), **trade_stats(all_trades)}
    long_perf    = {**performance_metrics(long_returns), **trade_stats(long_trades)}
    short_perf   = {**performance_metrics(short_returns), **trade_stats(short_trades)}

    years = sorted(set(date.fromisoformat(t['entry_date']).year for t in all_trades))
    by_year = {}
    for y in years:
        yr_trades  = [t for t in all_trades if date.fromisoformat(t['entry_date']).year == y]
        yr_returns = build_return_series(yr_trades, initial_capital)
        by_year[y] = {**performance_metrics(yr_returns), **trade_stats(yr_trades)}

    exit_breakdown = exit_reason_breakdown(all_trades)
    ic_short = compute_ic(all_trades, 'short')
    ic_long  = compute_ic(all_trades, 'long')

    quintile_short, quintile_long = {}, {}
    if RUN_QUINTILE_DIAGNOSTIC:
        print("\n  Running quintile monotonicity diagnostic (static, all eligible names)...")
        quintile_records = []
        for year, month in sg.months_range(BACKTEST_START, BACKTEST_END):
            quintile_records.extend(simulate_month_quintiles(history, year, month, crsp_idx, link))
        quintile_short = quintile_table(quintile_records, 'short')
        quintile_long  = quintile_table(quintile_records, 'long')

    # ── Console summary ──────────────────────────────────────────────────────
    print(f"\n  {'='*66}")
    print("  CAPITAL REQUIREMENTS  (1 contract per position)")
    print(f"  {'='*66}")
    print(f"    Peak date            : {capital_info['peak_date']}")
    print(f"    Long  premiums       : ${capital_info['peak_long']:>12,.0f}")
    print(f"    Short margin (RegT)  : ${capital_info['peak_short']:>12,.0f}")
    print(f"    PEAK TOTAL           : ${capital_info['peak_total']:>12,.0f}")
    print(f"    95th-pct daily total : ${capital_info['p95_daily']:>12,.0f}")
    print(f"    Average daily total  : ${capital_info['avg_daily']:>12,.0f}")
    print(f"    INITIAL_CAPITAL      : ${initial_capital:>12,.0f}  (peak x 1.20)")

    def _print_perf(label, perf):
        print(f"\n  {label}")
        print(f"    n_trades={perf.get('n_trades')}  win_rate={perf.get('win_rate')}  "
              f"avg_return={perf.get('avg_return')}")
        print(f"    total_return={perf.get('total_return')}  cagr={perf.get('cagr')}  "
              f"sharpe={perf.get('sharpe')}  max_dd={perf.get('max_drawdown')}")

    _print_perf("OVERALL (long+short combined)", overall_perf)
    _print_perf("LONG BOOK", long_perf)
    _print_perf("SHORT BOOK", short_perf)

    print("\n  BY YEAR")
    for y in years:
        p = by_year[y]
        print(f"    {y}: n={p.get('n_trades')} win={p.get('win_rate')} "
              f"total_return={p.get('total_return')} sharpe={p.get('sharpe')}")

    print("\n  EXIT REASON BREAKDOWN")
    for reason, v in exit_breakdown.items():
        print(f"    {reason:<16} n={v['n']:<5} share={v['share']:.1%}  avg_return={v['avg_return']}")

    print("\n  SIGNAL QUALITY")
    print(f"    IC (short book): mean={ic_short['ic_mean']}  tstat={ic_short['ic_tstat']}  "
          f"n_months={ic_short['n_months']}")
    print(f"    IC (long  book): mean={ic_long['ic_mean']}  tstat={ic_long['ic_tstat']}  "
          f"n_months={ic_long['n_months']}")

    if RUN_QUINTILE_DIAGNOSTIC:
        print("\n    Quintile monotonicity (SHORT pool, Q1->Q5, expect Q5 best):")
        for q, v in quintile_short.items():
            print(f"      Q{q}: n={v['n']:<5} avg_return={v['avg_return']}")
        print("    Quintile monotonicity (LONG pool, Q1->Q5, expect Q1 best):")
        for q, v in quintile_long.items():
            print(f"      Q{q}: n={v['n']:<5} avg_return={v['avg_return']}")

    print(f"\n  {'='*66}\n")

    # ── Files ────────────────────────────────────────────────────────────────
    trade_log_path = os.path.join(RESULTS_DIR, f"trade_log_{RUN_LABEL}.csv")
    pd.DataFrame(all_trades).to_csv(trade_log_path, index=False)
    print(f"  Trade log -> {trade_log_path}")

    returns_path = os.path.join(RESULTS_DIR, f"returns_{RUN_LABEL}.csv")
    overall_returns.rename('return').to_csv(returns_path, header=True)
    print(f"  Monthly returns -> {returns_path}")

    summary = {
        'capital': capital_info, 'initial_capital': initial_capital,
        'overall': overall_perf, 'long_book': long_perf, 'short_book': short_perf,
        'by_year': {str(y): v for y, v in by_year.items()},
        'exit_breakdown': exit_breakdown,
        'ic_short': ic_short, 'ic_long': ic_long,
        'quintile_short': quintile_short, 'quintile_long': quintile_long,
    }
    summary_path = os.path.join(RESULTS_DIR, f"performance_summary_{RUN_LABEL}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Performance summary -> {summary_path}")


if __name__ == "__main__":
    main()
