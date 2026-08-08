"""
backtest_v25.py — EXPERIMENTAL: v23 + v24 combined — no divergence exit AND no
entry smile filter, everything else v12.

v23 (divergence exit removed): Sharpe 0.80 -> 0.90, but max_dd worsened
(-20.8% -> -25.3%) — divergence was mostly a false-alarm early stop, but did
protect against genuine regime changes (e.g. 2020).
v24 (entry smile filter removed): Sharpe 0.80 -> 0.85, n_trades 161 -> 199,
max_dd IMPROVED a lot (-20.8% -> -8.3%) — but SHORT book got worse (0.62 ->
0.55) because previously-excluded "extreme smile" names are now enterable,
and a lot of them got cut via divergence shortly after (divergence share
23.6% -> 35.2%).

This variant tests whether combining them compounds the gains: if v24's newly
-enterable extreme-smile SHORT names were mostly being cut by divergence
specifically (not converging/forced_dte), then ALSO removing divergence here
should stop punishing exactly the trades v24 added, on top of v23's own
already-demonstrated improvement.

Monkey-patches sg.eligibility AND bt.check_exit in this process's own memory
only, same non-invasive pattern as v6/v7/v23/v24 combined — does not touch
signal_generation.py / config.py / backtest.py on disk. Reuses simulate_month,
capital tracking, and all reporting from backtest.py unmodified.

Run: python3 backtest_v25.py
"""

import signal_generation as sg
import backtest as bt
from config import CONVERGENCE_Z


def eligibility_no_smile_filter(z: dict) -> tuple:
    return True, True


def check_exit_no_divergence(pos: dict, om_day, crsp_idx: dict, cur_date):
    snap = sg.compute_smile_snapshot(om_day, pos['ticker'], pos['expiry_date'])
    if snap is None:
        return None
    rv = sg.realized_vol(crsp_idx, pos['permno'], cur_date)
    if rv is None:
        return None

    m, s = pos['baseline']['ivrp']
    ivrp_z = (snap['atm_iv'] - rv - m) / s

    if pos['leg'] == 'short':
        if ivrp_z < CONVERGENCE_Z:
            return 'convergence'
    else:
        if ivrp_z > -CONVERGENCE_Z:
            return 'convergence'
    return None


sg.eligibility = eligibility_no_smile_filter
bt.check_exit  = check_exit_no_divergence
bt.RUN_LABEL = "v25_no_filter_no_divergence"
# BACKTEST_START/END/RUN_QUINTILE_DIAGNOSTIC left as whatever backtest.py
# currently has (v12's 2015-2022 range).

if __name__ == "__main__":
    bt.main()
