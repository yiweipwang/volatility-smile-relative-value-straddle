"""
config.py — Volatility Smile Relative-Value Straddle Strategy
See README.md for the full design writeup.

Data: OptionMetrics (options) + CRSP (stock prices) + link table (secid->permno).
"""

import os
import sys

# ── Signal parameters ──────────────────────────────────────────────────────────
RV_WINDOW          = 21     # trading days trailing, close-to-close realized vol
Z_LOOKBACK_MONTHS  = 12     # trailing months used to build the z-score baseline
MIN_HISTORY        = 8      # min valid months before a name is scored
STD_FLOOR          = 0.005  # floor on z-score denominator (annualized vol points)

# ── Smile eligibility gate ──────────────────────────────────────────────────────
Z_EXTREME_HIGH = 1.5    # short-eligible requires curvature_z / |skew_z| below this
Z_EXTREME_LOW  = -1.5   # long-eligible excluded if curvature_z below this AND skew_z unremarkable

# ── Portfolio construction ──────────────────────────────────────────────────────
N_QUINTILES   = 5
LONG_QUINTILE  = 1   # cheapest vol, within the long-eligible pool
SHORT_QUINTILE = 5   # most expensive vol, within the short-eligible pool

# ── Option selection ────────────────────────────────────────────────────────────
MIN_DTE  = 20   # sanity bounds only — expiry itself is the deterministic next 3rd Friday
MAX_DTE  = 45

DELTA_MIN = 0.25   # ATM call delta band (target ≈0.50) — fixes the straddle strike
DELTA_MAX = 0.75

WING_DELTA_MIN = 0.10   # 25-delta wing band (target ≈0.25), each side independently
WING_DELTA_MAX = 0.40

MAX_SPREAD_PCT_ENTRY = 0.03   # discard quotes with bid-ask spread > 3% of mid

# ── Exit rules (used by backtest.py / monitor.py, not signal_generation.py) ────
CONVERGENCE_Z  = 0.1    # take-profit band: exit when ivrp_z reverts inside ±this
DIVERGENCE_Z   = 2.0    # stop-loss threshold: entry smile condition worsens past this
FORCE_EXIT_DTE = 5      # forced close once DTE drops to this or below

# ── Capital (derived, not an input assumption — see README §5) ────────────────
CONTRACTS_PER_LEG = 1      # fixed size, no % of capital
SHORT_MARGIN_PCT  = 0.20   # Reg T net margin rate: 20% x underlying x 100

# ── Data paths ──────────────────────────────────────────────────────────────────
_HERE                = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR          = os.path.join(_HERE, "results")
SIGNAL_HISTORY_FILE  = os.path.join(_HERE, "signal_history.json")
PORTFOLIO_TARGET     = os.path.join(_HERE, "portfolio_target.json")

if sys.platform == "win32":
    OPTIONMETRICS_BASE = r"C:\Users\Icarus\Desktop\data\[Option Price]"
    CRSP_PATH          = r"C:\Users\Icarus\Desktop\data\[Stock Price]\stock_price__1996_2022.parquet"
    LINK_TABLE_PATH    = r"C:\Users\Icarus\Desktop\data\[Link Table]\us_stock_permno_with_secid.parquet"
else:
    OPTIONMETRICS_BASE = "/mnt/d/Icarus/[Option Price Data]"
    CRSP_PATH          = "/mnt/d/Icarus/[Stock Price]/stock_price__1996_2022.parquet"
    LINK_TABLE_PATH    = "/mnt/d/Icarus/[Link Table]/us_stock_permno_with_secid.parquet"

# Columns to read from each daily OM parquet
OM_COLUMNS = [
    'secid', 'date', 'symbol', 'exdate', 'cp_flag',
    'strike_price', 'best_bid', 'best_offer', 'delta', 'impl_volatility',
    'open_interest',
]

MIN_OPEN_INTEREST = 100   # discard options with OI < 100

# ── Universe ─────────────────────────────────────────────────────────────────────
# Same 460 tickers as straddle_momentum (present in both CRSP universe and
# SP500 intraday data directory) — copied here to keep this project self-contained.
UNIVERSE = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM",
    "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM",
    "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP",
    "AMT", "AMZN", "ANET", "AON", "AOS", "APA", "APD", "APH", "APO", "APTV",
    "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO", "BA",
    "BAC", "BAX", "BBY", "BDX", "BEN", "BG", "BIIB", "BKNG", "BKR", "BLDR",
    "BLK", "BMY", "BR", "BRO", "BSX", "BX", "BXP", "C", "CAG", "CAH",
    "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CF",
    "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMCSA", "CME",
    "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COIN", "COO", "COP", "COR",
    "COST", "CPB", "CPRT", "CPT", "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX",
    "CTAS", "CTSH", "CTVA", "CVS", "CVX", "D", "DAL", "DASH", "DD", "DE",
    "DECK", "DELL", "DG", "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR", "DOC",
    "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", "EA",
    "EBAY", "ECL", "ED", "EFX", "EIX", "EL", "EMR", "EOG", "EPAM", "EQIX",
    "EQR", "EQT", "ERIE", "ES", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC",
    "EXPD", "EXPE", "EXR", "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE",
    "FFIV", "FICO", "FIS", "FITB", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV",
    "GD", "GDDY", "GE", "GEN", "GILD", "GIS", "GL", "GLW", "GM", "GNRC",
    "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN",
    "HCA", "HD", "HIG", "HII", "HLT", "HON", "HPE", "HPQ", "HRL", "HSIC",
    "HST", "HSY", "HUBB", "HUM", "HWM", "IBM", "ICE", "IDXX", "IEX", "IFF",
    "INCY", "INTC", "INTU", "INVH", "IP", "IQV", "IR", "IRM", "ISRG", "IT",
    "ITW", "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JPM", "KDP",
    "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC", "KMB", "KMI", "KO", "KR",
    "L", "LDOS", "LEN", "LH", "LHX", "LII", "LIN", "LLY", "LMT", "LNT",
    "LOW", "LRCX", "LULU", "LUV", "LVS", "LYB", "LYV", "MA", "MAA", "MAR",
    "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM",
    "MKC", "MLM", "MMM", "MNST", "MO", "MOS", "MPC", "MPWR", "MRK", "MRNA",
    "MS", "MSCI", "MSFT", "MSI", "MTB", "MTD", "MU", "NCLH", "NDAQ", "NDSN",
    "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP",
    "NTRS", "NUE", "NVDA", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC",
    "ON", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PAYX", "PCAR", "PCG", "PEG",
    "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR",
    "PM", "PNC", "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA",
    "PSX", "PTC", "PWR", "PYPL", "QCOM", "RCL", "REG", "REGN", "RF", "RJF",
    "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "SBAC", "SBUX",
    "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNPS", "SO", "SPG", "SPGI",
    "SRE", "STE", "STLD", "STT", "STX", "STZ", "SWK", "SWKS", "SYF", "SYK",
    "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TGT",
    "TJX", "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA",
    "TSN", "TT", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS",
    "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VICI", "VLO", "VMC",
    "VRSK", "VRSN", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT", "WDAY",
    "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB", "WSM", "WST",
    "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "YUM", "ZBH", "ZBRA", "ZTS",
]
