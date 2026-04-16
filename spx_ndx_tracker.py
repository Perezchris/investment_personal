import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
import os

# --- SETTINGS & FILE PATH ---
FILE_PATH = 'Portfolio Value-Summary-2026-04-16.csv'
CACHE_FILE = 'historical_prices.pkl'
BENCHMARKS = {'VOO': 'S&P 500', 'QQQ': 'Nasdaq 100'}
LOOKBACK_YEARS = 2

# --- PYCHARM DISPLAY SETTINGS ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def clean_portfolio(path):
    print("--- Cleaning Portfolio Data ---")
    df = pd.read_csv(path, skiprows=4)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # 1. REMOVE NULLS & DASHES
    df = df.dropna(subset=['Symbol'])
    df = df[df['Symbol'] != '-------------------------']
    df = df[~df['Security'].str.contains('Total', case=False, na=False)]
    df['Symbol'] = df['Symbol'].astype(str).str.strip()

    def safe_float(v):
        if isinstance(v, str):
            v = v.replace(',', '').replace('$', '').replace('%', '').strip()
            if not v or v == '-' or '---' in v: return 0.0
            try:
                return float(v)
            except ValueError:
                return 0.0
        return float(v) if v is not None else 0.0

    df['Market Value'] = df['Market Value'].apply(safe_float)

    def is_indiv_stock(row):
        sym = row['Symbol']
        sec = str(row['Security']).upper()
        if sym.lower() == 'nan' or sym == '' or sym == 'GS': return False
        if any(k in sec for k in [' ETF', ' FUND', ' FD ', ' INDEX', ' TRUST', ' ISHARES', ' VANGUARD']):
            return False
        return 1 <= len(sym) <= 5 and sym.isalpha()

    df = df[df.apply(is_indiv_stock, axis=1)].copy()
    df = df[df['Market Value'] > 0]

    total_val = df['Market Value'].sum()
    df['Weight'] = df['Market Value'] / total_val
    return df


def get_historical_data(tickers, start, end):
    # Always check cache first
    if os.path.exists(CACHE_FILE):
        cached_data = pd.read_pickle(CACHE_FILE)
        # Verify if all tickers exist in the cache
        if all(t in cached_data.columns for t in tickers):
            return cached_data[tickers]

    # If cache is missing data, download with AUTO_ADJUST
    # This ensures "Close" is ALWAYS the split-adjusted price
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)

    # Handle both MultiIndex and SingleIndex results
    prices = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data

    # Save back to cache
    prices.to_pickle(CACHE_FILE)
    return prices

# --- 2. EXECUTION ---
portfolio_df = clean_portfolio(FILE_PATH)
tickers = [str(t) for t in portfolio_df['Symbol'].unique() if isinstance(t, str) and t.lower() != 'nan']
all_needed = tickers + list(BENCHMARKS.keys())

start_date = (datetime.today() - timedelta(days=365 * LOOKBACK_YEARS)).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

# Get prices (via Cache or API)
prices = get_historical_data(all_needed, start_date, end_date)
returns = prices.pct_change().dropna(how='all')

# --- 3. CREATE SYNTHETIC PORTFOLIO RETURN ---
valid_tickers = [t for t in tickers if t in returns.columns]
weights = portfolio_df.set_index('Symbol')['Weight']
port_weights = weights[valid_tickers]
port_weights = port_weights / port_weights.sum()

portfolio_returns = (returns[valid_tickers] * port_weights).sum(axis=1)
benchmark_returns = returns[list(BENCHMARKS.keys())]


# --- 4. OPTIMIZATION ---
def tracking_error(w, bench_rets, port_rets):
    combined_bench = (bench_rets * w).sum(axis=1)
    common_idx = port_rets.index.intersection(combined_bench.index)
    return np.std(port_rets.loc[common_idx] - combined_bench.loc[common_idx]) * np.sqrt(252)


cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1), (0, 1)]
init_guess = [0.5, 0.5]

res = minimize(tracking_error, init_guess, args=(benchmark_returns, portfolio_returns),
               method='SLSQP', bounds=bounds, constraints=cons)

# --- 5. RESULTS ---
voo_w, qqq_w = res.x
print("\n" + "=" * 50)
print("   OPTIMIZED INDEX PROXY (EX-GS SLEEVE)")
print("=" * 50)
print(f"Optimal VOO (S&P 500) Weight:   {voo_w:.2%}")
print(f"Optimal QQQ (Nasdaq 100) Weight: {qqq_w:.2%}")
print("-" * 50)
print(f"Annualized Tracking Error:      {res.fun:.2%}")
print(f"Cache File: {os.path.abspath(CACHE_FILE)}")
print("=" * 50)

# --- 6. VISUALIZATION ---
import matplotlib.pyplot as plt

plt.style.use('dark_background')
plt.figure(figsize=(12, 7))

((1 + portfolio_returns).cumprod() * 100).plot(label='Actual SMA Sleeve (Ex-GS)', color='cyan')
((1 + (benchmark_returns * res.x).sum(axis=1)).cumprod() * 100).plot(label='Optimized Proxy Blend', linestyle='--',
                                                                     color='orange')

plt.title(f"Best Fit Proxy: {voo_w:.0%} VOO / {qqq_w:.0%} QQQ")
plt.ylabel("Growth of $100")
plt.legend()
plt.grid(True, alpha=0.1)
plt.savefig("proxy_comparison.png", dpi=300)
plt.show()