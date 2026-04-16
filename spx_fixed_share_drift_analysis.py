import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# --- SETTINGS ---
FILE_PATH = 'Portfolio Value-Summary-2026-04-16.csv'
CACHE_FILE = 'historical_prices.pkl'
LOOKBACK_DATE = '2025-10-16'
FORCE_REFRESH = False  # Set to True if returns still look like +600%

# --- PYCHARM DISPLAY SETTINGS ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def get_pure_stock_sleeve(path):
    print("--- Cleaning Portfolio Data: Strictly Isolating Individual Stocks ---")
    df = pd.read_csv(path, skiprows=4)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df[df['Symbol'] != '-------------------------']
    df = df[~df['Security'].str.contains('Total', case=False, na=False)]

    # Clean Shares and Market Value
    df['Shares'] = pd.to_numeric(df['Shares'].astype(str).str.replace(',', ''), errors='coerce')

    def safe_float(v):
        if isinstance(v, str):
            v = v.replace(',', '').replace('$', '').strip()
            if not v or v == '-' or '---' in v: return 0.0
            try:
                return float(v)
            except ValueError:
                return 0.0
        return float(v) if v is not None else 0.0

    df['Market Value'] = df['Market Value'].apply(safe_float)

    # RE-IMPLEMENTED COMPREHENSIVE FILTER
    def is_pure_stock(row):
        sym = str(row['Symbol']).strip()
        sec = str(row['Security']).upper()

        # A. Exclude GS and Nulls
        if sym == 'GS' or sym.lower() == 'nan' or sym == '': return False

        # B. Exclude Cash/Money Markets
        cash_k = ['CASH', 'MONEY MARKET', 'BDA', 'DEPOSIT', 'REPURCHASE', 'LIQUIDITY', 'FEDERATED']
        if any(k in sec for k in cash_k) or sym == 'USD': return False

        # C. Exclude Mutual Funds (5 letters ending in X)
        if len(sym) == 5 and sym.endswith('X'): return False

        # D. Exclude ETFs/Bonds/Funds via Keywords
        fund_k = [
            ' ETF', ' FUND', ' FD ', ' INDEX', ' TRUST', ' ISHARES', ' VANGUARD',
            ' INVESCO', ' SCHWAB', ' SPDR', ' DIMENSIONAL', ' SELECT ',
            ' TREASURY', ' MUNICIPAL', ' TAX FREE', ' INCOME', ' BOND', ' BD ', ' TR ',
            ' STRATEGIC ADVISERS', ' PIMCO', ' METROPOLITAN WEST', ' T ROWE PRICE'
        ]
        if any(k in sec for k in fund_k): return False

        # E. Validation: Individual stocks usually 1-5 letters
        return 1 <= len(sym) <= 5 and sym.isalpha()

    df = df[df.apply(is_pure_stock, axis=1)].copy()
    return df[['Security', 'Symbol', 'Shares', 'Market Value']].dropna()


def get_historical_drift_data(tickers, start_date):
    end_date = datetime.now().strftime('%Y-%m-%d')
    if os.path.exists(CACHE_FILE) and not FORCE_REFRESH:
        cached_data = pd.read_pickle(CACHE_FILE)
        if all(t in cached_data.columns for t in tickers):
            return cached_data[tickers]

    print(f"--- Downloading CLEAN split-adjusted data for {len(tickers)} stocks... ---")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    prices = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
    prices.to_pickle(CACHE_FILE)
    return prices


# --- EXECUTION ---
portfolio = get_pure_stock_sleeve(FILE_PATH)
tickers = portfolio['Symbol'].unique().tolist()

# 1. Fetch Prices
prices = get_historical_drift_data(tickers, LOOKBACK_DATE)
price_then, price_now = prices.iloc[0], prices.iloc[-1]

# 2. Build Comparison Table
analysis = portfolio.copy()
analysis['Price_Then'] = analysis['Symbol'].map(price_then)
analysis['Price_Now'] = analysis['Symbol'].map(price_now)

# 3. Calculate "Rewound" Value based on CURRENT shares
# This simulates what the weights WOULD have been if you held these shares back then
analysis['Value_Then'] = analysis['Shares'] * analysis['Price_Then']
analysis['Value_Now'] = analysis['Shares'] * analysis['Price_Now']

# 4. Normalize Weights to the Stock Sleeve only
analysis['Weight_Then'] = analysis['Value_Then'] / analysis['Value_Then'].sum()
analysis['Weight_Now'] = analysis['Value_Now'] / analysis['Value_Now'].sum()
analysis['Drift_Delta'] = analysis['Weight_Now'] - analysis['Weight_Then']
analysis['Return'] = (analysis['Price_Now'] / analysis['Price_Then']) - 1

# --- REPORT ---
print("\n" + "=" * 145)
print(f"   PURE STOCK DRIFT ANALYSIS: {LOOKBACK_DATE} vs. TODAY (EX-CASH/FUNDS/GS)")
print("=" * 145)
print(
    f"{'Security Name':<45} | {'Symbol':<8} | {'Weight THEN':>12} | {'Weight NOW':>12} | {'Drift Delta':>12} | {'Return'}")
print("-" * 145)

top_drift = analysis.sort_values('Weight_Now', ascending=False).head(35)
for _, row in top_drift.iterrows():
    print(
        f"{row['Security'][:45]:<45} | {row['Symbol']:<8} | {row['Weight_Then']:>11.2%} | {row['Weight_Now']:>11.2%} | {row['Drift_Delta']:>+11.2%} | {row['Return']:>+10.1%}")

print("=" * 145)