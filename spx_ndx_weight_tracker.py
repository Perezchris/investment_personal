import pandas as pd
import numpy as np
from scipy.optimize import minimize

# --- SETTINGS & FILE PATH ---
FILE_PATH = 'Portfolio Value-Summary-2026-04-16.csv'


def get_pure_stock_sleeve(path):
    print("--- Cleaning Portfolio Data: Isolating Individual Stocks (Ex-Cash/Funds/GS) ---")
    df = pd.read_csv(path, skiprows=4)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # 1. REMOVE FORMATTING ROWS AND TOTALS
    df = df[df['Symbol'] != '-------------------------']
    df = df[~df['Security'].str.contains('Total', case=False, na=False)]

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

    # 2. DEFINE FILTERS
    def is_pure_stock(row):
        sym = str(row['Symbol']).strip()
        sec = str(row['Security']).upper()

        # A. Exclude Goldman Sachs
        if sym == 'GS': return False

        # B. Exclude Cash & Money Markets
        cash_keywords = ['CASH', 'MONEY MARKET', 'BDA', 'DEPOSIT', 'REPURCHASE', 'LIQUIDITY']
        if any(k in sec for k in cash_keywords) or sym == 'USD':
            return False

        # C. Exclude Mutual Funds (5 letters ending in X)
        if len(sym) == 5 and sym.endswith('X'):
            return False

        # D. Exclude ETFs/Bonds/Funds via Keywords
        fund_keywords = [
            ' ETF', ' FUND', ' FD ', ' INDEX', ' TRUST', ' ISHARES', ' VANGUARD',
            ' INVESCO', ' SCHWAB', ' SPDR', ' DIMENSIONAL', ' SELECT ',
            ' TREASURY', ' MUNICIPAL', ' TAX FREE', ' INCOME', ' BOND', ' BD ', ' TR '
        ]
        if any(k in sec for k in fund_keywords):
            return False

        # E. Final validation: Individual stocks are typically 1-5 letters
        return 1 <= len(sym) <= 5 and sym.isalpha()

    # Apply filters and remove zero/negative values
    df = df[df.apply(is_pure_stock, axis=1)].copy()
    df = df[df['Market Value'] > 0]

    # 3. CALCULATE WEIGHTS (Percentage of filtered individual stock total)
    total_val = df['Market Value'].sum()
    df['Port_Weight'] = df['Market Value'] / total_val

    print(f"Isolated {len(df)} individual stocks.")
    print(f"Total Value of Stock Sleeve (Ex-GS/Cash): ${total_val:,.2f}")
    return df[['Security', 'Symbol', 'Port_Weight']]


# --- 3. TOP 50 INDEX WEIGHTS (ESTIMATED APRIL 2026) ---
spx_holdings = {
    'NVDA': 0.0755, 'AAPL': 0.0606, 'MSFT': 0.0487, 'AMZN': 0.0419, 'GOOGL': 0.0329,
    'GOOG': 0.0304, 'AVGO': 0.0295, 'META': 0.0267, 'TSLA': 0.0228, 'BRKB': 0.0160,
    'WMT': 0.0156, 'JPM': 0.0130, 'LLY': 0.0126, 'XOM': 0.0099, 'V': 0.0095,
    'JNJ': 0.0088, 'MU': 0.0081, 'ORCL': 0.0080, 'MA': 0.0073, 'NFLX': 0.0071,
    'AMD': 0.0070, 'COST': 0.0068, 'BAC': 0.0060, 'CVX': 0.0058, 'ABBV': 0.0058,
    'CAT': 0.0056, 'INTC': 0.0054, 'PLTR': 0.0053, 'HD': 0.0053, 'PG': 0.0052,
    'CSCO': 0.0052, 'LRCX': 0.0051, 'KO': 0.0051, 'GE': 0.0049, 'AMAT': 0.0048,
    'MS': 0.0047, 'UNH': 0.0045, 'MRK': 0.0045, 'GEV': 0.0041, 'RTX': 0.0041,
    'WFC': 0.0039, 'PM': 0.0038, 'IBM': 0.0037, 'LIN': 0.0036, 'KLAC': 0.0036,
    'AXP': 0.0035, 'C': 0.0035, 'MCD': 0.0034, 'TMUS': 0.0034, 'PEP': 0.0034
}

ndx_holdings = {
    'NVDA': 0.0881, 'AAPL': 0.0729, 'MSFT': 0.0547, 'AMZN': 0.0493, 'META': 0.0359,
    'GOOGL': 0.0358, 'AVGO': 0.0345, 'TSLA': 0.0337, 'GOOG': 0.0333, 'WMT': 0.0322,
    'MU': 0.0254, 'NFLX': 0.0230, 'COST': 0.0230, 'AMD': 0.0213, 'LRCX': 0.0177,
    'CSCO': 0.0173, 'INTC': 0.0172, 'AMAT': 0.0166, 'PLTR': 0.0160, 'LIN': 0.0125,
    'KLAC': 0.0123, 'PEP': 0.0113, 'TMUS': 0.0112, 'TXN': 0.0104, 'AMGN': 0.0100,
    'GILD': 0.0091, 'ADI': 0.0090, 'ISRG': 0.0086, 'HON': 0.0079, 'SHOP': 0.0075,
    'BKNG': 0.0074, 'QCOM': 0.0074, 'PANW': 0.0070, 'ASML': 0.0070, 'APP': 0.0068,
    'WDC': 0.0063, 'MRVL': 0.0061, 'STX': 0.0059, 'VRTX': 0.0059, 'SBUX': 0.0059,
    'CEG': 0.0056, 'INTU': 0.0054, 'CMCSA': 0.0054, 'CRWD': 0.0054, 'ADBE': 0.0052,
    'MAR': 0.0050, 'MELI': 0.0049, 'SNPS': 0.0042, 'CDNS': 0.0042, 'ORLY': 0.0042
}

# --- 4. OPTIMIZATION LOGIC ---
pure_sleeve = get_pure_stock_sleeve(FILE_PATH)
portfolio_weights = pure_sleeve.set_index('Symbol')['Port_Weight'].to_dict()
symbol_to_name = pure_sleeve.set_index('Symbol')['Security'].to_dict()

all_tickers = list(set(list(spx_holdings.keys()) + list(ndx_holdings.keys()) + list(portfolio_weights.keys())))


def objective_function(w):
    w_spx, w_ndx = w[0], 1 - w[0]
    total_sq_diff = 0
    for t in all_tickers:
        p_w = portfolio_weights.get(t, 0)
        s_w = spx_holdings.get(t, 0)
        n_w = ndx_holdings.get(t, 0)
        target_w = (w_spx * s_w) + (w_ndx * n_w)
        total_sq_diff += (p_w - target_w) ** 2
    return total_sq_diff


res = minimize(objective_function, [0.5], bounds=[(0, 1)])
w_spx_opt, w_ndx_opt = res.x[0], 1 - res.x[0]

# --- 5. REPORT GENERATION ---
print("\n" + "=" * 125)
print("   OPTIMIZED INDIVIDUAL STOCK WEIGHT COMPARISON (TOP 50)")
print("=" * 125)
print(f"Optimal VOO (S&P 500) Blend:    {w_spx_opt:.2%}")
print(f"Optimal QQQ (Nasdaq 100) Blend:  {w_ndx_opt:.2%}")
print("-" * 125)
print(f"{'Security Name':<45} | {'Symbol':<8} | {'Sleeve %':>8} | {'Index %':>8} | {'Diff %':>8} | {'Action'}")
print("-" * 125)

sorted_tickers = sorted(all_tickers, key=lambda t: portfolio_weights.get(t, 0), reverse=True)
for t in sorted_tickers[:40]:
    p_w = portfolio_weights.get(t, 0) * 100
    b_w = ((w_spx_opt * spx_holdings.get(t, 0)) + (w_ndx_opt * ndx_holdings.get(t, 0))) * 100
    diff = p_w - b_w

    name = symbol_to_name.get(t, "--- INDEX CONSTITUENT ---")
    action = "OVERWEIGHT" if diff > 0.5 else "UNDERWEIGHT" if diff < -0.5 else "MATCH"

    if p_w > 0 or b_w > 0:
        print(f"{name[:45]:<45} | {t:<8} | {p_w:>8.2f}% | {b_w:>8.2f}% | {diff:>+8.2f}% | {action}")

print("=" * 125)