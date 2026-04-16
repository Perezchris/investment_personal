import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# --- PYCHARM DISPLAY FIXES ---
# Set width high enough to prevent wrapping of the wide After-Tax table
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.colheader_justify', 'center')

# 1. PORTFOLIOS (Bogleheads Wiki Favorites)
lazy_portfolios = {
    "Three-Fund": {"VTI": 0.42, "VXUS": 0.18, "BND": 0.40},
    "Two-Fund": {"VT": 0.60, "BNDW": 0.40},
    "Rick Ferri Core Four": {"VTI": 0.48, "VXUS": 0.24, "BND": 0.20, "VNQ": 0.08},
    "Coffeehouse": {"VTI": 0.10, "VTV": 0.10, "VBR": 0.10, "VXUS": 0.10, "VXF": 0.10, "VNQ": 0.10, "BND": 0.40},
    "Margaritaville": {"VTI": 0.333, "VXUS": 0.333, "VTIP": 0.334},
    "No-Brainer": {"VTI": 0.25, "VEU": 0.25, "VB": 0.25, "BND": 0.25},
    "Second Grader": {"VTI": 0.60, "VXUS": 0.30, "BND": 0.10},
    "Swensen (Yale)": {"VTI": 0.30, "VEA": 0.15, "VWO": 0.05, "VNQ": 0.20, "TIP": 0.15, "BND": 0.15},
    "Ideal Index": {"VTI": 0.0625, "VTV": 0.0937, "VB": 0.0625, "VBR": 0.0938, "VEA": 0.15, "VWO": 0.05, "VNQ": 0.08,
                    "BND": 0.4075},
    "Permanent Portfolio": {"VTI": 0.25, "TLT": 0.25, "GLD": 0.25, "BIL": 0.25}
}

# Mapping for Assets (used for tax/yield lookups)
asset_metadata = {
    "VTI": "equity", "VXUS": "equity", "VT": "equity", "VEU": "equity", "VB": "equity",
    "VTV": "equity", "VBR": "equity", "VXF": "equity", "VEA": "equity", "VWO": "equity",
    "BND": "fixed_income", "BNDW": "fixed_income", "TLT": "fixed_income", "BIL": "fixed_income",
    "VNQ": "reit", "VTIP": "tips", "TIP": "tips", "GLD": "commodity"
}

# 2026 Greenwich Tax Rates ($450k Income: 35% Fed + 3.8% NIIT + 6.99% CT)
tax_rates = {"equity": 0.257, "fixed_income": 0.419, "reit": 0.419, "tips": 0.419, "commodity": 0.28}
yields = {"equity": 0.016, "fixed_income": 0.038, "reit": 0.042, "tips": 0.025, "commodity": 0.0}
CACHE_FILE = "bogleheads_master_cache.pkl"


# 2. DATA FETCHING
def fetch_data(tickers, start, end):
    if os.path.exists(CACHE_FILE):
        if datetime.fromtimestamp(os.path.getmtime(CACHE_FILE)).date() == datetime.today().date():
            return pd.read_pickle(CACHE_FILE)
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
    prices.to_pickle(CACHE_FILE)
    return prices


# 3. STATS ENGINE
def get_metrics(rets, tax_drag=0):
    r = rets - (tax_drag / 252)

    def c_ret(s): return (1 + s).prod() - 1 if len(s) > 10 else np.nan

    years = len(r) / 252
    cagr = ((1 + r).prod()) ** (1 / years) - 1 if years > 0 else np.nan

    growth = (1 + r).cumprod()
    peaks = growth.cummax()
    dds = (growth - peaks) / peaks

    # 2nd DD (Mask Covid 2020-02-15 to 2020-05-15)
    non_cov = dds[(dds.index < '2020-02-15') | (dds.index > '2020-05-15')]

    return {
        "YTD": c_ret(r[r.index.year == 2026]),
        "1Y": c_ret(r.iloc[-252:]), "3Y": c_ret(r.iloc[-756:]),
        "5Y": c_ret(r.iloc[-1260:]), "10Y": c_ret(r.iloc[-2520:]),
        "CAGR": cagr, "Vol": r.std() * np.sqrt(252),
        "MaxDD": dds.min(), "MaxDD_Date": dds.idxmin().strftime('%Y-%m-%d'),
        "2ndDD": non_cov.min(), "2ndDD_Date": non_cov.idxmin().strftime('%Y-%m-%d'),
        "Growth": growth, "DD_Series": dds
    }


# 4. EXECUTION
all_t = list(set(t for p in lazy_portfolios.values() for t in p.keys()))
raw_prices = fetch_data(all_t, (datetime.today() - timedelta(days=365 * 10 + 40)).strftime('%Y-%m-%d'),
                        datetime.today().strftime('%Y-%m-%d'))

pre_tab, post_tab = {}, {}
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

for name, assets in lazy_portfolios.items():
    available = [t for t in assets.keys() if t in raw_prices.columns]
    if len(available) < len(assets): continue

    p_data = raw_prices[available].dropna()
    weights = pd.Series({t: w for t, w in assets.items()})
    p_rets = (p_data.pct_change().dropna() * weights).sum(axis=1)

    drag = sum(w * yields[asset_metadata[t]] * tax_rates[asset_metadata[t]] for t, w in assets.items())

    pre = get_metrics(p_rets, 0)
    post = get_metrics(p_rets, drag)

    # FORMATTING TABLES
    time_keys = ["YTD", "1Y", "3Y", "5Y", "10Y", "CAGR"]

    pre_tab[name] = {k: f"{pre[k]:.2%}" if pd.notnull(pre[k]) else "N/A" for k in time_keys}

    post_tab[name] = {k: f"{post[k]:.2%}" if pd.notnull(post[k]) else "N/A" for k in time_keys}
    post_tab[name].update({
        "Vol": f"{post['Vol']:.2%}", "Sharpe": f"{(post['CAGR'] - 0.03) / post['Vol']:.2f}",
        "MaxDD": f"{post['MaxDD']:.2%}", "MaxDD Date": post['MaxDD_Date'],
        "2ndDD": f"{post['2ndDD']:.2%}", "2ndDD Date": post['2ndDD_Date'],
        "Tax Drag": f"{drag:.2%}"
    })

    ax1.plot(post['Growth'] * 100000, label=name)
    ax2.plot(post['DD_Series'], label=name, alpha=0.5)

# 5. OUTPUT
print("\n" + "=" * 120 + "\nTABLE 1: PRE-TAX PERFORMANCE (GROSS RETURNS)\n" + "=" * 120)
print(pd.DataFrame(pre_tab).T)

print("\n" + "=" * 180 + "\nTABLE 2: AFTER-TAX PERFORMANCE & RISK METRICS (GREENWICH, CT - $450K INCOME)\n" + "=" * 180)
print(pd.DataFrame(post_tab).T)

ax1.set_title("After-Tax Growth ($100k Initial)"); ax1.legend(loc='upper left', fontsize='x-small')
ax2.set_title("Drawdown Profile (Peak-to-Trough)"); ax2.legend(loc='lower left', fontsize='x-small')
plt.tight_layout()

# Save the chart to a file in the same directory as the script
output_filename = "lazy_portfolio_analysis.png"
plt.savefig(output_filename, dpi=300)
print(f"\nCharts saved to: {os.path.abspath(output_filename)}")

# Show the popup window
plt.show()
