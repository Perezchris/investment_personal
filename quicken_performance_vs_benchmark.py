import io
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Configuration & Benchmark Mapping
# ---------------------------------------------------------
ACCOUNT_BENCHMARKS = {
    'Fidelity TRUST': 'AOR',  # Swapped from VBIAX to AOR (60/40 ETF) for reliable yfinance data
    'GS JTWROS AGES': 'ACWI',  # MSCI All-Country World Index Proxy
    'GS JTWROS DES': 'VT',  # Global Diversified Equity Proxy
    'GS JTWROS FIF': 'MUB',  # Municipal Bond Index Proxy
    'GS Tax Adv LH': 'SPY'  # S&P 500 Core Proxy
}

CSV_FILENAME = "data/Chris's Finances - Investing - Performance - By Account - 2026-09-24.csv"


# ---------------------------------------------------------
# 2. Enhanced CSV Parser & Loader (Quicken Mac / BOM Safe)
# ---------------------------------------------------------
def load_account_data(filepath):
    """
    Reads Quicken performance CSV: handles embedded BOM (\ufeff),
    detects the header via 'Avg. Annual Return' / 'ROI (%)',
    separates accounts from indented holdings, and converts % strings to floats.
    """
    raw_bytes = None
    try:
        with open(filepath, 'rb') as f:
            raw_bytes = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'. Ensure the path is correct.")
        return None

    # Decode bytes (UTF-8 with BOM handles \ufeff at the file start, latin-1 as fallback)
    text = None
    for enc in ['utf-8-sig', 'utf-8', 'latin-1']:
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        text = raw_bytes.decode('latin-1', errors='replace')

    # Remove any stray BOM markers throughout the file
    text = text.replace('\ufeff', '')
    lines = text.splitlines(keepends=True)

    # Detect header row by looking for the return column markers
    header_idx = None
    for idx, line in enumerate(lines):
        line_clean = line.lower()
        if "avg. annual return" in line_clean or ("roi (%)" in line_clean and "ytd" in line_clean):
            header_idx = idx
            break

    if header_idx is None:
        print("Error: Could not find table header row.")
        return None

    # Slice the CSV starting at the detected header line
    table_csv_str = "".join(lines[header_idx:])

    # Read into DataFrame
    df = pd.read_csv(io.StringIO(table_csv_str), sep=',', on_bad_lines='skip')

    # Map the columns by position
    col_names = [
        'Account_Name', 'Avg_Annual_YTD', 'Avg_Annual_1Y', 'Avg_Annual_3Y', 'Avg_Annual_5Y',
        'ROI_YTD', 'ROI_1Y', 'ROI_3Y', 'ROI_5Y', 'ROI_Total'
    ]
    rename_dict = {orig: new for orig, new in zip(df.columns[:len(col_names)], col_names)}
    df = df.rename(columns=rename_dict)

    # Filter top-level accounts vs indented asset positions
    def is_account_row(val):
        if pd.isna(val):
            return False
        val_str = str(val).replace('\ufeff', '')
        if len(val_str.strip()) < 2:
            return False
        # Holdings are indented with spaces or tabs
        if val_str.startswith(' ') or val_str.startswith('\t') or (len(val_str) > 1 and val_str[1].isspace()):
            return False
        return True

    df['Is_Account'] = df['Account_Name'].apply(is_account_row)
    accounts_df = df[df['Is_Account']].copy()
    accounts_df['Account_Name'] = accounts_df['Account_Name'].astype(str).str.replace('\ufeff', '').str.strip()

    # Convert percentage strings to numeric floats
    metric_cols = [c for c in col_names if c != 'Account_Name' and c in accounts_df.columns]
    for col in metric_cols:
        accounts_df[col] = (
            accounts_df[col]
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.replace('"', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        accounts_df[col] = pd.to_numeric(accounts_df[col], errors='coerce')

    # Map accounts to our defined benchmark strategies
    accounts_df['Mapped Account'] = accounts_df['Account_Name'].apply(
        lambda x: next((key for key in ACCOUNT_BENCHMARKS.keys() if key.lower() in str(x).lower()), None)
    )

    tracked_accounts = accounts_df.dropna(subset=['Mapped Account']).copy()
    return tracked_accounts


# ---------------------------------------------------------
# 3. Fetch Multi-Period Benchmark Data
# ---------------------------------------------------------
def get_benchmark_returns(tickers):
    """Fetches historical cumulative returns for YTD, 1Y, 3Y, and 5Y periods."""
    print(f"\nFetching multi-period benchmark data for: {tickers}...")
    # yfinance period mapping
    periods = {'ytd': 'YTD', '1y': '1Y', '3y': '3Y', '5y': '5Y'}
    results = {period_name: {} for period_name in periods.values()}

    for yf_period, col_name in periods.items():
        # Download data silently
        data = yf.download(tickers, period=yf_period, progress=False)

        # Handle formatting depending on if 1 or multiple tickers are passed
        if 'Close' in data:
            data = data['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])

        for ticker in tickers:
            if ticker in data.columns:
                series = data[ticker].dropna()
                if len(series) > 0:
                    # Calculate cumulative percentage return
                    ret = (series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100
                    results[col_name][ticker] = ret
                else:
                    results[col_name][ticker] = np.nan
            else:
                results[col_name][ticker] = np.nan

    return results


# ---------------------------------------------------------
# 4. Analysis and Comparison
# ---------------------------------------------------------
def main():
    account_df = load_account_data(CSV_FILENAME)
    if account_df is None or account_df.empty:
        print("No matching top-level account data found.")
        return

    # Retrieve ETF proxy benchmarks for matched accounts
    needed_benchmarks = list(set([ACCOUNT_BENCHMARKS[acc] for acc in account_df['Mapped Account'].unique()]))
    bench_returns = get_benchmark_returns(needed_benchmarks)

    account_df['Benchmark Proxy'] = account_df['Mapped Account'].map(ACCOUNT_BENCHMARKS)

    # Map returns and calculate deltas for all periods
    time_horizons = ['YTD', '1Y', '3Y', '5Y']
    for period in time_horizons:
        account_df[f'Bench_{period}'] = account_df['Benchmark Proxy'].map(bench_returns[period])
        account_df[f'Delta_{period}'] = account_df[f'ROI_{period}'] - account_df[f'Bench_{period}']

    # Print Clean Formatted Tables
    pd.options.display.float_format = '{:,.2f}%'.format

    for period in time_horizons:
        print(f"\n--- {period} Performance vs. Benchmark ---")
        cols = ['Account_Name', 'Benchmark Proxy', f'ROI_{period}', f'Bench_{period}', f'Delta_{period}']

        # Only show accounts that have actual data for this period (e.g. skipping 5Y for newer accounts)
        view_df = account_df.dropna(subset=[f'ROI_{period}'])
        print(view_df[cols].to_string(index=False))

    # ---------------------------------------------------------
    # 5. Visualization (1-Year & 3-Year Comparison)
    # ---------------------------------------------------------
    # Filter out any accounts missing 3-year data for a clean chart
    chart_df = account_df.dropna(subset=['ROI_1Y', 'ROI_3Y']).copy()

    if chart_df.empty:
        print("\nNot enough data to plot multi-year chart.")
        return

    x = np.arange(len(chart_df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    # Grouped bars for 1Y
    ax.bar(x - width * 1.5, chart_df['ROI_1Y'], width, label='Account 1Y', color='#1f77b4')
    ax.bar(x - width * 0.5, chart_df['Bench_1Y'], width, label='Benchmark 1Y', color='#aec7e8')

    # Grouped bars for 3Y
    ax.bar(x + width * 0.5, chart_df['ROI_3Y'], width, label='Account 3Y', color='#ff7f0e')
    ax.bar(x + width * 1.5, chart_df['Bench_3Y'], width, label='Benchmark 3Y', color='#ffbb78')

    ax.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax.set_title('1-Year and 3-Year Account Performance vs. Targets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(chart_df['Account_Name'], rotation=15, ha='right', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()