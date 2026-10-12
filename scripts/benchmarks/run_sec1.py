# ==============================================================================
# SECTION 1.1: DEPENDENCY MANAGEMENT & GLOBAL REPRODUCIBILITY
# ==============================================================================
import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import requests

# Suppress annoying, non-critical package warnings for clean academic presentation
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def seed_everything(seed: int = 42):
    """
    Enforces strict deterministic behavior across random processes,
    shuffles, and statistical bootstraps to ensure cross-machine verification.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)
print("✅ Section 1.1 Success: Environment initialized and random states locked.")

# ==============================================================================
# SECTION 1.2: HIGH-FIDELITY DATA INGESTION PIPELINE
# ==============================================================================

# 🔴 CONFIGURATION: Read your FRED API key from an environment variable instead
# of hardcoding it in the notebook. Set it before running, e.g.:
#   export FRED_API_KEY="your_32_char_key"
# If unset, the engine automatically switches to Web-Scrape Fallback Mode.
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

def fetch_systematic_risk_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Ingests high-fidelity equity index parameters and macroeconomic risk features.

    Academic Defenses:
    1. Proxy Integrity: Uses 'OEF' (iShares S&P 100 ETF) to account for transactional
       frictions and tracking errors absent in theoretical index benchmarks.
    2. Feature Mapping: Captures implied risk via the VIX index, structural credit
       stress via the BofA High-Yield Spread, and market liquidity via the Volume-to-Float ratio.
    """
    print(f"📥 Initiating data ingestion window: {start_date} to {end_date}...")

    # --------------------------------------------------------------------------
    # Part A: Financial Market Data Extraction (Yahoo Finance)
    # --------------------------------------------------------------------------
    tickers = {'SP100_Price': 'OEF', 'VIX': '^VIX'}
    print("   👉 Pulling equity metrics from Yahoo Finance API...")

    market_data = yf.download(
        list(tickers.values()),
        start=start_date,
        end=end_date,
        interval='1d',
        auto_adjust=True,
        progress=False
    )

    # Securely flatten and map multi-index columns
    df_market = pd.DataFrame(index=market_data.index)
    df_market['SP100_Close']  = market_data['Close']['OEF']
    df_market['SP100_Open']   = market_data['Open']['OEF']
    df_market['SP100_High']   = market_data['High']['OEF']
    df_market['SP100_Low']    = market_data['Low']['OEF']
    df_market['SP100_Volume'] = market_data['Volume']['OEF']
    df_market['VIX_Close']    = market_data['Close']['^VIX']

    # --------------------------------------------------------------------------
    # Part B: Macroeconomic Risk Ingestion (FRED API with Multi-Tier Fallbacks)
    # --------------------------------------------------------------------------
    series_id = "BAMLH0A0HYM2"  # ICE BofA High Yield Option-Adjusted Spread
    df_fred = pd.DataFrame()

    # Mode 1: Authentication via Official JSON API
    # 🔧 FIX: the previous URL pointed at the bare FRED homepage with no API
    # path, so it never actually reached the observations endpoint. This is
    # the real endpoint for pulling a single series as JSON.
    if FRED_API_KEY:
        print("   👉 Authenticating with St. Louis FRED via JSON API endpoint...")
        fred_url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }
        try:
            response = requests.get(fred_url, params=params, timeout=15)
            if response.status_code == 200:
                json_data = response.json()
                obs = json_data['observations']
                df_fred = pd.DataFrame(obs)
                df_fred['date'] = pd.to_datetime(df_fred['date'])
                df_fred.set_index('date', inplace=True)
                df_fred['Credit_Spread'] = pd.to_numeric(df_fred['value'], errors='coerce')
                df_fred = df_fred[['Credit_Spread']]
                print("      ✅ Authenticated JSON query successful.")
            else:
                print(f"      ⚠️ API Error (Status {response.status_code}). Switching to Fallback Mode...")
        except Exception as e:
            print(f"      ⚠️ Request failed ({str(e)}). Switching to Fallback Mode...")
    else:
        print("   👉 No FRED_API_KEY set. Skipping Mode 1 and trying the unauthenticated fallback...")

    # Mode 2: Unauthenticated Web-Scrape Fallback (Direct CSV stream parsing)
    # 🔧 FIX: the previous URL concatenated the series id straight onto the bare
    # domain (an invalid hostname). This is FRED's actual public CSV graph endpoint.
    if df_fred.empty:
        print("   👉 Mode 2: Direct web-stream parsing activated (No API key required)...")
        csv_fallback_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            df_csv = pd.read_csv(csv_fallback_url, parse_dates=['DATE'], index_col='DATE')
            df_csv[series_id] = pd.to_numeric(df_csv[series_id], errors='coerce')
            df_fred = df_csv.rename(columns={series_id: 'Credit_Spread'})
            df_fred = df_fred.loc[start_date:end_date]
            print("      ✅ Web-stream parsing successful.")
        except Exception as e:
            print(f"      ❌ Mode 2 Failure: Server blocked raw query ({str(e)}). Activating Mode 3...")

    # Mode 3: Local Parametric Simulation (Ensures code compile during live defense)
    if df_fred.empty:
        print("   ⚠️ Mode 3: Generating a synthetic econometric proxy matrix...")
        df_fred = pd.DataFrame(index=df_market.index)
        # Emulates high-yield spread profiles (Mean=4.0%, SD=0.5%) matching standard distributions
        df_fred['Credit_Spread'] = np.random.normal(4.0, 0.5, len(df_market))
        print("      ✅ Local econometric matrix synthesized.")
        print("      ⚠️ NOTE: This is synthetic noise, not real credit-spread data — any causal")
        print("         pathway involving Credit_Spread_Diff downstream is not economically meaningful")
        print("         until Mode 1 or Mode 2 succeeds.")

    # Join structural dataframes using a left outer execution map on trading days
    raw_df = df_market.join(df_fred, how='left')
    return raw_df

# Execute ingestion across a robust ~25-year historical window (OEF ETF inception is Oct 2000)
raw_data = fetch_systematic_risk_data(start_date="2001-01-01", end_date="2026-01-01")
print(f"📊 Section 1.2 Success: Raw Matrix compiled. Dimensions: {raw_data.shape}")

# ==============================================================================
# SECTION 1.3: DATA ALIGNMENT, BOUNDARY HANDLING & VOLATILITY COMPUTATION
# ==============================================================================
def process_and_align_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans structural alignment artifacts and engineers advanced variance parameters.

    Academic Defenses against Look-Ahead and Frequency Contamination:
    1. Multi-Frequency Isolation: Drops calendar days without transactional stock
       history and forward-fills macro variables to match data release lags safely.
    2. Variance Efficiency: Replaces simple close-to-close measures with Garman-Klass
       volatility estimators. Incorporating opening gaps and intraday extremes provides
       an estimation efficiency gains up to 8x higher.
    """
    print("\n⚙️ Running alignment corrections and variance calculations...")
    cleaned_df = df.copy()

    # Eliminate non-trading calendar frames (weekends and national market closures)
    cleaned_df = cleaned_df[cleaned_df['SP100_Close'].notna() & (cleaned_df['SP100_Volume'] > 0)]

    # Handle data release gaps by forward-filling macro attributes to ensure temporal consistency
    cleaned_df['Credit_Spread'] = cleaned_df['Credit_Spread'].ffill().bfill()
    cleaned_df['VIX_Close'] = cleaned_df['VIX_Close'].ffill().bfill()

    # Engineer Liquidity Proxy: Volume-to-Float Index (Standardized tracking proxy)
    cleaned_df['Liquidity_Proxy'] = cleaned_df['SP100_Volume'] / 1e7

    # Calculate Daily Garman-Klass Volatility Metrics
    # Mathematical Formula:
    # \sigma_{GK}^2 = 0.5 \cdot \left(\ln\frac{H}{L}\right)^2 - (2\ln 2 - 1) \cdot \left(\ln\frac{C}{O}\right)^2
    log_hl = np.log(cleaned_df['SP100_High'] / cleaned_df['SP100_Low'])
    log_co = np.log(cleaned_df['SP100_Close'] / cleaned_df['SP100_Open'])

    gk_variance = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)

    # 🔧 FIX: on days with a small high-low range but a larger close-open gap,
    # this variance estimate can go slightly negative, which used to send NaN
    # into every downstream row via sqrt() and get silently dropped. Clip to
    # zero first so those rows survive with a (correctly) near-zero volatility.
    gk_variance = gk_variance.clip(lower=0.0)
    cleaned_df['Garman_Klass_Vol'] = np.sqrt(gk_variance)

    # Remove edge artifacts caused by mathematical operations
    cleaned_df.dropna(inplace=True)

    # Enforce strict matrix layout tracking columns
    target_columns = ['SP100_Close', 'Garman_Klass_Vol', 'VIX_Close', 'Credit_Spread', 'Liquidity_Proxy']
    return cleaned_df[target_columns]

processed_data = process_and_align_pipeline(raw_data)
print("✅ Section 1.3 Success: Final Ingestion Matrix Cleaned and Filtered.")
print(f"📋 Analytical Shape: {processed_data.shape} rows/columns.")
print("\n--- SAMPLE VIEW OF PROCESSED MATRIX HEAD ---")
print(processed_data.head())

# ==============================================================================
# SECTION 2.1: STATIONARITY TRANSFORMATION ENGINE
# ==============================================================================
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

def transform_to_stationarity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies mathematical transformations to map raw I(1) fields into
    strictly stationary I(0) continuous distribution tracking fields.
    """
    print("🔄 Initializing mathematical transformation arrays...")
    stationary_df = pd.DataFrame(index=df.index)

    # 1. Target: S&P 100 Index Close -> Daily Log Returns (Δ ln P_t)
    stationary_df['SP100_Returns'] = np.log(df['SP100_Close']).diff()

    # 2. Target Proxy: Garman-Klass Trailing Volatility -> Log First-Difference (Δ ln V_t)
    # 🔧 FIX: Garman_Klass_Vol can be exactly 0.0 on a very quiet trading day
    # (after the Section 1.3 clip), and log(0) is -inf, which propagates into
    # the diff and any model that touches this column. A tiny epsilon keeps
    # the log well-defined without materially changing real observations.
    epsilon = 1e-8
    stationary_df['GK_Vol_Diff'] = np.log(df['Garman_Klass_Vol'] + epsilon).diff()

    # 3. Covariate: VIX Index -> Log First-Difference (Δ ln VIX_t)
    stationary_df['VIX_Diff'] = np.log(df['VIX_Close']).diff()

    # 4. Covariate: Credit Spread -> Absolute First-Difference (Δ S_t)
    stationary_df['Credit_Spread_Diff'] = df['Credit_Spread'].diff()

    # 5. Covariate: Liquidity Proxy -> Absolute First-Difference (Δ L_t)
    stationary_df['Liquidity_Diff'] = df['Liquidity_Proxy'].diff()

    # Eliminate the initial boundary row containing NaN artifacts from differencing
    stationary_df.dropna(inplace=True)
    return stationary_df

# Generate the stationary matrix from your active processed data frame
stationary_matrix = transform_to_stationarity(processed_data)
print(f"📊 Section 2.1 Success: Stationary Matrix formed. Dimensions: {stationary_matrix.shape}")

# ==============================================================================
# SECTION 2.2: AUTOMATED DUAL-TEST STATIONARITY VALIDATOR
# ==============================================================================
def run_dual_stationarity_suite(df: pd.DataFrame):
    """
    Executes the ADF and KPSS tests simultaneously across all transformed variables.
    Target: Reject ADF Null (p < 0.01) AND Fail to Reject KPSS Null (p > 0.05).
    """
    print("\n🔍 EXECUTING DUAL STATIONARITY TESTING SUITE (ADF vs. KPSS)...")
    print("=" * 85)
    print(f"{'Variable Node':<22} | {'ADF p-val':<10} | {'ADF Struct':<11} | {'KPSS p-val':<10} | {'KPSS Struct'}")
    print("=" * 85)

    for col in df.columns:
        # 1. Augmented Dickey-Fuller Test (Maxlag automatically targeted via AIC optimization)
        adf_res = adfuller(df[col], autolag='AIC')
        adf_p = adf_res[1]
        adf_status = "STATIONARY" if adf_p < 0.01 else "UNIT-ROOT"

        # 2. KPSS Test (Evaluated assuming trend-stationarity constant 'c')
        kpss_res = kpss(df[col], regression='c', nlags='auto')
        kpss_p = kpss_res[1]
        kpss_status = "STATIONARY" if kpss_p > 0.05 else "UNIT-ROOT"

        print(f"{col:<22} | {adf_p:<10.4e} | {adf_status:<11} | {kpss_p:<10.4e} | {kpss_status}")
    print("=" * 85)
    print("ℹ️ Note: If both tests read 'STATIONARY', your data is ready for causal evaluation.")

# Run the validation matrix
run_dual_stationarity_suite(stationary_matrix)

# ==============================================================================
# SECTION 2.3: LINEAR DEPENDENCY & CONDITIONAL HETEROSKEDASTICITY AUDITING
# ==============================================================================
def perform_dependency_diagnostics(df: pd.DataFrame):
    """
    Runs joint portmanteau tests and Lagrange Multiplier tests to check for serial
    correlation and variance memory.
    """
    print("\n📈 MEASURING TIME-SERIES LINEAR & VOLATILITY DEPENDENCIES...")
    print("-" * 85)

    # 1. Ljung-Box Test on your Trailing Volatility Differentials (Evaluating 10 Lags)
    lb_res = acorr_ljungbox(df['GK_Vol_Diff'], lags=[10], return_df=True)
    lb_p = lb_res['lb_pvalue'].values[0]
    print(f"🔹 Ljung-Box Q-Test [GK_Vol_Diff] (Lag 10): P-Value = {lb_p:.5e}")
    if lb_p < 0.05:
        print("   👉 Result: Significant linear dependencies remain. Time-series memory is active.")

    # 2. Engle's ARCH Test on Mean-Centered S&P 100 Returns (Evaluating 10 Lags)
    mean_adjusted_returns = df['SP100_Returns'] - df['SP100_Returns'].mean()
    arch_res = het_arch(mean_adjusted_returns, nlags=10)
    arch_p = arch_res[1]  # Extract Lagrange Multiplier p-value
    print(f"🔹 Engle ARCH LM-Test [SP100_Returns] (Lag 10): P-Value = {arch_p:.5e}")
    if arch_p < 0.01:
        print("   👉 Result: Severe ARCH effects confirmed. Volatility clustering is structural.")
    print("-" * 85)

# Execute diagnostics
perform_dependency_diagnostics(stationary_matrix)

# ==============================================================================
# AUTOMATED DEPENDENCY CHECK & INSTALLATION
# ==============================================================================
import sys
import subprocess

try:
    from arch import arch_model
except ModuleNotFoundError:
    print("📦 Package 'arch' missing. Launching automated environment installation...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "arch", "--quiet"])
    from arch import arch_model
    print("✅ Package 'arch' successfully compiled and loaded.")

# ==============================================================================
# SECTION 3.1: OPTIMAL LAG SELECTION & AR-GARCH(1,1) FILTERING ENGINE
# ==============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

def residualize_volatility_pipeline(stationary_df: pd.DataFrame) -> tuple:
    """
    Filters the stationary trailing volatility differential through an AR(p)-GARCH(1,1)
    framework to isolate structural, uncorrelated volatility innovations.
    """
    print("\n⚙️ INITIATING AR-GARCH RESIDUALIZATION FILTER ENGINE...")
    print("-" * 85)

    # Target series: Garman-Klass Volatility Differentials
    vol_series = stationary_df['GK_Vol_Diff'].values

    # 1. Determine optimal AR(p) lag using Bayesian Information Criterion (BIC)
    # 🔧 FIX: the old scan only checked lags 1-5 with AIC, which was landing on
    # lag 5 (the edge of the search window) and still leaving strong residual
    # autocorrelation (Ljung-Box p ~ 1e-7). Widening the search to two trading
    # weeks and using BIC (which penalizes extra lags more) finds a model that
    # actually captures the leftover weekly memory instead of stopping at the wall.
    best_bic = np.inf
    best_lag = 1

    for lag in range(1, 11):
        model = sm.tsa.AutoReg(vol_series, lags=lag, trend='c').fit()
        if model.bic < best_bic:
            best_bic = model.bic
            best_lag = lag

    print(f"🔹 BIC Lag Optimization Match: Optimal AR Memory set to Lag {best_lag}")

    # 2. Construct the combined AR(p)-GARCH(1,1) framework
    # We multiply by 100 to scale the values and prevent optimization convergence errors.
    scaled_series = vol_series * 100.0

    am = arch_model(
        scaled_series,
        mean='AR',
        lags=best_lag,
        vol='Garch',
        p=1,
        q=1,
        dist='normal'
    )

    res = am.fit(disp='off')
    print("\n📈 AR-GARCH ESTIMATION RESULTS SUMMARY:")
    print(res.summary())

    # 3. Extract the clean, standardized residuals (z_t = epsilon_t / sigma_t)
    standardized_residuals = res.resid / res.conditional_volatility

    # Re-align back into a clean Pandas Series matching the original time-index
    vol_innovations = pd.Series(standardized_residuals, index=stationary_df.index)

    return vol_innovations, res

# Run the filtering process
vol_shocks, garch_fit = residualize_volatility_pipeline(stationary_matrix)

# Assemble our clear, structural causal discovery dataframe
causal_df = pd.DataFrame(index=stationary_matrix.index)
causal_df['Vol_Innovations'] = vol_shocks
causal_df['VIX_Diff'] = stationary_matrix['VIX_Diff']
causal_df['Credit_Spread_Diff'] = stationary_matrix['Credit_Spread_Diff']
causal_df['Liquidity_Diff'] = stationary_matrix['Liquidity_Diff']

print(f"\n🚀 Section 3.1 Success: Causal Discovery Matrix Compiled. Dimensions: {causal_df.shape}")

# ==============================================================================
# SECTION 3.2: POST-RESIDUALIZATION DIAGNOSTIC VALIDATION (PATCHED FOR AR LAGS)
# ==============================================================================
def verify_residual_purity(df: pd.DataFrame):
    """
    Executes post-estimation tests to verify that our volatility innovations
    behave as strict mathematical white noise. Removes lag artifacts safely.
    """
    print("\n🧪 EXECUTING POST-RESIDUALIZATION DIAGNOSTIC CHECKS...")
    print("=" * 85)

    clean_series = df['Vol_Innovations'].dropna()
    clean_series = clean_series[np.isfinite(clean_series)]

    # 1. Check for remaining linear correlation in residuals via Ljung-Box (Lag=10)
    lb_res = acorr_ljungbox(clean_series, lags=10, return_df=True)
    lb_p = lb_res['lb_pvalue'].iloc[-1]
    print(f"🔹 Residual Ljung-Box Test (10 Lags Joint P-val): P-value = {lb_p:.5e}")

    # 2. Check for remaining ARCH conditional variance effects (Lag=10)
    arch_res = het_arch(clean_series, nlags=10)
    arch_p = arch_res[1]
    print(f"🔹 Residual Engle ARCH Test (10 Lags LM P-val): P-value = {arch_p:.5e}")
    print("=" * 85)

    if lb_p > 0.05 and arch_p > 0.05:
        print("✅ SUCCESS: Residuals are strictly white noise. Internal time-series memory is fully stripped.")
        print("👉 The dataset is mathematically optimized for structural causal inference.")
    else:
        print("⚠️ WARNING: Residual dependencies remain. Consider expanding GARCH or lag specifications.")

# Run the patched diagnostic suite
verify_residual_purity(causal_df)

