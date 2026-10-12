import pandas as pd
import numpy as np
import time

n = 6281
backtest_df = pd.DataFrame({
    'Close': np.random.randn(n),
    'Base_Vol': np.random.randn(n),
    'Causal_Multiplier': np.random.randn(n),
    'Close_MA5': np.random.randn(n)
})

standard_stop_arr = np.zeros(n)
causal_stop_arr = np.zeros(n)
standard_state = np.ones(n)
causal_state = np.ones(n)
static_multiplier = 2.0

start = time.time()
for i in range(n):
    current_close = backtest_df['Close'].iloc[i]
    current_vol = backtest_df['Base_Vol'].iloc[i]
    current_causal_mult = backtest_df['Causal_Multiplier'].iloc[i]
    
    raw_standard_stop = current_close - (static_multiplier * current_vol * current_close)
    raw_causal_stop = current_close - (current_causal_mult * current_vol * current_close)
    
    if i == 0:
        standard_stop_arr[i] = raw_standard_stop
        causal_stop_arr[i] = raw_causal_stop
    else:
        # standard state logic ... (omitted branches to keep it simple, just taking time of iloc)
        pass
end = time.time()
print(f"Time for iloc loop: {end - start:.4f} s")

start = time.time()
# vector equivalent extraction
close_arr = backtest_df['Close'].values
vol_arr = backtest_df['Base_Vol'].values
mult_arr = backtest_df['Causal_Multiplier'].values
for i in range(n):
    current_close = close_arr[i]
    current_vol = vol_arr[i]
    current_causal_mult = mult_arr[i]
    
    raw_standard_stop = current_close - (static_multiplier * current_vol * current_close)
    raw_causal_stop = current_close - (current_causal_mult * current_vol * current_close)
end = time.time()
print(f"Time for values loop: {end - start:.4f} s")

