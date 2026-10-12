import sys
import warnings
warnings.filterwarnings('ignore')

with open('temp.py') as f:
    code = f.read()

# Execute temp.py to load stationary_matrix into globals
exec(code, globals())

import numpy as np
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox

vol_series = stationary_matrix['GK_Vol_Diff'].values
scaled_series = vol_series * 100.0

print("\n--- SEARCHING FOR BEST AR-GARCH SPECIFICATION ---")
for p in range(1, 3):
    for q in range(1, 3):
        for lag in range(1, 15):
            try:
                am = arch_model(scaled_series, mean='AR', lags=lag, vol='Garch', p=p, q=q, dist='normal')
                res = am.fit(disp='off')
                std_resid = res.resid / res.conditional_volatility
                lb_p = acorr_ljungbox(std_resid.dropna(), lags=[10], return_df=True)['lb_pvalue'].values[0]
                if lb_p > 0.05:
                    print(f"✅ FOUND! AR Lag: {lag}, GARCH(p={p}, q={q}) -> Ljung-Box P-val: {lb_p:.4f}")
            except Exception as e:
                pass
print("--- SEARCH COMPLETE ---")
