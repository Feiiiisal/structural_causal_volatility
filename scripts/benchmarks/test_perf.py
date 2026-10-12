import pandas as pd
import numpy as np
import time

n = 6281
smoothed_risk = pd.Series(np.random.randn(n))

start = time.time()
danger_pct = smoothed_risk.rolling(252, min_periods=20).apply(
    lambda w: (w.iloc[-1] > w).mean(), raw=False
).fillna(0.0).values
end = time.time()

print(f"Time with raw=False: {end - start:.2f} seconds")

start = time.time()
danger_pct_fast = smoothed_risk.rolling(252, min_periods=20).apply(
    lambda w: (w[-1] > w).mean(), raw=True
).fillna(0.0).values
end = time.time()

print(f"Time with raw=True: {end - start:.2f} seconds")
