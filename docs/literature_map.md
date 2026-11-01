# Literature map

The project uses several established modelling traditions. They answer different questions, so the empirical design treats them as complements rather than substitutes.

| Literature / method | What it does well | Limitation for this research question | Role in this project |
| --- | --- | --- | --- |
| Markowitz covariance models | Gives a transparent portfolio-level description of joint risk | Covariance is symmetric and does not identify a transmitter-receiver ordering | Baseline motivation for dependence-aware risk management |
| DCC-GARCH | Allows correlations to change through time | Dynamic dependence remains symmetric and multivariate scaling can be difficult | Background for time-varying dependence |
| Graphical LASSO | Estimates sparse conditional dependence after controlling for the other observed nodes | The precision network is undirected and its standard interpretation is tied to Gaussian graphical models | Main undirected benchmark |
| Billio et al. / Diebold-Yilmaz connectedness | Makes financial connectedness explicitly directional and time-varying | Direction depends on the forecasting/decomposition framework rather than a DAG structural model | Economic motivation for transmitter-receiver structure |
| NOTEARS / DAGMA / DYNOTEARS | Converts directed structure learning into continuous optimization and can incorporate time dependence | Observational direction still depends on structural assumptions and latent confounding remains possible | Directed network estimator |
| CatBoost forecasting | Captures nonlinear predictive interactions and supports feature ablation | Predictive importance is not causal importance | Tests whether network features add out-of-sample forecasting information |
| Volatility-managed portfolio rules | Show that changing risk exposure can improve portfolio outcomes in some settings | A better volatility forecast does not guarantee a better allocation rule | Motivates the separate portfolio hurdle |

The gap addressed by the capstone is empirical: whether a directed volatility representation adds information after factor adjustment and an undirected conditional-dependence benchmark, and whether any incremental information survives both forecasting and portfolio tests.
