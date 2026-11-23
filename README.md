# Structural Causal Volatility Networks for Adaptive Risk Management

MScFE 690 Capstone Project

**Authors:** Halkano Molu Guracha, Feisal Ali Hassan, Tushar Gahlot

## Research question

Can a directed volatility network add useful risk information beyond factor exposure, correlation and conditional-dependence networks, and does that information improve out-of-sample volatility forecasts or portfolio risk control?

The study compares two fixed, balanced large-cap equity panels over 2016-2025: 20 U.S. stocks and 20 Indian stocks. Both markets follow the same estimation and validation protocol. The panels are designed for cross-market comparison rather than historical index replication.

## Research design

The pipeline has six stages:

1. remove common Fama-French factor exposure from daily equity returns;
2. construct an idiosyncratic volatility panel from the residuals;
3. estimate a rolling directed network with a Student-t DAG model under a DAGMA acyclicity constraint;
4. compare the directed network with correlation and Graphical LASSO benchmarks;
5. measure structural change in the network and test whether network features improve forward-volatility forecasts;
6. evaluate a network-aware portfolio overlay against equal-weight and inverse-volatility benchmarks.

The main estimation window is 252 trading days and is updated every 21 trading days. Model selection and structural-break thresholds are based only on information available at the time. The out-of-sample period is kept separate from CatBoost early stopping and model selection.

## Empirical hypotheses

- **H1:** Factor adjustment materially changes the cross-sectional dependence structure of equity returns.
- **H2:** Directed network centrality contains information that is not redundant with correlation or precision-network centrality.
- **H3:** Large changes in network topology are associated with periods of market stress.
- **H4:** Network features improve out-of-sample forward-volatility forecasts relative to a baseline built from market, volatility, correlation and precision-network features.
- **H5:** A network-aware risk overlay improves downside-risk measures after transaction costs.
- **H6:** The strength and usefulness of the network signal differ between developed and emerging markets.

## Method

### Factor adjustment

For asset \(i\),

\[
R_{i,t}-R_{f,t}=\alpha_i+\beta_{i,M}(MKT-RF)_t+\beta_{i,S}SMB_t+\beta_{i,H}HML_t+\varepsilon_{i,t}.
\]

Betas are estimated chronologically. After the initial training window, later residuals use only information that was available before the corresponding date.

### Volatility network

The graph is estimated from \(|\varepsilon_{i,t}|\), a residual-based volatility proxy. The directed model uses a Student-t loss to allow for heavy-tailed innovations. A smooth log-determinant acyclicity constraint is used to estimate the contemporaneous graph.

### Benchmarks

The directed network is compared with:

- pairwise correlation networks;
- Graphical LASSO partial-correlation networks;
- conventional rolling volatility and momentum features.

Low correlation between network summaries is treated as evidence of non-redundancy, not as proof of causality.

### Structural change

For consecutive adjacency matrices \(W_t\) and \(W_{t-1}\),

\[
D_t=\lVert W_t-W_{t-1}\rVert_F.
\]

The structural-change threshold is estimated from earlier drift observations only.

### Forecasting

CatBoost predicts five-day forward realized volatility. The baseline model uses conventional market and dependence features. The augmented model adds directed-network centrality, network drift and break indicators.

### Portfolio test

The portfolio experiment compares:

- equal-weight buy and hold;
- inverse-volatility risk parity;
- a network-aware allocation that reduces exposure to strong volatility transmitters and can reduce market exposure during stressed regimes.

Transaction costs are applied to active rebalancing rules.

## Final empirical results

The frozen, seeded 2016-2025 experiment is archived under [results/final](results/final). The submission data snapshot is locked to GitHub Actions run **45203001872**; later robustness checks use those exact files rather than redownloading market history.

| Metric | U.S. | India |
| --- | ---: | ---: |
| Assets | 20 | 20 |
| Daily observations | 2,513 | 2,469 |
| Baseline volatility RMSE | 0.13895 | 0.11122 |
| Network-augmented RMSE | 0.13690 | 0.13614 |
| RMSE change | 1.48% improvement | 22.40% deterioration |
| Baseline OOS R² | 0.173 | -0.197 |
| Network OOS R² | 0.198 | -0.794 |
| Risk-parity Sharpe | 1.131 | 1.650 |
| Network-overlay Sharpe | 1.237 | 0.874 |
| Risk-parity max drawdown | -19.58% | -16.43% |
| Network-overlay max drawdown | -16.33% | -21.64% |

The final results use the 23 November 2026 frozen price/return snapshot; the previous 1 November snapshot differs. Do not mix result tables or robustness intervals between runs.

The result is market dependent. In the U.S. panel the network block produces a small forecast gain and the portfolio overlay has a better Sharpe ratio and smaller maximum drawdown than risk parity. In India the same feature block materially worsens the volatility forecast and the portfolio overlay underperforms risk parity.

The project therefore does not treat directed network structure as automatically valuable. Forecasting, structural description and portfolio usefulness are evaluated as separate claims.

## Reviewer-motivated robustness

The frozen base specification is not re-selected after seeing these checks. Uncertainty and sensitivity results are archived under [results/robustness](results/robustness).

- **U.S. forecast:** 1.48% RMSE reduction. The 95% moving-block bootstrap interval is **+0.72% to +2.52%**; the HAC loss-difference test gives **p = 0.092**. The two uncertainty procedures are not equally decisive at the 5% level.
- **India forecast:** 22.40% RMSE deterioration, with a 95% interval of **-33.88% to -6.96%** and **p = 0.015**. The augmented model remains worse than the baseline in the current frozen experiment.
- **U.S. portfolio:** network-minus-inverse-volatility Sharpe difference **+0.106**, 95% interval **[-0.578, 0.802]**. The sample point estimate is favourable, but its uncertainty interval contains zero.
- **India portfolio:** Sharpe difference **-0.776**, 95% interval **[-1.262, -0.287]**. The network portfolio underperforms the inverse-volatility benchmark in the base run.
- **Sensitivity on the 23 November data:** All six U.S. forecast-related non-base variants have negative RMSE improvement, despite the positive base result. In India, five of the six forecast-related variants worsen RMSE; the 15-stock variant improves RMSE by 15.58%, although its baseline and network R² values remain negative. The Indian network portfolio has lower Sharpe than the benchmark in every non-base portfolio variant.

The robustness exercise therefore narrows the claim. The directed network is a distinct structural representation, but predictive and portfolio value are market- and specification-dependent. The Indian underperformance is the most stable economic finding.

## Repository layout

```text
catboost_dagma/
├── benchmark/       # correlation/precision-network comparisons
├── breaks/          # rolling network-drift detection
├── dagma/           # directed network model and rolling solver
├── data/            # return/factor loading and residualization
├── ml/              # feature construction and CatBoost ablation
├── notebooks/       # analysis notebook
├── scripts/         # experiment runner
├── strategy/        # portfolio experiment
├── tests/           # unit and chronology tests
├── visualization/   # report figures
└── pipeline.py      # end-to-end research pipeline

docs/
├── research_design.md
├── literature_map.md
├── robustness_protocol.md
├── final_data_lock.md
└── reproducibility.md
```

## Data

The repository does not distribute licensed market data.

The pipeline accepts CSV or pickle return panels and CSV/pickle factor panels. Set the paths with environment variables:

```bash
export EQUITY_DATA_PATH=/path/to/equity_returns.pkl
export FF_DATA_PATH=/path/to/fama_french_daily.csv
```

For the U.S. panel, Fama-French daily factors can also be downloaded through `pandas-datareader` by running the loader with `offline=False`.

The final U.S. run uses Kenneth French daily factors. The Indian run uses the survivorship-adjusted daily factor series from IIM Ahmedabad. The price panels are fixed large-cap samples with continuous coverage; they are not reconstructed historical S&P 100 or Nifty 50 memberships.

More detail is in [docs/research_design.md](docs/research_design.md).

## Installation

```bash
git clone https://github.com/Feiiiisal/structural_causal_volatility.git
cd structural_causal_volatility

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Windows:

```powershell
.venv\Scripts\activate
```

## Run the study

```bash
python catboost_dagma/scripts/run_experiment.py \
  --returns-path /path/to/equity_returns.pkl \
  --factor-path /path/to/factors.csv
```

Or from Python:

```python
from catboost_dagma.pipeline import CatBoostDagmaPipeline

results = CatBoostDagmaPipeline(
    returns_path="/path/to/equity_returns.pkl",
    factor_path="/path/to/factors.csv",
).run()
```

## Tests

```bash
pytest -q catboost_dagma/tests
```

The test suite checks data alignment, factor residualization, rolling graph estimation, precision-network construction, structural-break chronology, CatBoost train/validation/test separation and the portfolio backtest. GitHub Actions runs the same suite on pushes and pull requests.

## Reproducibility

The experiment is designed so that data preparation, graph estimation, forecasting and portfolio evaluation can be rerun from the same input panels. Chronology-sensitive tests are included to catch look-ahead leakage.

See [docs/reproducibility.md](docs/reproducibility.md) for the full protocol.

## Authors

This is a joint MScFE 690 capstone project by:

- **Halkano Molu Guracha**
- **Feisal Ali Hassan**
- **Tushar Gahlot**

Research design, implementation, validation, interpretation and manuscript preparation are treated as group responsibilities.

## Citation

If you refer to this project, use the citation in [CITATION.cff](CITATION.cff).
