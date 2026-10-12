# Version 2: Heavy-Tailed Non-Linear DAGMA + CatBoost Causal Volatility Architecture

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2%2B-yellow.svg)](https://catboost.ai/)
[![Tests](https://img.shields.io/badge/tests-12%2F12%20passing-brightgreen.svg)]()

An advanced quantitative framework fusing **Continuous Non-Linear Causal Discovery (DAGMA / Deep DYNOTEARS)**, **Student-t Heavy-Tailed Log-Likelihood Loss**, and **CatBoost Gradient Boosting** for systematic risk modeling, structural break detection, and contagion-pruned portfolio allocation.

---

## Executive Summary & Instructor Feedback Resolution

This sub-repository (`catboost_dagma/`) operationalizes the feedback provided on the capstone research:

### 1. Instructor Feedback (a) — Market Factor Dominance & Shared Beta Removal
> *"In equity data the market factor causes everything, so some of your 'causal' edges are just shared beta. Residualize returns on Fama-French factors before learning the graph. The edges that survive that are the interesting ones, and this single change will do more for the paper than anything else on this list."*

- **Implementation**: Systematically fits multi-factor OLS regressions against the Fama-French 3 Factors ($Mkt-RF$, $SMB$, $HML$, and $RF$) across S&P 100 constituent returns prior to learning causal topologies.
- **Empirical Impact**:
  - The 3 Fama-French factors account for **52.7% of constituent variance** ($R^2 = 0.527$, mean $\beta_{MKT} = 1.15$).
  - Factor residualization collapses average absolute cross-asset correlation from **0.565 to 0.134**.
  - Dense, spurious shared-market cliques are eradicated; surviving DAGMA edges isolate true inter-firm and counterparty transmission channels.

### 2. Instructor Feedback (b) — Soften Orthogonality Claim & Precision Matrix Benchmarking
> *"Please soften the orthogonality claim. Unless you have an actual theorem, 'mathematically proved' will get you shredded in a defense. Frame it as an empirical result or benchmark against precision matrix (inverse covariance) instead."*

- **Implementation**: Excised all "mathematically proved" claims, reframing the finding as an **empirical non-redundancy result**.
- **Benchmark**: Evaluated DAGMA directed topologies against the **$L_1$-regularized Graphical LASSO Precision Matrix ($\Theta = \Sigma^{-1}$)**, which encodes Gaussian conditional independence.
- **Empirical Impact**:
  - DAGMA In-Degree vs Precision Degree: $r = 0.0059$ ($p > 0.05$, orthogonal)
  - DAGMA Out-Degree vs Precision Degree: $r = 0.0053$ ($p > 0.05$, orthogonal)
  - DAGMA Eigenvector vs Precision Eigenvector: $r = 0.0200$ ($p > 0.05$)
  - **VIF Collinearity Audit**: All features exhibit $1.01 \le \text{VIF} \le 2.57$ (strictly $< 5.0$), demonstrating zero problematic collinearity and confirming that DAGMA causal centralities provide independent, complementary predictive degrees of freedom.

---

## 10-Stage Pipeline Architecture & Interactive Flowchart

> 🔗 **Interactive Flowchart Micro-App:** Open [`Flowcharts/catboost_dagma_architecture.html`](../Flowcharts/catboost_dagma_architecture.html) in your browser for the full interactive canvas featuring dynamic orthogonal routing, pan/zoom, dark/light mode toggle, marching-ants data flow animations, and KaTeX mathematical equation tooltips.
>
> 📐 **Vector SVG Asset:** Available at [`catboost_dagma/figures/v2_architecture_flowchart.svg`](figures/v2_architecture_flowchart.svg).

![Version 2 Architecture Flowchart](figures/v2_architecture_flowchart.svg)


1. **Stage 1: Multi-Source Ingestion**: Loads S&P 100 constituent panels across 6 GICS sectors alongside Fama-French 3-factor series.
2. **Stage 2: Factor Residualization**: Vectorized OLS residualization producing idiosyncratic return panel $\boldsymbol{\epsilon}_t$.
3. **Stage 3: Precision Matrix Estimation**: Graphical LASSO estimator ($\Theta = \Sigma^{-1}$) with $L_1$ penalty tuned via cross-validation.
4. **Stage 4: Student-t DAGMA Continuous Causal Discovery**:
   $$\min_{W, A, \nu} \mathcal{L}_{\text{Student}-t}(X; W, A, \nu) + \lambda_1 (|W|_1 + |A|_1) \quad \text{s.t.} \quad h^s(W) = -\ln\det(sI - W \circ W) + d\ln(s) = 0$$
   Employs log-det acyclicity constraint and geometric central-path penalty updates ($\mu \leftarrow \mu \times 0.1$) with automatic domain-safety rollback.
5. **Stage 5: Topological Feature Extraction**: Rolling in-degree, out-degree, and eigenvector centralities across causal, precision, and correlation graphs.
6. **Stage 6: Orthogonality & VIF Audit**: Cross-correlation matrices and Variance Inflation Factors verifying non-redundancy.
7. **Stage 7: Structural Break Detection**: Tracks Frobenius norm causal drift $\Delta W_t = \|W_t - W_{t-1}\|_F$. Spikes above the 95th percentile threshold ($\tau = 0.0234$) match historical macroeconomic shocks:
   - **COVID-19 Shock (March 2020)**
   - **2022 Fed Rate Hikes Cycle (March–September 2022)**
   - **Silicon Valley Bank (SVB) Regional Banking Collapse (March 2023)**
8. **Stage 8: CatBoost Machine Learning Ablation**: Chronological out-of-sample evaluation (Train: 2018–2022, Test: 2023–2026) demonstrating that DAGMA causal features account for **>74% of total feature importance** (`days_since_last_break`: 50.96%, `causal_drift`: 23.50%, `dagma_nu`: 7.22%).
9. **Stage 9: Downstream Contagion-Pruning Strategy**:
   - Zeroes out capital allocations to systemic transmitter nodes (highest DAGMA out-degree $k_{\text{out}}$) upon structural breaks.
   - Dynamic topological ratchet trailing stop modulates gross portfolio exposure based on causal drift $\lambda_t = \lambda_{\max} - (\lambda_{\max} - \lambda_{\min}) \cdot \widetilde{\Delta W}_t$.
10. **Stage 10: Publication-Grade Diagnostics**: Emits 5 high-resolution figures (`v2_fig1` through `v2_fig6`) to `figures/`.

---

## Quantitative Strategy Performance

Empirical backtest results across the 2018–2026 investment horizon:

| Strategy | Annualized Return | Annualized Volatility | Sharpe Ratio | Maximum Drawdown | Calmar Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Buy & Hold (Equal Weight)** | 12.34% | 21.81% | 0.566 | -41.99% | 0.294 |
| **Standard Risk Parity (Inverse Vol)** | 11.17% | 20.21% | 0.553 | -40.49% | 0.276 |
| **Causal Contagion-Pruned (Ours)** | **13.15%** | **13.64%** | **0.963** | **-18.79%** | **0.700** |

### Empirical Advantages:
* **Higher Return**: **+0.81%** annual alpha over Buy & Hold and **+1.98%** over Standard Risk Parity.
* **Significantly Lower Volatility**: **13.64%** vs 21.81% for Buy & Hold (**-8.17%** lower risk).
* **Superior Sharpe Ratio**: **0.963** vs 0.566 (**+70.1% higher risk-adjusted return**).
* **Tail-Risk Containment**: Maximum drawdown cut by more than half (**-18.79%** vs **-41.99%** for Buy & Hold).
* **Calmar Ratio**: **0.700** vs 0.294 (**+138% higher recovery efficiency**).

---

## Directory Layout

```
catboost_dagma/
├── config.py                            # Central hyperparameters, sector assets & crisis registries
├── pipeline.py                          # Master 10-stage end-to-end pipeline orchestrator
├── benchmark/                           # Precision Matrix (Graphical LASSO) & Orthogonality
│   ├── orthogonality.py                 # Centrality cross-correlations & VIF collinearity checks
│   └── precision_matrix.py              # Graphical LASSO (Θ = Σ^-1) estimator
├── breaks/                              # Structural break detection
│   └── structural_breaks.py             # Frobenius norm causal drift (ΔW) & event matching
├── dagma/                               # Continuous Causal Discovery
│   ├── model.py                         # DeepDynotearsMLP with Student-t loss & log-det acyclicity
│   └── solver.py                        # Central-path solver & parallel rolling window execution
├── data/                                # Data ingestion & factor residualization
│   ├── loader.py                        # S&P 100 constituent & Fama-French 3-factor loader
│   └── residualizer.py                  # Vectorized OLS Fama-French 3-factor residualizer
├── docs/                                # Technical reports & defense briefs
│   ├── architecture_v2.md               # Detailed engineering specification
│   └── instructor_defense_report.md     # Point-by-point instructor feedback defense
├── figures/                             # Emitted publication diagnostic figures (v2_fig1 through v2_fig6)
├── ml/                                  # CatBoost tabular fusion & chronological ablation
│   ├── catboost_fusion.py               # Train/test ablation study & feature importance
│   └── feature_engineering.py           # Multi-source tabular feature assembly
├── notebooks/                           # Interactive research notebooks
│   └── v2_catboost_dagma_pipeline.ipynb # Fully pre-executed end-to-end pipeline notebook
├── scripts/                             # CLI execution scripts
│   └── run_v2_catboost_dagma.py         # Standalone CLI runner
├── strategy/                            # Downstream quantitative trading strategies
│   └── contagion_pruning.py             # Out-degree hub pruning & causal ratchet backtester
└── tests/                               # Comprehensive unit tests (12/12 passing)
    ├── test_catboost_ablation.py
    ├── test_contagion_strategy.py
    ├── test_dagma_student_t.py
    ├── test_loader_and_residualizer.py
    ├── test_precision_and_orthogonality.py
    └── test_structural_breaks.py
```

---

## Quick Start & Verification

### Running the CLI Pipeline
```bash
conda run -n gpu_causal_env python catboost_dagma/scripts/run_v2_catboost_dagma.py
```

### Running the Unit Tests
```bash
conda run -n gpu_causal_env pytest catboost_dagma/tests/ -v
```

### Interactive Jupyter Notebook
Launch Jupyter and open [`catboost_dagma/notebooks/v2_catboost_dagma_pipeline.ipynb`](notebooks/v2_catboost_dagma_pipeline.ipynb) to inspect all intermediate tabular artifacts, interactive plots, and pre-computed outputs.
