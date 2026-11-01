# Experiment outputs

Each run of `catboost_dagma/scripts/run_experiment.py` creates a dated folder here.

The folder contains:

- `metrics.json`: run settings and headline forecast/portfolio metrics;
- `tables/`: factor, network, break, feature-importance and portfolio tables;
- `figures/`: report-ready diagnostic and performance figures.

Generated runs are ignored by Git by default. For a submitted paper, archive the exact run used for the manuscript together with the frozen data manifest and commit SHA.
