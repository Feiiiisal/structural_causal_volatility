# Robustness and uncertainty results

These analyses respond to peer-review requests. They do not replace or re-select the frozen base specification.

## Forecast uncertainty

| Market | RMSE reduction | 95% block-bootstrap CI | HAC loss-difference p-value |
| --- | ---: | ---: | ---: |
| US | 1.48% | [0.72%, 2.52%] | 0.0919 |
| INDIA | -22.40% | [-33.88%, -6.96%] | 0.0145 |

## Portfolio uncertainty

| Market | Sharpe difference (network - risk parity) | 95% CI | MDD difference | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| US | 0.106 | [-0.578, 0.802] | 3.25 pp | [-6.82, 15.42] pp |
| INDIA | -0.776 | [-1.262, -0.287] | -5.21 pp | [-16.60, 3.41] pp |

## Sensitivity files

Full variant-level results are stored in each market sensitivity.csv file.
The base specification remains the one reported in results/final.
