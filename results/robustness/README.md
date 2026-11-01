# Robustness and uncertainty results

These analyses respond to peer-review requests. They do not replace or re-select the frozen base specification.

## Forecast uncertainty

| Market | RMSE reduction | 95% block-bootstrap CI | HAC loss-difference p-value |
| --- | ---: | ---: | ---: |
| US | 0.96% | [-4.68%, 3.60%] | 0.6305 |
| INDIA | -22.63% | [-35.55%, -1.19%] | 0.0031 |

## Portfolio uncertainty

| Market | Sharpe difference (network - risk parity) | 95% CI | MDD difference | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| US | 0.182 | [-0.490, 0.906] | 7.19 pp | [-5.04, 16.20] pp |
| INDIA | -0.750 | [-1.288, -0.239] | -3.04 pp | [-16.87, 3.42] pp |

## Sensitivity files

Full variant-level results are stored in each market sensitivity.csv file.
The base specification remains the one reported in results/final.
