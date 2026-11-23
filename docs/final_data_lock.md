# Frozen final data snapshot

The empirical results reported in the final manuscript use the frozen market inputs produced by GitHub Actions run **45203001872**.

This run is the data source of truth for the submission. The earlier run 41350658046 used different frozen prices and returns; its forecast, portfolio, and robustness values must not be mixed with those of the new run. Reviewer-motivated robustness analyses must use these archived inputs rather than downloading prices again.

## U.S. panel

- 20 large-cap equities
- 2,513 aligned daily observations
- sample: 5 January 2016 to 31 December 2025
- price source: Yahoo Finance via yfinance
- factor source: Kenneth R. French Data Library

SHA-256:

- adjusted_close.csv: `93ead7b29c0ba51eb3527e55073ea5b2edecafecad23fa25def682471301902c`
- returns.csv: `a37ec8c17573d6a69b48bd584e7496460897c449596e63f35c6775413eaaf65a`
- factors.csv: `08c447fc1a2afd8468631e808116e2877d0fc2ffe6fdfa071670533adaaf02df`

## Indian panel

- 20 NSE-listed large-cap equities
- 2,469 aligned daily observations
- sample: 4 January 2016 to 31 December 2025
- price source: Yahoo Finance via yfinance
- factor source: IIM Ahmedabad Indian Fama-French and Momentum Data Library, 2025-12 release

SHA-256:

- adjusted_close.csv: `e55068f940f4b0020cb29d6e4a6c564757efc7693f257a58501c310ab5348720`
- returns.csv: `90d11903fbeb73ad9ffe014facedf89ce84a4d66e0e098d19a4ec5cf71d6c5d0`
- factors.csv: `08c447fc1a2afd8468631e808116e2877d0fc2ffe6fdfa071670533adaaf02df`

The raw frozen files are retained in the private workflow artifacts and in the final submission source/results ZIP. The robustness and sensitivity results must be generated from the same 23 November frozen data. Live redownloads are not used for the final analysis.
