# Frozen final data snapshot

The empirical results reported in the final manuscript use the frozen market inputs produced by GitHub Actions run **41350658046**.

This run is the data source of truth for the submission. Reviewer-motivated robustness analyses must use these archived inputs rather than downloading prices again.

## U.S. panel

- 20 large-cap equities
- 2,513 aligned daily observations
- sample: 5 January 2016 to 31 December 2025
- price source: Yahoo Finance via yfinance
- factor source: Kenneth R. French Data Library

SHA-256:

- adjusted_close.csv: `f4f32620134a1b20c57f2bdb66473b466dd6dd9806c1c56c0b582a754b873be4`
- returns.csv: `293f46745abe5dfb55890575259045d8305ead3764fab6f4bfacf0797db89147`
- factors.csv: `fcec9f01f011730e5ddb0315a03de8315c09da3e2b089e1cd02625991f0d5a0f`

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

The raw frozen files are retained in the private workflow artifacts and in the final submission source/results ZIP. Live redownloads are not used for the uncertainty or sensitivity results.
