"""Dual stationarity diagnostic suite and serial/ARCH dependence auditor."""

import warnings
from typing import Optional, Union
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss

try:
    from statsmodels.tools.sm_exceptions import InterpolationWarning
except ImportError:
    InterpolationWarning = UserWarning


class DualStationaritySuite:
    """Automated Dual Stationarity Testing Suite running ADF and KPSS tests.

    Applies both the Augmented Dickey-Fuller (ADF) unit-root test and the
    Kwiatkowski-Phillips-Schmidt-Shin (KPSS) stationarity test to evaluate time-series
    persistence. Suppresses lookup table InterpolationWarnings from statsmodels
    when empirical p-values fall outside precomputed asymptotic tables.
    """

    def __init__(
        self,
        adf_autolag: str = "AIC",
        kpss_regression: str = "c",
        kpss_nlags: str = "auto",
        alpha: float = 0.05,
    ):
        """Initialize DualStationaritySuite.

        Parameters
        ----------
        adf_autolag : str, default 'AIC'
            Lag selection criterion for Augmented Dickey-Fuller test.
        kpss_regression : str, default 'c'
            Regression component for KPSS test ('c' for level stationarity, 'ct' for trend).
        kpss_nlags : str or int, default 'auto'
            Number of lags for KPSS test covariance estimator.
        alpha : float, default 0.05
            Significance threshold for hypothesis rejection.
        """
        self.adf_autolag = adf_autolag
        self.kpss_regression = kpss_regression
        self.kpss_nlags = kpss_nlags
        self.alpha = alpha

    def run(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Run ADF and KPSS tests across all series in the input dataframe or series.

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
            Transformed time series to test for stationarity.

        Returns
        -------
        pd.DataFrame
            Summary dataframe indexed by 'Variable' containing:
            ['ADF_p', 'ADF_Status', 'KPSS_p', 'KPSS_Status', 'Verdict'].
        """
        if isinstance(data, pd.Series):
            df = data.to_frame()
        else:
            df = data

        results = []
        for col in df.columns:
            series = df[col].dropna()
            if len(series) < 10:
                raise ValueError(f"Series '{col}' has fewer than 10 non-null observations.")

            # 1. ADF Test (Null hypothesis H0: series has a unit root / is non-stationary)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                adf_res = adfuller(series, autolag=self.adf_autolag)
            adf_p = float(adf_res[1])

            # 2. KPSS Test (Null hypothesis H0: series is level/trend stationary)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                warnings.filterwarnings("ignore", category=InterpolationWarning)
                kpss_res = kpss(
                    series,
                    regression=self.kpss_regression,
                    nlags=self.kpss_nlags,
                )
            kpss_p = float(kpss_res[1])

            # Status assignment
            adf_stat = "STATIONARY" if adf_p < self.alpha else "UNIT-ROOT"
            kpss_stat = "STATIONARY" if kpss_p >= self.alpha else "NON-STATIONARY"

            # Combined verdict
            if adf_p < self.alpha and kpss_p >= self.alpha:
                verdict = "STATIONARY"
            elif adf_p >= self.alpha and kpss_p < self.alpha:
                verdict = "NON-STATIONARY"
            else:
                verdict = "AMBIGUOUS"

            results.append(
                {
                    "Variable": col,
                    "ADF_p": adf_p,
                    "ADF_Status": adf_stat,
                    "KPSS_p": kpss_p,
                    "KPSS_Status": kpss_stat,
                    "Verdict": verdict,
                }
            )

        summary_df = pd.DataFrame(results).set_index("Variable")
        return summary_df

    def __call__(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Execute suite via direct instance invocation."""
        return self.run(data)


class DependencyAuditor:
    """Audits linear serial autocorrelation and non-linear ARCH volatility clustering.

    Executes Ljung-Box Q-test for serial correlation and Engle's Lagrange Multiplier (LM)
    test for autoregressive conditional heteroskedasticity (ARCH effects).
    """

    def __init__(self, lags: int = 10, alpha: float = 0.05):
        """Initialize DependencyAuditor.

        Parameters
        ----------
        lags : int, default 10
            Number of lags to test for serial and ARCH dependence.
        alpha : float, default 0.05
            Significance threshold for hypothesis rejection.
        """
        self.lags = lags
        self.alpha = alpha

    def audit(
        self,
        data: Union[pd.DataFrame, pd.Series],
        lags: Optional[int] = None,
    ) -> dict:
        """Audit dependence structure of series or dataframe.

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
            Time series or dataframe of transformed variables.
        lags : int, optional
            Number of lags to evaluate. If None, uses initialized lags.

        Returns
        -------
        dict
            Dictionary containing:
            - 'ljung_box_p': Ljung-Box test p-value
            - 'arch_lm_p': Engle ARCH LM test p-value
            - 'is_white_noise': bool, True if neither serial correlation nor ARCH effect is detected
        """
        test_lags = lags if lags is not None else self.lags

        # Determine target series for Ljung-Box and Engle ARCH
        if isinstance(data, pd.Series):
            lb_series = data.dropna()
            arch_series = data.dropna()
        elif isinstance(data, pd.DataFrame):
            # Prefer GK_Vol_Diff or Vol_Innovations for Ljung-Box if present
            if "GK_Vol_Diff" in data.columns:
                lb_series = data["GK_Vol_Diff"].dropna()
            elif "Vol_Innovations" in data.columns:
                lb_series = data["Vol_Innovations"].dropna()
            elif "SP100_Returns" in data.columns:
                lb_series = data["SP100_Returns"].dropna()
            else:
                lb_series = data.iloc[:, 0].dropna()

            # Prefer SP100_Returns for ARCH test if present
            if "SP100_Returns" in data.columns:
                arch_series = data["SP100_Returns"].dropna()
            elif "GK_Vol_Diff" in data.columns:
                arch_series = data["GK_Vol_Diff"].dropna()
            elif "Vol_Innovations" in data.columns:
                arch_series = data["Vol_Innovations"].dropna()
            else:
                arch_series = data.iloc[:, 0].dropna()
        else:
            raise TypeError(f"Expected pd.DataFrame or pd.Series, got {type(data)}")

        # 1. Ljung-Box Q-Test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lb = acorr_ljungbox(lb_series, lags=[test_lags], return_df=True)
            lb_p = float(lb["lb_pvalue"].iloc[0])

        # 2. Engle ARCH LM Test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                arch = het_arch(arch_series, nlags=test_lags)
            except TypeError:
                arch = het_arch(arch_series, maxlag=test_lags)
            arch_p = float(arch[1])

        is_white_noise = bool(lb_p >= self.alpha and arch_p >= self.alpha)

        return {
            "ljung_box_p": lb_p,
            "arch_lm_p": arch_p,
            "is_white_noise": is_white_noise,
        }

    def audit_series(self, series: pd.Series, lags: Optional[int] = None) -> dict:
        """Audit a specific single series for serial correlation and ARCH effects."""
        return self.audit(series, lags=lags)

    def __call__(
        self,
        data: Union[pd.DataFrame, pd.Series],
        lags: Optional[int] = None,
    ) -> dict:
        """Execute auditor via direct instance invocation."""
        return self.audit(data, lags=lags)


def run_dual_stationarity_tests(
    df: Union[pd.DataFrame, pd.Series],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Convenience functional interface for DualStationaritySuite.

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Time series data to evaluate.
    alpha : float, default 0.05
        Significance threshold.

    Returns
    -------
    pd.DataFrame
        Dual stationarity test summary table.
    """
    suite = DualStationaritySuite(alpha=alpha)
    return suite.run(df)


def run_dependence_tests(
    df: Union[pd.DataFrame, pd.Series],
    lags: int = 10,
) -> dict:
    """Convenience functional interface for DependencyAuditor.

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Time series data to evaluate.
    lags : int, default 10
        Number of lags.

    Returns
    -------
    dict
        Diagnostics dictionary containing 'ljung_box_p', 'arch_lm_p', 'is_white_noise'.
    """
    auditor = DependencyAuditor(lags=lags)
    return auditor.audit(df, lags=lags)
