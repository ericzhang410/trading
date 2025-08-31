import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Union, Mapping, Tuple, Literal
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
import warnings

def prep_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().strip("[]").lower().replace(" ", "_") for c in out.columns]
    out["date"] = df.index
    return out

def zero_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = (df.index - df.index.min()).days
    out["avg"] = (df["open"] + df["close"]) / 2
    return out[["date", "avg"]]

def date_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = df.index
    out["avg"] = (df["open"] + df["close"]) / 2
    return out[["date", "avg"]]

def log_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = df["date"]
    out["log_returns"] = np.log(df["avg"] / df["avg"].shift(1))
    return out[["date", "log_returns"]].dropna()

def s_param(ds: pd.Series):
    r = ds.dropna().astype(float)
    r = r - r.mean()  # mean-center

    # 1) persistence from ACF of squared returns
    rho1 = acf(r**2, nlags=1, fft=True)[1]
    p = np.clip(rho1, 0.60, 0.995)

    # 2) ARCH slope from OLS on r_t^2 ~ 1 + r_{t-1}^2
    y = (r**2).iloc[1:].values
    x = (r**2).shift(1).iloc[1:].values
    X = sm.add_constant(x)
    phi = sm.OLS(y, X).fit().params[1]

    alpha = float(np.clip(phi, 0.01, 0.15))
    beta  = float(np.clip(p - alpha, 0.70, 0.995 - 1e-6))  # keep α+β<1
    # If clipping pushed beta to bound, re-adjust alpha to keep α+β=p (but <1)
    if alpha + beta >= 0.999:
        beta = 0.999 - alpha

    # 3) omega from long-run variance
    s2 = float(np.var(r, ddof=1))
    omega = max(1e-12, s2 * (1 - (alpha + beta)))

    return omega, alpha, beta


def QMLER(param, log_returns, eps=1e-6):
    w, a, b = map(float, param)

    # Feasibility checks
    if w <= 0 or a < 0 or b < 0:
        return 1e12
    denom = 1.0 - a - b
    if denom <= eps:
        return 1e12

    # Clean & scale data
    r = np.asarray(log_returns, float).ravel()
    if r.size < 2: 
        return 1e12
    r = r[np.isfinite(r)]
    r = (r - r.mean()) * 100.0

    # Init variance safely
    s2 = w / denom
    if not np.isfinite(s2) or s2 <= 0:
        return 1e12

    nll = np.log(s2) + (r[0]**2)/s2
    for t in range(1, r.size):
        s2 = w + a*(r[t-1]**2) + b*s2
        if not np.isfinite(s2) or s2 <= 0:
            return 1e12
        nll += np.log(s2) + (r[t]**2)/s2

    return nll


def QMLE(param, log_returns, eps):

    bounds = [(eps, 5), (eps, 1 - eps), (eps, 1 - eps)]
    cons = [{'type': 'ineq', 'fun': lambda p, e=eps: 1 - e - (p[1] + p[2])}]
    out = minimize(
        QMLER, param,
        args=(log_returns,),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options=dict(maxiter=1000, ftol=1e-12, disp=False),
    )
    out.param_names = ("omega", "alpha", "beta")
    return out

def simul(param, last_sigma2, last_observ, t, rng=None):
    """
    Simulate t steps of a GARCH(1,1) process.

    Parameters
    ----------
    param : (omega, alpha, beta)
        Positive GARCH(1,1) parameters.
    last_sigma2 : float
        Previous conditional variance (> 0), i.e., sigma_{0}^2.
    last_observ : float
        Previous observed return y_{0}.
    t : int
        Number of periods to simulate.
    rng : np.random.Generator, optional
        Random generator for reproducibility (np.random.default_rng() if None).

    Returns
    -------
    sim : np.ndarray of shape (t,)
        Simulated returns.
    sigma2 : np.ndarray of shape (t,)
        Conditional variances.
    """
    omega, alpha, beta = map(float, param)

    if not (omega > 0 and alpha > 0 and beta > 0):
        raise ValueError("All params (omega, alpha, beta) must be > 0.")
    if last_sigma2 <= 0:
        raise ValueError("last_sigma2 must be > 0.")
    if t <= 0:
        raise ValueError("t must be a positive integer.")
    if alpha + beta >= 1:
        warnings.warn("alpha + beta >= 1 (nonstationary); simulation may explode.", RuntimeWarning)

    if rng is None:
        rng = np.random.default_rng()

    innov = rng.standard_normal(t)
    sigma2 = np.empty(t, dtype=float)
    sim = np.empty(t, dtype=float)

    # First step (Python is 0-indexed)
    sigma2[0] = omega + alpha * (last_observ ** 2) + beta * last_sigma2
    if not np.isfinite(sigma2[0]) or sigma2[0] <= 0:
        raise FloatingPointError("Initial sigma2 is non-positive or NaN; check params/inputs.")
    sim[0] = innov[0] * np.sqrt(sigma2[0])

    # Recursion
    for i in range(1, t):
        sigma2[i] = omega + alpha * (sim[i - 1] ** 2) + beta * sigma2[i - 1]
        if not np.isfinite(sigma2[i]) or sigma2[i] <= 0:
            raise FloatingPointError(f"sigma2 became invalid at step {i}.")
        sim[i] = innov[i] * np.sqrt(sigma2[i])

    return sim, sigma2

