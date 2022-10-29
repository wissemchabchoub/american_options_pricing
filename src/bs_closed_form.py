import math
import numpy as np
from scipy.stats import norm
from scipy.stats import mvn
from scipy.stats import norm
import math


def bs_american_closed_form_pricer(S, K, T, r, b, sigma, call):
    """
    The bs_american_closed_form_pricer function computes the price of an American option using a closed-form formula.
    The function takes as input parameters:
        S, K, T, r, b (the cost of carry), sigma (volatility), and call (True for call option; False for put)

    Parameters
    ----------
        S
            Represent the underlying asset price
        K
            Strike
        T
            Compute the time to expiry of the option
        r
            Compute the interest rate for the closed form solution
        b
            Cost of carry
        sigma
            Volatility
        call
            Distinguish between a call and put option

    Returns
    -------

        The price of a american option

    """

    if call == False:
        S, K = K, S
        r, b = r-b, -b

    # compute params
    t__sqrt = math.sqrt(T)
    d1 = (math.log(S / K) + (b + (sigma * sigma) / 2) * T) / (sigma * t__sqrt)
    d2 = d1 - sigma * t__sqrt

    price_euro = S * math.exp((b - r) * T) * norm.cdf(d1) - \
        K * math.exp(-r * T) * norm.cdf(d2)

    # if b >= r, it is not optimal to exercise before maturity
    if b >= r:
        return price_euro

    # Second step
    v2 = sigma ** 2
    t1 = 0.5 * (math.sqrt(5) - 1) * T
    t2 = T

    beta_inside = ((b / v2 - 0.5) ** 2) + 2 * r / v2

    beta_inside = abs(beta_inside)
    beta = (0.5 - b / v2) + math.sqrt(beta_inside)
    b_infinity = (beta / (beta - 1)) * K
    b_zero = max(K, (r / (r - b)) * K)

    h1 = -(b * t1 + 2 * sigma * math.sqrt(t1)) * \
        ((K ** 2) / ((b_infinity - b_zero) * b_zero))
    h2 = -(b * t2 + 2 * sigma * math.sqrt(t2)) * \
        ((K ** 2) / ((b_infinity - b_zero) * b_zero))

    i1 = b_zero + (b_infinity - b_zero) * (1 - math.exp(h1))
    i2 = b_zero + (b_infinity - b_zero) * (1 - math.exp(h2))

    alpha1 = (i1 - K) * (i1 ** (-beta))
    alpha2 = (i2 - K) * (i2 ** (-beta))

    # immediate exercise check
    if S >= i2:
        price = S - K
    else:
        # main computation
        price = (alpha2 * (S ** beta)
                 - alpha2 * phi(S, t1, beta, i2, i2, r, b, sigma)
                 + phi(S, t1, 1, i2, i2, r, b, sigma)
                 - phi(S, t1, 1, i1, i2, r, b, sigma)
                 - K * phi(S, t1, 0, i2, i2, r, b, sigma)
                 + K * phi(S, t1, 0, i1, i2, r, b, sigma)
                 + alpha1 * phi(S, t1, beta, i1, i2, r, b, sigma)
                 - alpha1 * psi(S, t2, beta, i1, i2, i1, t1, r, b, sigma)
                 + psi(S, t2, 1, i1, i2, i1, t1, r, b, sigma)
                 - psi(S, t2, 1, K, i2, i1, t1, r, b, sigma)
                 - K * psi(S, t2, 0, i1, i2, i1, t1, r, b, sigma)
                 + K * psi(S, t2, 0, K, i2, i1, t1, r, b, sigma))

    price = max(price, price_euro)

    return price


def psi(fs, t2, gamma, h, i2, i1, t1, r, b, v):
    vsqrt_t1 = v * math.sqrt(t1)
    vsqrt_t2 = v * math.sqrt(t2)

    bgamma_t1 = (b + (gamma - 0.5) * (v ** 2)) * t1
    bgamma_t2 = (b + (gamma - 0.5) * (v ** 2)) * t2

    d1 = (math.log(fs / i1) + bgamma_t1) / vsqrt_t1
    d3 = (math.log(fs / i1) - bgamma_t1) / vsqrt_t1

    d2 = (math.log((i2 ** 2) / (fs * i1)) + bgamma_t1) / vsqrt_t1
    d4 = (math.log((i2 ** 2) / (fs * i1)) - bgamma_t1) / vsqrt_t1

    e1 = (math.log(fs / h) + bgamma_t2) / vsqrt_t2
    e2 = (math.log((i2 ** 2) / (fs * h)) + bgamma_t2) / vsqrt_t2
    e3 = (math.log((i1 ** 2) / (fs * h)) + bgamma_t2) / vsqrt_t2
    e4 = (math.log((fs * (i1 ** 2)) / (h * (i2 ** 2))) + bgamma_t2) / vsqrt_t2

    tau = math.sqrt(t1 / t2)
    lambda1 = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * (v ** 2))
    kappa = (2 * b) / (v ** 2) + (2 * gamma - 1)

    psi = math.exp(lambda1 * t2) * (fs ** gamma) * (cbnd(-d1, -e1, tau)
                                                    - ((i2 / fs) ** kappa) *
                                                    cbnd(-d2, -e2, tau)
                                                    - ((i1 / fs) ** kappa) *
                                                    cbnd(-d3, -e3, -tau)
                                                    + ((i1 / i2) ** kappa) * cbnd(-d4, -e4, -tau))
    return psi


def phi(fs, t, gamma, h, i, r, b, v):
    d1 = -(math.log(fs / h) + (b + (gamma - 0.5)
           * (v ** 2)) * t) / (v * math.sqrt(t))
    d2 = d1 - 2 * math.log(i / fs) / (v * math.sqrt(t))

    lambda1 = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * (v ** 2))
    kappa = (2 * b) / (v ** 2) + (2 * gamma - 1)

    phi = math.exp(lambda1 * t) * (fs ** gamma) * \
        (norm.cdf(d1) - ((i / fs) ** kappa) * norm.cdf(d2))

    return phi


def cbnd(a, b, rho):
    down = np.array([0, 0])
    up = np.array([a, b])
    infin = np.array([0, 0])
    correlation = rho
    error, value, inform = mvn.mvndst(down, up, infin, correlation)
    return value
