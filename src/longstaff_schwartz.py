import numpy as np
import random
import matplotlib.pyplot as plt


def bs_mvt(S, n, sigma, r, T):
    L = []
    L.append(S)
    for i in range(1, n):
        L.append(L[i-1]*(1+r*T/n+sigma*random.gauss(0, 1)*np.sqrt(T/n)))
    # plt.plot(L)
    return L


def longstaff_schwartz_pricer(S, n, sigma, r, T, K, scenarios, call=True):
    """
    The longstaff_schwartz_pricer function computes the price of an American option using the Longstaff-Schwartz algorithm.

    Parameters
    ----------
        S
            Define the underlying price process
        n
            Define the number of time steps
        sigma
            The volatility of the stock price
        r
            Rate
        T
            Maturity
        K
            Strike
        scenarios
            Number the random paths
        call=True
            Indicate whether the option is a call or put option

    Returns
    -------

        The price of an american option using the longstaff-schwartz algorithm

    """
    X = []
    for i in range(scenarios):
        X.append(bs_mvt(S, n, sigma, r, T))
    X = np.array(X).reshape(n, scenarios)

    df = np.exp(-r*T/n)
    if (call == True):
        payoff = np.maximum(-K + X, 0)
    else:
        payoff = np.maximum(K - X, 0)

    # values matrix
    matrix = np.zeros_like(payoff)
    matrix[:, -1] = payoff[:, -1]

    # loop
    for t in range(n-2, 0, -1):

        # paths where the value is positive
        chemins_choisis = payoff[:, t] > 0

        # linear regression
        regressor = np.polyfit(X[chemins_choisis, t],
                               matrix[chemins_choisis, t+1] * df, 2)
        result = np.polyval(regressor, X[chemins_choisis, t])

        # init
        exercise = np.zeros(len(chemins_choisis), dtype=bool)

        # exercise path
        exercise[chemins_choisis] = payoff[chemins_choisis, t] > result

        matrix[exercise, t] = payoff[exercise, t]
        matrix[exercise, t+1:] = 0

        # non-exercise paths
        chemin_non_exercice = (matrix[:, t] == 0)

        # fill in the value matrix
        matrix[chemin_non_exercice, t] = matrix[chemin_non_exercice, t+1] * df

    price = np.mean(matrix[:, 1]) * df
    return price
