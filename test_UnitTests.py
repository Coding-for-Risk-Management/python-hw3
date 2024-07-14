import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import python_hw3


def test_get_returns():

    # Unit Tests for get_returns
    x = np.array([100, 120, 150, 200])
    rets1 = python_hw3.get_returns(x, 1)
    rets2 = python_hw3.get_returns(x, 2)
    rets3 = python_hw3.get_returns(x, 3)

    assert np.round(rets1[0], 2) == 0.20
    assert np.round(rets1[1], 2) == 0.25
    assert np.round(rets1[2], 2) == 0.33
    assert np.round(rets2[0], 2) == 0.50
    assert np.round(rets3[0], 2) == 1.00


# Function to calculate Percent VaR
def test_percent_var():
    # Unit test
    r = np.random.normal(0.05, 0.03, 1000000)
    # Probability under normal curve within 2 standard deviations
    probability2SD = norm.cdf(2)

    myalpha = probability2SD
    my_percent_var = python_hw3.percent_var(r, myalpha)

    assert np.round(my_percent_var, 2) == 0.01


def test_es():
    # Unit test
    u = np.random.uniform(0, 100, 100000)
    assert np.round(python_hw3.es(losses=u, alpha=0.8), 0) == 60
    assert np.round(python_hw3.es(losses=u, VaR=80), 0) == 90
