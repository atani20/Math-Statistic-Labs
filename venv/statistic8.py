import numpy as np
import scipy.stats
import statistics
from scipy.stats import t
from scipy.stats import norm
from scipy.stats import chi2
from scipy.stats import moment
import csv


def mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    n = data.size
    m, se = np.mean(data), statistics.variance(data)
    h = se * t.ppf((1 + confidence) / 2., n - 1) / (n - 1)**0.5
    return 'mean', round(m - h, 3), round(m + h, 3)

def variance_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    n = data.size
    m, se = np.mean(data), statistics.variance(data)
    h = se * t._ppf((1 + confidence) / 2., n - 1)
    a = se * (n / (chi2.ppf((1 + confidence) / 2, n - 1)))**0.5
    b = se * n **0.5 / (chi2.ppf(( 1 - confidence )/ 2, n - 1))**0.5
    return 'variance', round(a, 3), round(b,  3)

def mean_confidence_interval_asimptotic(data, confidence=0.95):
    data = np.array(data)
    n = data.size
    m, se = np.mean(data), statistics.variance(data)
    u = norm.ppf((1 + confidence) / 2)
    h = se * u / n ** 0.5
    return 'mean asimptotic', round(m - h, 3), round(m + h, 3)


def variance_confidence_interval_asimptotic(data, confidence=0.95):
    data = np.array(data)
    n = data.size
    m, se = np.mean(data), statistics.variance(data)
    m4 = moment(data, 4)
    e = m4 / se**4 - 3
    u = norm.ppf((1 + confidence) / 2)
    U = u * ((e+ 2) / n)**0.5
    a = se * (1 + U) ** (-0.5)
    b = se * (1 - U) ** (-0.5)
    return 'variance asimptotic', round(a, 3), round(b,  3)

def main():
    size = [20, 100]
    for i in range(len(size)):
        print(size[i])
        data = norm.rvs(size=size[i])
        print(mean_confidence_interval(data))
        print(variance_confidence_interval(data))
        print(mean_confidence_interval_asimptotic(data))
        print(variance_confidence_interval_asimptotic(data))

if __name__ == '__main__':
    main()