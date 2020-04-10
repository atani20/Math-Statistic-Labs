import numpy as np
import scipy.stats
import statistics
from scipy.stats import norm
from scipy.stats import chi2
import csv

def F(x, data):
    res = data[data < x]

def main(table):
    k = 4
    template = '{:.' + str(k) + 'f}'
    size = 100
    data = norm.rvs(size=size)
    mu = np.mean(data)
    sigma = statistics.variance(data)
    print(mu, sigma)
    p = 0.95
    df = 6
    value = chi2.ppf(p, df - 1)
    delta = 0.66
    a0 = -1.05
    n = data[data <= a0].size
    p = norm.cdf(a0)
    table.writerow(['1', 'до -1', template.format(n), template.format(p), template.format(size * p), template.format(n - size*p), template.format((n - size*p)**2/(size*p))])
    for i in range(2, df):
        a1 = a0 + delta
        n = data[(data <= a1) & (data > a0)].size
        p = norm.cdf(a1) - norm.cdf(a0)
        table.writerow([template.format(i), template.format(a0) + '  ' + template.format(a1), template.format(n), template.format(p), template.format(size * p), template.format(n - size * p), template.format((n - size * p) ** 2 / (size * p))])
        a0 = a1
    n = data[data > a0].size
    p = 1 - norm.cdf(a0)
    table.writerow(
        [template.format(df), template.format( a0), template.format(n), template.format(p), template.format(size * p), template.format(n - size * p), template.format((n - size * p) ** 2 / (size * p))])


if __name__ == '__main__':
    with open('chi2.csv', mode='w') as file:
        table = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        table.writerow(['i', 'Границы', '$n_i$', '$p_i$', ' $np_i$', '$n_i - np_i$', ' $\frac{(n_i -np_i)^2}{np_i}$'])
        main(table)
