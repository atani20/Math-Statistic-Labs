import numpy as np
import scipy.stats
import statistics
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import chi2
import matplotlib.pyplot as plt
import csv


def main(table, data, size):
    k = 4
    template = '{:.' + str(k) + 'f}'
    # mu = np.mean(data)
    # sigma = statistics.variance(data)
    # print(mu, sigma)
    p = 0.95
    df = 4
    value = chi2.ppf(p, df - 1)
    print(value)
    delta = 1
    a0 = -1.5
    n = data[data <= a0].size
    # p = norm.cdf(a0)
    p = norm.cdf(a0)
    chi2_sum = 0
    table.writerow(['1', str('до' + str(a0)), n, template.format(p), template.format(size * p), template.format(n - size*p), template.format((n - size*p)**2/(size*p))])
    for i in range(2, df):
        a1 = a0 + delta
        n = data[(data <= a1) & (data > a0)].size
        p = norm.cdf(a1) - norm.cdf(a0)
        c = (n - size * p) ** 2 / (size * p)
        chi2_sum += c
        table.writerow([i, template.format(a0) + '  ' + template.format(a1), n, template.format(p), template.format(size * p), template.format(n - size * p), template.format(c)])
        a0 = a1
    n = data[data > a0].size
    p = 1 - norm.cdf(a0)
    c = (n - size * p) ** 2 / (size * p)
    chi2_sum += c
    table.writerow(
        [df, template.format( a0), n, template.format(p), template.format(size * p), template.format(n - size * p), template.format(c)])
    table.writerow(
        ['sum', '-', size, template.format(1), template.format(100), template.format(100),
         template.format(chi2_sum)])


if __name__ == '__main__':
    size = [20, 50]
    a = float(3/2)
    for i in range(len(size)):
        with open('chi2unif' + str(size[i]) + '.csv' , mode='w') as file:
            table = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            table.writerow(['i', 'Границы', '$n_i$', '$p_i$', ' $np_i$', '$n_i - np_i$', ' $ \ frac{(n_i -np_i)^2}{np_i}$'])
            data = uniform.rvs(-a, 2 * a, size = size[i])
            main(table, data, size[i])
    # x_axis = np.arange(-4, 4, 0.01)
    # plt.plot(x_axis,  norm.pdf(x_axis, 0,1 ))
    # # a = float(3 ** 0.5)
    # y_axis = uniform.pdf(x_axis, -a, 2*a)
    # plt.plot(x_axis, y_axis)
    # plt.show()
