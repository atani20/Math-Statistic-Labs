import random
import statistics
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.stats import cauchy
from scipy.stats import laplace
from scipy.stats import poisson
from scipy.stats import uniform
import matplotlib.pyplot as plt
import matplotlib

def percentOfOutliers(data):
    LQ = np.quantile(data, 0.25)
    UQ = np.quantile(data, 0.75)
    IQR = UQ - LQ
    xl = LQ - 1.5 * IQR
    xn = UQ + 1.5 * IQR
    count = np.count_nonzero(data > xn)
    count += np.count_nonzero(data < xl)
    return count / data.size

def createBoxplot(data20, data100, title):
    fig1, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.boxplot([data20, data100], vert=False, widths=0.7)
    ax1.set_yticklabels([20, 100])
    ax1.set_xlabel('x')
    ax1.set_ylabel('n')
    plt.savefig(title + '.png')

func20_dict = {
 'Normal': norm.rvs(size = 20),
 'Cauchy': cauchy.rvs(size=20),
 'Laplace': laplace.rvs(0, 1 / 2 ** 0.5, size = 20),
 'Poisson': poisson.rvs(10, size=20),
 'Uniform': uniform.rvs(-float(3 ** 0.5), 2*float(3 ** 0.5), size = 20),
}
func100_dict = {
 'Normal': norm.rvs(size = 100),
 'Cauchy': cauchy.rvs(size=100),
 'Laplace': laplace.rvs(0, 1 / 2 ** 0.5, size = 100),
 'Poisson': poisson.rvs(10, size=100),
 'Uniform': uniform.rvs(-float(3 ** 0.5), 2*float(3 ** 0.5), size = 100),
}

for elements in func20_dict:
    createBoxplot(func20_dict[elements], func100_dict[elements], elements)


percent20 = np.empty(1000)
percent100 = np.empty(1000)
for i in range(1000):
    data = func20_dict[elements]
    percent20[i] = percentOfOutliers(poisson.rvs(10, size=20))
    percent100[i] = percentOfOutliers(poisson.rvs(10, size=100))
print("Uniform")
print('20  = ' + str(np.mean(percent20)))
print('100 = ' + str(np.mean(percent100)))

