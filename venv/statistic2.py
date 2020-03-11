import random
import numpy as np
import statistics
import scipy.stats as stat
from scipy.stats import norm
from scipy.stats import cauchy
from scipy.stats import laplace
from scipy.stats import poisson
from scipy.stats import uniform
import csv


def calculationValue():
    with open('uniform.csv', mode='w') as file:
        table = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        table.writerow(['', 'Sample mean', 'Median', 'zR', 'zQ', 'zTr'])
        np.set_printoptions(precision=2)
        size = [10, 100, 1000]
        for i in range(0, 3):
            sampleMean = np.empty(1000)
            median = np.empty(1000)
            zR = np.empty(1000)
            zQ = np.empty(1000)
            zTr = np.empty(1000)
            a = float(3 ** 0.5)
            for j  in range(1000):
                data = uniform.rvs(-a, 2*a, size = size[i])
                data.sort()
                sampleMean[j] = np.mean(data)
                median[j] = np.median(data)
                zR[j] = (data[0] + data[size[i] - 1]) / 2
                zQ[j] = (np.quantile(data, 0.25) + np.quantile(data, 0.75)) / 2
                zTr[j] = stat.trim_mean(data, 0.25)
            print("size = " + str(size[i]))
            n = 4
            template = '{:.' + str(n) + 'f}'
            table.writerow(["uniform, n = " + str(size[i]), "", "",  "", "", ""])
            table.writerow(["E", template.format(np.mean(sampleMean)), template.format(np.mean(median)), template.format(np.mean(zR)), template.format(np.mean(zQ)), template.format(np.mean(zTr))])
            table.writerow(["D", template.format(statistics.variance(sampleMean)), template.format(statistics.variance(median)),                                                                                        template.format(statistics.variance(zR)), template.format(statistics.variance(zQ)), template.format(statistics.variance(zTr))])
            table.writerow(["", "", "", "", "", ""])


calculationValue()

