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


def drawPlot(data, bin, x_data, y_data, title):
    plt.title(title)
    plt.plot(x_data, y_data, label = 'Плотность вероятности')
    plt.hist(data,bin, density = True, label = 'Гистограмма частот')
    plt.legend()
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('Плотность')
    plt.savefig(title + '.png')
    plt.show()

size = [10, 50, 1000]
bin = [5, 10, 25]

# 1 нормальное распределение N(x, 0, 1)
for i in range(0, 3):
    data = norm.rvs(size = size[i])
    #data = [random.normalvariate(0, 1) for count in range(0,  size[i])]
    x_axis = np.arange(-3, 3, 0.01)
    drawPlot(data, bin[i], x_axis, norm.pdf(x_axis, 0,1 ), "Нормальное распределение. Выборка из " + str(size[i]) + " элементов")

#2 распределение Коши C(x, 0, 1)
for i in range(0, 3):
    #data = [np.random.standard_cauchy() for count in range(0,  size[i])]
    data = cauchy.rvs(size = size[i])
    data = data[(data >= -5) & (data <= 5)] #нужно ли здесь урезать?
    x_axis = np.arange(-5, 5, 0.01)
    drawPlot(data, bin[i], x_axis, cauchy.pdf(x_axis), "Распределение Коши. Выборка из " + str(size[i]) + " элементов")

#3  распределение Лапласа L(x, 0, 1/sqrt(2))
for i in range(0, 3):
    data = laplace.rvs(0, 1 / 2 ** 0.5, size = size[i])#[np.random.standard_cauchy() for count in range(0,  size[i])]
    x_axis = np.arange(-5, 5, 0.01)
    drawPlot(data, bin[i], x_axis, laplace.pdf(x_axis, 0, 1 / 2 ** 0.5), "Распределение Лапласа. Выборка из " + str(size[i]) + " элементов")

#4 Респределение Пуассона P(k, 10)
for i in range(0, 3):
    data = poisson.rvs(10, size = size[i])#[np.random.standard_cauchy() for count in range(0,  size[i])]
    x_axis = np.arange(0, 20, 1)
    drawPlot(data, bin[i], x_axis, poisson.pmf(x_axis, 10), "Распределение Пуассона. Выборка из " + str(size[i]) + " элементов")

#5 Равномерное распределение U(x, -sqrt(3), sqrt(3))
a = float(3 ** 0.5)
for i in range(0, 3):
    data = uniform.rvs(-a, 2*a, size = size[i])
    x_axis = [-a, a]
    drawPlot(data, bin[i], x_axis, uniform.pdf([-a, a], -a, 2*a), "Нормальное распределение. Выборка из " + str(size[i]) + " элементов")
