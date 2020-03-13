import random
import statistics
import numpy as np
import statsmodels.api as sm
from statsmodels.distributions.mixture_rvs import mixture_rvs
import scipy.stats as stat
from scipy.stats import norm
from scipy.stats import cauchy
from scipy.stats import laplace
from scipy.stats import poisson
from scipy.stats import uniform
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

def drawPlot(xEcdf, yEcdf, x_axis, y_axis, sector, title, size):
    plt.subplot(130 + sector)
    plt.title(title + '  n = ' + str(size))
    plt.plot(x_axis, y_axis, label = 'CDF')
    plt.step(xEcdf, yEcdf, label = 'ECDF')
    plt.legend()
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('F(x)')



def drawPlotKde(data, title, x_axis, y_axis, size):
    kde = stat.gaussian_kde(data, bw_method='silverman')
    h_n = kde.factor
    fig, ax = plt.subplots(1, 3)
    fig.suptitle(title + ' n = ' + str(size), fontsize="x-large")
    coef = [0.5, 1, 2]
    titles = [r'$h = \frac{h_n}{2}$', r'$h = h_n$', r'$h = 2 * h_n$']
    for i in range(3):
        ax[i].plot(x_axis, y_axis, label='Pdf')
        sns.kdeplot(data, ax=ax[i], bw=h_n * coef[i], label='Kde')
        ax[i].set_title(titles[i])
        ax[i].grid()
        ax[i].legend()



size = [20, 60, 100]
x_axis = np.arange(-4, 4, 0.01)
# # 1 нормальное распределение N(x, 0, 1)
# y_axis = norm.pdf(x_axis, 0,1 )
# for i in range(0, 3):
#     data = norm.rvs(size = size[i])
#     data = data[(data >= -4) & (data <= 4)]
#     drawPlotKde(data, 'Normal', x_axis, y_axis, size[i])
#     plt.savefig('Normal' + str(size[i]) + '.png')
#     plt.show()
#     # ecdf = sm.distributions.ECDF(data)
#     # xEcdf = np.linspace(min(data), max(data))
#     # yEcdf = ecdf(xEcdf)
#     #drawPlot(xEcdf, yEcdf, x_axis, y_axis, i + 1, "Normal", size[i])
#
# #2 распределение Коши C(x, 0, 1)
# y_axis = cauchy.pdf(x_axis, 0, 1)
# for i in range(0, 3):
#     data = cauchy.rvs(size = size[i])
#     data = data[(data >= -4) & (data <= 4)]
#     drawPlotKde(data, 'Cauchy', x_axis, y_axis, size[i])
#     plt.savefig('Cauchy' + str(size[i]) + '.png')
#     plt.show()
#     # ecdf = sm.distributions.ECDF(data)
#     # xEcdf = np.linspace(min(data), max(data))
#     # yEcdf = ecdf(xEcdf)
#     # drawPlot(xEcdf, yEcdf, x_axis, y_axis, i + 1, "Cauchy", size[i])
#
# #3  распределение Лапласа L(x, 0, 1/sqrt(2))
# y_axis = laplace.pdf(x_axis, 0, 1 / 2 ** 0.5)
# for i in range(0, 3):
#     data = laplace.rvs(0, 1 / 2 ** 0.5, size = size[i])
#     data = data[(data >= -4) & (data <= 4)]
#     drawPlotKde(data, 'Laplace', x_axis, y_axis, size[i])
#     plt.savefig('Laplace' + str(size[i]) + '.png')
#     plt.show()
#     # ecdf = sm.distributions.ECDF(data)
#     # xEcdf = np.linspace(min(data), max(data))
#     # yEcdf = ecdf(xEcdf)
#     # drawPlot(xEcdf, yEcdf, x_axis, y_axis, i + 1, "Laplace", size[i])
# # plt.savefig('Laplace.png')
# # plt.show()
#
# #5 Равномерное распределение U(x, -sqrt(3), sqrt(3))
# a = float(3 ** 0.5)
# y_axis = uniform.pdf(x_axis, -a, 2*a)
# for i in range(0, 3):
#     data = uniform.rvs(-a, 2*a, size = size[i])
#     data = data[(data >= -4) & (data <= 4)]
#     drawPlotKde(data, 'Uniform', x_axis, y_axis, size[i])
#     plt.savefig('Uniform' + str(size[i]) + '.png')
#     plt.show()
#     # ecdf = sm.distributions.ECDF(data)
#     # xEcdf = np.linspace(min(data), max(data))
#     # yEcdf = ecdf(xEcdf)
#     # drawPlot(xEcdf, yEcdf, x_axis, y_axis, i + 1, "Uniform", size[i])
# # plt.savefig('Uniform.png')
# # plt.show()

x_axis = np.arange(0, 20, 1)
#4 Респределение Пуассона P(k, 10)
y_axis = poisson.pmf(x_axis, 10)
for i in range(0, 3):
    data = poisson.rvs(10, size = size[i])#[np.random.standard_cauchy() for count in range(0,  size[i])]
    data = data[(data >= 6) & (data <= 14)]
    drawPlotKde(data, 'Poisson', x_axis, y_axis, size[i])
    plt.savefig('Poisson' + str(size[i]) + '.png')
    plt.show()
    # ecdf = sm.distributions.ECDF(data)
    # xEcdf = np.linspace(min(data), max(data))
    # yEcdf = ecdf(xEcdf)
    # drawPlot(xEcdf, yEcdf, x_axis, poisson.cdf(x_axis, 10), i + 1, "Poisson", size[i])
plt.savefig('Poisson.png')
plt.show()

