from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import statistics
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import csv

def quadrantr(x, y):
    size = x.size
    med_x = np.median(x)
    med_y = np.median(y)
    new_x = np.empty(size, dtype=float)
    new_x.fill(med_x)
    new_x = x - new_x
    new_y = np.empty(size, dtype=float )
    new_y.fill(med_y)
    new_y = y - new_y
    n1 = int(0)
    n2 = int(0)
    n3 = int(0)
    n4 = int(0)
    for k in range(x.size):
        if new_x[k] >= 0 and new_y[k] >= 0:
            n1 += 1
        elif new_x[k] < 0 and new_y[k] > 0:
            n2 += 1
        elif new_x[k] < 0 and new_y[k] < 0:
            n3 += 1
        elif new_x[k] > 0 and new_y[k] < 0:
            n4 += 1
    return ((n1 + n3) - (n2 + n4)) / new_x.size


def create_table(table, pearson_coef, spearman_coef, quadrant_coef, po):
    n = 4
    template = '{:.' + str(n) + 'f}'
    table.writerow(['po = ' + str(po), 'r', 'r_S', 'r_Q'])
    p = np.median(pearson_coef)
    s = np.median(spearman_coef)
    q = np.median(quadrant_coef)
    table.writerow(['E(z)', template.format(p), template.format(s), template.format(q)])
    p = np.median([pearson_coef[i] ** 2 for i in range(1000)])
    s = np.median([spearman_coef[i] ** 2 for i in range(1000)])
    q = np.median([quadrant_coef[i] ** 2 for i in range(1000)])
    table.writerow(['E(z**2)', template.format(p), template.format(s), template.format(q)])
    p = statistics.variance(pearson_coef)
    s = statistics.variance(spearman_coef)
    q = statistics.variance(quadrant_coef)
    table.writerow(['D(z)', template.format(p), template.format(s), template.format(q)])
    table.writerow(["", "", "", ""])


def research_сoef(po, size, table):
    rv_mean = [0, 0]
    rv_cov = [[1.0, po], [po, 1.0]]
    pearson_coef = np.empty(1000, dtype=float)
    spearman_coef = np.empty(1000, dtype=float)
    quadrant_coef = np.empty(1000, dtype=float)
    for k in range(1000):
        rv = multivariate_normal.rvs(rv_mean, rv_cov, size=size)
        x = rv[:, 0]
        y = rv[:, 1]
        pearson_coef[k], t = pearsonr(x, y)
        spearman_coef[k], t = spearmanr(x, y)
        quadrant_coef[k] = quadrantr(x, y)
    create_table(table, pearson_coef, spearman_coef, quadrant_coef, po)


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x*1.5 , height=ell_radius_y *1.5,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def draw_plot_scatter(n):
    po = [0, 0.5, 0.9]
    rv_mean = [0, 0]
    fig, ax = plt.subplots(1, 3)
    fig.suptitle('n = ' + str(n), fontsize="x-large")
    titles = [r'$ \rho = 0$', r'$\rho = 0.5 $', r'$ \rho = 0.9$']
    for i in range(3):
        rv_cov = [[1.0, po[i]], [po[i], 1.0]]
        rv = multivariate_normal.rvs(rv_mean, rv_cov, size=n)
        x = rv[:, 0]
        y = rv[:, 1]
        ax[i].scatter(x, y, s=3)
        confidence_ellipse(x, y, ax[i], edgecolor='red')
        ax[i].set_title(titles[i])
        ax[i].scatter(np.mean(x), np.mean(y), c='red', s=3)
    plt.show()


def research1():
    po = [0, 0.5, 0.9]
    size = [20, 60, 100]
    for j in range(3):
        with open('2D' + str(size[j]) + '.csv', mode='w') as file:
            table = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(3):
                research_сoef(po[i], size[j], table)
        file.close()

def research2():
    size = [20, 60, 100]
    for i in range(3):
        draw_plot_scatter(size[i])
    # with open('2D' + str(size[j]) + '.csv', mode='w') as file:
    #     table = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     for i in range(3):
    #         research_сoef(po[i], size[j], table)
    # file.close()


def research_comb():
    size = [20, 60, 100]
    weights = np.array([0.9, 0.1])
    with open('comb.csv', mode='w') as file:
        table = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for j in range(3):
            pearson_coef = np.empty(1000, dtype=float)
            spearman_coef = np.empty(1000, dtype=float)
            quadrant_coef = np.empty(1000, dtype=float)
            for k in range(1000):
                rv = []
                for i in range(2):
                    x = 0.9 * multivariate_normal.rvs([0, 0], [[1, 0.9], [0.9, 1]], size=size[j])+ 0.1* multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], size=size[j])
                    rv += list(x)
                rv = np.array(rv)
                x = rv[:, 0]
                y = rv[:, 1]
                pearson_coef[k], t = pearsonr(x, y)
                spearman_coef[k], t = spearmanr(x, y)
                quadrant_coef[k] = quadrantr(x, y)
            create_table(table, pearson_coef, spearman_coef, quadrant_coef, size[j])
    file.close()


def main():
    # research1()
    research2()
    # research_comb()


if __name__ == "__main__":
    main()