import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize


def lsm_linreg(x, y):
    betta1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    betta0 = np.mean(y) - betta1 * np.mean(x)
    return betta0, betta1


def cost_function(params, x, y):
    a, b = params
    res = float(0)
    for i in range(x.size):
        res += abs(a * x[i] + b - y[i])
    return res


def research(x, y, title):
    print(title)
    a, b = lsm_linreg(x, y)
    print("МНК:")
    print('betta0 = ' + str(a))
    print('betta1 = ' + str(b))
    result = minimize(cost_function, [a, b], args=(x, y), method='Nelder-Mead')
    if result.success:
        fitted_params = result.x
        print(fitted_params)
    else:
        return
    a1, b1 = fitted_params[0], fitted_params[1]
    plt.scatter(x[1:-2], y[1:-2],  label='Выборка', edgecolors="black", color='white')
    plt.plot(x, 2 * np.ones(20) + x * (2 * np.ones(20)), label='Модель', color='red')
    plt.plot(x, a * np.ones(20) + x * (b * np.ones(20)), label='МНК', color='green')
    plt.plot(x, a1 * np.ones(20) + x * (b1 * np.ones(20)), label='МНМ', color='blue')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(title + '.png')
    plt.show()

def main():
    x = np.arange(-1.8, 2.2, 0.2)
    y = 2 * x + 2 * np.ones(20) + np.random.normal(size=x.size)
    research(x, y, 'Без возмущений')
    y[0] = 10
    y[-1] = -10
    research(x, y, 'С возмущениями')


if __name__ == "__main__":
    main()