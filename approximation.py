import numpy as np
import matplotlib.pyplot as plt
from interpolation import accurate_model

N = 1000
I0 = 1
r_beta = 1

M = 1000


def get_approx_func(bound):
    def approx_func(x):
        return 1000 / (1 + 999 * np.power((1 + -x / M), M))

    return approx_func


if __name__ == "__main__":

    bound = input("Please input the error bound:")
    approx_func = get_approx_func(float(bound))

    x = np.arange(0, 10, 0.1)
    I_approx = approx_func(x)
    I = accurate_model(x)

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x, I)
    ax2.plot(x, I_approx, c='r')
    plt.show()