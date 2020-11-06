import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 1000
I0 = 1
r_beta = 1


def load_table():
    df = pd.read_csv('LUT.csv')
    t = df['t'].values
    I = df['I'].values
    return t, I


def accurate_model(t):
    # calculate real I(t)
    It = N * I0 / (I0 + (N - I0) * np.exp(-t))
    return It


def get_hermite_func(t, I):
    # input: t, I -- given t_k and I(t_k)
    # output: heimit function

    h = t[1] - t[0]  # node interval

    def hermite_func(x):

        k = np.floor((x - t[0]) / h).astype(int)
        x_k = t[k]
        x_k1 = t[k + 1]
        f_k = I[k]
        f_k1 = I[k + 1]
        f_k_d = f_k * (N - f_k) / N
        f_k1_d = f_k1 * (N - f_k1) / N

        I_1 = np.power(x - x_k1, 2) * (2 * x - 3 * x_k + x_k1) / np.power(
            x_k1 - x_k, 3) * f_k
        I_2 = np.power(x - x_k, 2) * (2 * x - 3 * x_k1 + x_k) / np.power(
            x_k - x_k1, 3) * f_k1
        I_3 = np.power(x - x_k1, 2) * (x - x_k) / np.power(x_k - x_k1,
                                                           2) * f_k_d
        I_4 = np.power(x - x_k, 2) * (x - x_k1) / np.power(x_k - x_k1,
                                                           2) * f_k1_d

        Ix = I_1 + I_2 + I_3 + I_4
        return Ix

    return hermite_func


def show_res(x, I, I_hat):
    fig, ax = plt.subplots()
    ax.plot(x, I, label='accurate')
    ax.plot(x, I_hat, label='interpolar')
    ax.set_xlabel('time')
    ax.set_ylabel('I')
    ax.set_title('interpolar vs accurate')
    ax.legend()
    plt.show()


if __name__ == "__main__":

    tk, I_tk = load_table()  # t & I values with interval 0.5
    t = np.arange(0, 15, 0.1)  # t values with interval 0.1

    # calculate each accurate I(t) of interval 0.1 with SI model analytic solution
    I = accurate_model(t)

    # closure with parameters which is estimated via hermit interpolation
    hermite_func = get_hermite_func(tk, I_tk)

    # calculate estimated I(t) of interval 0.1 with hermite function
    I_hat = hermite_func(t)

    show_res(t, I, I_hat)

    # print maximum error
    print(np.abs(I_hat - I).max())