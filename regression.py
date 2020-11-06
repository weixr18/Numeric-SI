import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data(data_path):
    df = pd.read_csv(data_path)
    t = df['t'].values
    I = df['I'].values
    return t, I


def MSE(x, y):
    # input: x, y
    # output: a, b

    if (len(x.shape) != 1 or len(y.shape) != 1):
        print("Error: MSE wrong input format.")
        return None
    elif (x.shape[0] != y.shape[0]):
        print("Error: MSE input mismatch.")
        return None

    n = x.shape[0]
    S_x = np.sum(x)
    S_y = np.sum(y)
    S_xx = np.dot(x, x)
    S_xy = np.dot(x, y)

    a = (n * S_xy - S_x * S_y) / (n * S_xx - S_x * S_x)
    b = (S_xx * S_y - S_x * S_xy) / (n * S_xx - S_x * S_x)
    return a, b


def solve(t, I, r_beta):
    # input: t, I
    # output: N, I0

    x = np.exp(-t)
    y = 1 / I
    a, b = MSE(x, y)
    N = 1 / b
    I0 = 1 / (a + b)
    return N, I0


if __name__ == "__main__":
    # known parameters
    r_beta = 1

    # load data
    t, I = load_data('data.csv')
    # solve for unknown parameters
    N, I0 = solve(t, I, r_beta)
    print("N =", int(N), ", I0 =", int(I0))