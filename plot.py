import matplotlib.pyplot as plt
import numpy as np


def plot(f1, f2, lower, upper, levels):
    X, Y = np.meshgrid(
        np.linspace(lower, upper, 256), np.linspace(lower, upper, 256)
    )
    Z1 = np.vectorize(f1)(X, Y)
    Z2 = np.vectorize(f2)(X, Y)
    levels = np.linspace(Z1.min(), Z1.max(), levels)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.contourf(X, Y, Z1, levels=levels, cmap="plasma")
    ax2.contourf(X, Y, Z2, levels=levels, cmap="plasma")
    plt.show()
