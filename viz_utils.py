import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D         # don't delete this, used for 3d projection
plt.style.use("seaborn-colorblind")


def plot_losses(losses):

    for key in sorted(losses.keys()):

        value = losses[key]

        label = "{} loss".format(key)

        plt.plot(value, label=label)

    plt.legend()
    plt.show()


def plot_latent_space(zs, real, z_size):

    if z_size == 1:
        plt.scatter(zs[:, 0], np.zeros_like(zs[:, 0]), c=real)
        plt.colorbar()
    elif z_size == 2:
        plt.scatter(zs[:, 0], zs[:, 1], c=real)
        plt.colorbar()
    elif z_size == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(zs[:, 0], zs[:, 1], zs[:, 2], c=real)
        fig.colorbar(sc)
    else:
        raise ValueError("Cannot plot 4D or higher.")

    plt.show()
