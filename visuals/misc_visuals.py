import matplotlib.pyplot as plt
import numpy as np


def xy_ax_indicator(ax):
    """
    draw xy axis indicator
    """
    with plt.style.context("dark_background"):

        ax.plot(
            np.array([-1, 1]), np.array([0, 0]), c="white", zorder=1, alpha=0.9, lw=1
        )
        ax.plot(
            np.array([0, 0]), np.array([-1, 1]), c="white", zorder=1, alpha=0.9, lw=1
        )
        ax.plot(
            np.array([-1, 1]),
            np.array([-0.50, 0.50]),
            c="white",
            zorder=1,
            alpha=0.9,
            lw=1,
        )

        ax.plot(
            np.array([-1, 0]),
            np.array([-0.25, 0.25]),
            c="white",
            alpha=0.4,
            ls="--",
            lw=1,
        )
        ax.plot(
            np.array([0, 1]),
            np.array([-0.25, 0.25]),
            c="white",
            alpha=0.4,
            ls="--",
            lw=1,
        )
        ax.plot(
            np.array([-1.0, 0]),
            np.array([-0.25, -0.25]),
            c="white",
            alpha=0.4,
            ls="--",
            lw=1,
        )
        ax.plot(
            np.array([0, 1.0]),
            np.array([0.25, 0.25]),
            c="white",
            alpha=0.4,
            ls="--",
            lw=1,
        )

        ax.fill([0, 1, 0, -1], [0.25, 0.25, -0.25, -0.25], color="grey", alpha=0.4)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axis("off")
        ax.grid(True, ls="--", c="grey")


def xz_ax_indicator(ax):
    """
    draw xz axis indicator
    """
    with plt.style.context("dark_background"):

        ax.plot(
            np.array([-1, 1]), np.array([0, 0]), c="white", zorder=1, alpha=0.9, lw=1
        )
        ax.plot(
            np.array([0, 0]), np.array([-1, 1]), c="white", zorder=1, alpha=0.9, lw=1
        )
        ax.plot(
            np.array([-1, 1]),
            np.array([-0.50, 0.50]),
            c="white",
            zorder=1,
            alpha=0.9,
            lw=1,
        )
        ax.plot(
            np.array([-0.5, 0.5]),
            np.array([0.5, 1]),
            c="white",
            alpha=0.4,
            ls="--",
            lw=1,
        )
        ax.plot(
            np.array([-0.5, 0.5]),
            np.array([-1, -0.5]),
            c="white",
            alpha=0.4,
            ls="--",
            lw=1,
        )
        ax.plot(
            np.array([-0.5, -0.5]),
            np.array([-1, 0.5]),
            c="white",
            alpha=0.4,
            ls="--",
            lw=1,
        )
        ax.plot(
            np.array([0.5, 0.5]),
            np.array([-0.5, 1]),
            c="white",
            alpha=0.4,
            ls="--",
            lw=1,
        )
        ax.fill([-0.5, 0.5, 0.5, -0.5], [0.5, 1, -0.5, -1], color="grey", alpha=0.4)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axis("off")


def yz_ax_indicator(ax):
    """
    draw yz axis indicator
    """
    with plt.style.context("dark_background"):

        ax.plot(
            np.array([-1, 1]), np.array([0, 0]), c="white", zorder=1, alpha=0.9, lw=1
        )
        ax.plot(
            np.array([0, 0]), np.array([-1, 1]), c="white", zorder=1, alpha=0.9, lw=1
        )
        ax.plot(
            np.array([-1, 1]),
            np.array([-0.50, 0.50]),
            c="white",
            zorder=1,
            alpha=0.9,
            lw=1,
        )
        ax.plot(
            np.array([-0.75, 0.75]),
            np.array([0.75, 0.75]),
            c="white",
            alpha=0.4,
            ls="--",
            lw=1,
        )
        ax.plot(
            np.array([-0.75, 0.75]),
            np.array([-0.75, -0.75]),
            c="white",
            alpha=0.4,
            ls="--",
            lw=1,
        )
        ax.plot(
            np.array([-0.75, -0.75]),
            np.array([-0.75, 0.75]),
            c="white",
            alpha=0.4,
            ls="--",
            lw=1,
        )
        ax.plot(
            np.array([0.75, 0.75]),
            np.array([-0.75, 0.75]),
            c="white",
            alpha=0.4,
            ls="--",
            lw=1,
        )
        ax.fill(
            [-0.75, 0.75, 0.75, -0.75],
            [0.75, 0.75, -0.75, -0.75],
            color="grey",
            alpha=0.4,
        )
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axis("off")
