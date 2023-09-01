import numpy as np
from numpy.typing import NDArray as Arr
from numpy import float64 as f64
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.figure as mplf


_TENSOR_VECTOR_MAP: dict[int, tuple[int, int]] = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 0),
    3: (1, 1),
}

_LEGEND_KWARGS: dict[str, str | int | float | bool | None] = {
    "loc": "outside lower center",
    "ncol": 4,
    "handlelength": 1.0,
    "frameon": False,
}


def plot_scalar(
    *data: tuple[Arr[f64], Arr[f64]] | list[Arr[f64]],
    x_lim: list[float] | None = None,
    y_lim: list[float] | None = None,
    figsize: tuple[float, float] = (4, 3),
    dpi: int = 150,
    x_label: str | None = None,
    y_label: str | None = None,
    curve_labels: list[str] | None = None,
    colors: list[str] = ["k", "r", "b", "g", "c", "m"],
    lines: list[str] = ["-", "-", "-", "-", "-", "-"],
    marker: list[str] = ["none", "none", "none", "none", "none", "none"],
    markersize: int | float = 4,
    markerskip: int | list[int] | float | list[float] | None = None,
    markeredgewidth: float = 0.3,
    fillstyle: str = "full",
    fout: str | None = None,
    **kwargs,
) -> None:
    styles = {
        "markersize": markersize,
        "markevery": markerskip,
        "fillstyle": fillstyle,
        "markeredgewidth": markeredgewidth,
    }
    cyclers = (
        cycler("color", colors)
        + cycler("linestyle", lines)
        + cycler("marker", marker)
        + cycler("mec", colors)
    )
    fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=figsize, layout="constrained")
    ax.set_prop_cycle(cyclers)
    for x, y in data:
        ax.plot(x, y, **styles, **kwargs)
    if curve_labels:
        fig.legend(curve_labels, **_LEGEND_KWARGS)
    if x_label:
        ax.set_xlabel(x_label, fontsize=12)
    if y_label:
        ax.set_ylabel(y_label, fontsize=12)
    if x_lim:
        plt.setp(fig.axes, xlim=x_lim)
    if y_lim:
        plt.setp(fig.axes, ylim=y_lim)
    if fout:
        fig.savefig(fout)
    else:
        plt.show()
    plt.close(fig)


def plot_stress_vs_strain_1D(
    *data: tuple[Arr[f64], Arr[f64]] | list[Arr[f64]],
    x_lim: list[float] | None = None,
    y_lim: list[float] | None = None,
    figsize: tuple[float, float] = (4, 3),
    dpi: int = 150,
    x_label: str = r"$E$",
    y_label: str = r"$S$ (kPa)",
    curve_labels: list[str] | None = None,
    colors: list[str] = ["k", "r", "b", "g", "c", "m"],
    lines: list[str] = ["-", "-", "-", "-", "-", "-"],
    marker: list[str] = ["none", "none", "none", "none", "none", "none"],
    markersize: int | float = 4,
    markerskip: int | list[int] | float | list[float] | None = None,
    markeredgewidth: float = 0.3,
    fillstyle: str = "full",
    fout: str | None = None,
    **kwargs,
) -> None:
    styles = {
        "markersize": markersize,
        "markevery": markerskip,
        "fillstyle": fillstyle,
        "markeredgewidth": markeredgewidth,
    }
    cyclers = (
        cycler("color", colors)
        + cycler("linestyle", lines)
        + cycler("marker", marker)
        + cycler("mec", colors)
    )
    fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=figsize, layout="constrained")
    ax.set_prop_cycle(cyclers)
    for x, y in data:
        ax.plot(x[:, 0, 0], y[:, 0, 0], **styles, **kwargs)
    if curve_labels:
        fig.legend(curve_labels, **_LEGEND_KWARGS)
    if x_label:
        ax.set_xlabel(x_label, fontsize=12)
    if y_label:
        ax.set_ylabel(y_label, fontsize=12)
    if x_lim:
        plt.setp(fig.axes, xlim=x_lim)
    if y_lim:
        plt.setp(fig.axes, ylim=y_lim)
    if fout:
        fig.savefig(fout)
    else:
        plt.show()
    plt.close(fig)


def plot_stress_vs_strain_2D(
    *data: tuple[Arr[f64], Arr[f64]] | list[Arr[f64]],
    x_lim: list[float] | None = None,
    y_lim: list[float] | None = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 150,
    x_label: list[str] = [r"$E_{11}$", r"$E_{12}$", r"$E_{21}$", r"$E_{22}$"],
    y_label: list[str] = [
        r"$S_{11}$ (kPa)",
        r"$S_{12}$ (kPa)",
        r"$S_{21}$ (kPa)",
        r"$S_{22}$ (kPa)",
    ],
    curve_labels: list[str] | None = None,
    colors: list[str] = ["k", "r", "b", "g", "c", "m"],
    lines: list[str] = ["-", "-", "-", "-", "-", "-"],
    marker: list[str] = ["none", "none", "none", "none", "none", "none"],
    markersize: int | float = 4,
    markerskip: int | list[int] | float | list[float] | None = None,
    markeredgewidth: float = 0.3,
    fillstyle: str = "full",
    fout: str | None = None,
    **kwargs,
) -> None:
    styles = {
        "markersize": markersize,
        "markevery": markerskip,
        "fillstyle": fillstyle,
        "markeredgewidth": markeredgewidth,
    }
    cyclers = (
        cycler("color", colors)
        + cycler("linestyle", lines)
        + cycler("marker", marker)
        + cycler("mec", colors)
    )
    fig, axs = plt.subplots(2, 2, dpi=dpi, figsize=figsize, layout="constrained")

    for k in range(4):
        i, j = _TENSOR_VECTOR_MAP[k]
        axs[i, j].set_prop_cycle(cyclers)
        for x, y in data:
            axs[i, j].plot(x[:, i, j], y[:, i, j], **styles, **kwargs)
        if x_label:
            axs[i, j].set_xlabel(x_label[k], fontsize=12)
        if y_label:
            axs[i, j].set_ylabel(y_label[k], fontsize=12)
    if curve_labels:
        fig.legend(curve_labels, **_LEGEND_KWARGS)
    if x_lim:
        plt.setp(fig.axes, xlim=x_lim)
    if y_lim:
        plt.setp(fig.axes, ylim=y_lim)
    if fout:
        fig.savefig(fout)
    else:
        plt.show()
    plt.close(fig)


def plot_strain_vs_time_1D(
    time: Arr[f64],
    *data: Arr[f64],
    x_lim: list[float] | None = None,
    y_lim: list[float] | None = None,
    figsize: tuple[float, float] = (4, 3),
    dpi: int = 150,
    x_label: str = r"time (s)",
    y_label: str = r"$E$ (kPa)",
    curve_labels: list[str] | None = None,
    colors: list[str] = ["k", "r", "b", "g", "c", "m"],
    lines: list[str] = ["-", "-", "-", "-", "-", "-"],
    marker: list[str] = ["none", "none", "none", "none", "none", "none"],
    markersize: int | float = 4,
    markerskip: int | list[int] | float | list[float] | None = None,
    markeredgewidth: float = 0.3,
    fillstyle: str = "full",
    fout: str | None = None,
    **kwargs,
) -> None:
    styles = {
        "markersize": markersize,
        "markevery": markerskip,
        "fillstyle": fillstyle,
        "markeredgewidth": markeredgewidth,
    }
    cyclers = (
        cycler("color", colors)
        + cycler("linestyle", lines)
        + cycler("marker", marker)
        + cycler("mec", colors)
    )
    fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=figsize, layout="constrained")
    ax.set_prop_cycle(cyclers)
    for x, y in data:
        ax.plot(time, y[:, 0, 0], **styles, **kwargs)
    if x_label:
        ax.set_xlabel(x_label, fontsize=12)
    if y_label:
        ax.set_ylabel(y_label, fontsize=12)
    if curve_labels:
        fig.legend(curve_labels, **_LEGEND_KWARGS)
    if x_lim:
        plt.setp(fig.axes, xlim=x_lim)
    if y_lim:
        plt.setp(fig.axes, ylim=y_lim)
    if fout:
        fig.savefig(fout)
    else:
        plt.show()
    plt.close(fig)


def plot_strain_vs_time_2D(
    time: Arr[f64],
    *data: Arr[f64],
    x_lim: list[float] | None = None,
    y_lim: list[float] | None = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 150,
    x_label: str = r"time (s)",
    y_label: list[str] = [
        r"$E_{11}$ (kPa)",
        r"$E_{12}$ (kPa)",
        r"$E_{21}$ (kPa)",
        r"$E_{22}$ (kPa)",
    ],
    curve_labels: list[str] | None = None,
    colors: list[str] = ["k", "r", "b", "g", "c", "m"],
    lines: list[str] = ["-", "-", "-", "-", "-", "-"],
    marker: list[str] = ["none", "none", "none", "none", "none", "none"],
    markersize: int | float = 4,
    markerskip: int | list[int] | float | list[float] | None = None,
    markeredgewidth: float = 0.3,
    fillstyle: str = "full",
    fout: str | None = None,
    **kwargs,
) -> None:
    styles = {
        "markersize": markersize,
        "markevery": markerskip,
        "fillstyle": fillstyle,
        "markeredgewidth": markeredgewidth,
    }
    cyclers = (
        cycler("color", colors)
        + cycler("linestyle", lines)
        + cycler("marker", marker)
        + cycler("mec", colors)
    )
    fig, axs = plt.subplots(4, 1, dpi=dpi, figsize=figsize, layout="constrained")
    for k in range(4):
        i, j = _TENSOR_VECTOR_MAP[k]
        axs[k].set_prop_cycle(cyclers)
        for y in data:
            axs[k].plot(time, y[:, i, j], **styles, **kwargs)
        if x_label:
            axs[k].set_xlabel(x_label, fontsize=12)
        if y_label:
            axs[k].set_ylabel(y_label[k], fontsize=12)
    if curve_labels:
        fig.legend(curve_labels, **_LEGEND_KWARGS)
    if x_lim:
        plt.setp(fig.axes, xlim=x_lim)
    if y_lim:
        plt.setp(fig.axes, ylim=y_lim)
    if fout:
        fig.savefig(fout)
    else:
        plt.show()
    plt.close(fig)


def plot_stress_vs_time_1D(
    time: Arr[f64],
    *data: Arr[f64],
    x_lim: list[float] | None = None,
    y_lim: list[float] | None = None,
    figsize: tuple[float, float] = (4, 3),
    dpi: int = 150,
    x_label: str = r"time (s)",
    y_label: str = r"$S$ (kPa)",
    curve_labels: list[str] | None = None,
    colors: list[str] = ["k", "r", "b", "g", "c", "m"],
    lines: list[str] = ["-", "-", "-", "-", "-", "-"],
    marker: list[str] = ["none", "none", "none", "none", "none", "none"],
    markersize: int | float = 4,
    markerskip: int | list[int] | float | list[float] | None = None,
    markeredgewidth: float = 0.3,
    fillstyle: str = "full",
    fout: str | None = None,
    **kwargs,
) -> None:
    styles = {
        "markersize": markersize,
        "markevery": markerskip,
        "fillstyle": fillstyle,
        "markeredgewidth": markeredgewidth,
    }
    cyclers = (
        cycler("color", colors)
        + cycler("linestyle", lines)
        + cycler("marker", marker)
        + cycler("mec", colors)
    )
    fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=figsize, layout="constrained")
    ax.set_prop_cycle(cyclers)
    for x, y in data:
        ax.plot(time, y[:, 0, 0], **styles, **kwargs)
    if x_label:
        ax.set_xlabel(x_label, fontsize=12)
    if y_label:
        ax.set_ylabel(y_label, fontsize=12)
    if curve_labels:
        fig.legend(curve_labels, **_LEGEND_KWARGS)
    if x_lim:
        plt.setp(fig.axes, xlim=x_lim)
    if y_lim:
        plt.setp(fig.axes, ylim=y_lim)
    if fout:
        fig.savefig(fout)
    else:
        plt.show()
    plt.close(fig)


def plot_stress_vs_time_2D(
    time: Arr[f64],
    *data: Arr[f64],
    x_lim: list[float] | None = None,
    y_lim: list[float] | None = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 150,
    x_label: str = r"time (s)",
    y_label: list[str] = [
        r"$S_{11}$ (kPa)",
        r"$S_{12}$ (kPa)",
        r"$S_{21}$ (kPa)",
        r"$S_{22}$ (kPa)",
    ],
    curve_labels: list[str] | None = None,
    colors: list[str] = ["k", "r", "b", "g", "c", "m"],
    lines: list[str] = ["-", "-", "-", "-", "-", "-"],
    marker: list[str] = ["none", "none", "none", "none", "none", "none"],
    markersize: int | float = 4,
    markerskip: int | list[int] | float | list[float] | None = None,
    markeredgewidth: float = 0.3,
    fillstyle: str = "full",
    fout: str | None = None,
    **kwargs,
) -> None:
    styles = {
        "markersize": markersize,
        "markevery": markerskip,
        "fillstyle": fillstyle,
        "markeredgewidth": markeredgewidth,
    }
    cyclers = (
        cycler("color", colors)
        + cycler("linestyle", lines)
        + cycler("marker", marker)
        + cycler("mec", colors)
    )
    fig, axs = plt.subplots(4, 1, dpi=dpi, figsize=figsize, layout="constrained")
    for k in range(4):
        i, j = _TENSOR_VECTOR_MAP[k]
        axs[k].set_prop_cycle(cyclers)
        for y in data:
            axs[k].plot(time, y[:, i, j], **styles, **kwargs)
        if x_label:
            axs[k].set_xlabel(x_label, fontsize=12)
        if y_label:
            axs[k].set_ylabel(y_label[k], fontsize=12)
    if curve_labels:
        fig.legend(curve_labels, **_LEGEND_KWARGS)
    if x_lim:
        plt.setp(fig.axes, xlim=x_lim)
    if y_lim:
        plt.setp(fig.axes, ylim=y_lim)
    if fout:
        fig.savefig(fout)
    else:
        plt.show()
    plt.close(fig)
