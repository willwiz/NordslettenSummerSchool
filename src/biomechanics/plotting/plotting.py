from typing import Dict, List, Tuple, TypedDict, Union
from matplotlib import markers
import numpy as np
from numpy.typing import NDArray as Arr
from numpy import float64 as f64, int32 as i32
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.figure as mplf


_TENSOR_VECTOR_MAP: Dict[int, Tuple[int, int]] = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 0),
    3: (1, 1),
}

_LEGEND_KWARGS: Dict[str, Union[str, int, float, bool, None]] = {
    "loc": "outside lower center",
    "handlelength": 1.0,
    "frameon": False,
    "fontsize": 9,
    "labelspacing": 0.25,
    "columnspacing": 1.0,
}


class PlotCyclers(TypedDict, total=False):
    color: List[str]
    mec: List[str]
    alpha: List[float]
    linestyle: List[str]
    linewidth: List[float]
    marker: List[str]


def _create_cyclers(
    color: Union[List[str], None] = ["k", "r", "b", "g", "c", "m"],
    alpha: Union[List[float], None] = None,
    linestyle: Union[List[str], None] = ["-", "-", "-", "-", "-", "-"],
    linewidth: Union[List[float], None] = None,
    marker: Union[List[str], None] = ["none", "none", "none", "none", "none", "none"],
) -> PlotCyclers:
    cyclers: PlotCyclers = {}
    if color:
        cyclers["color"] = color
        cyclers["mec"] = color
    if alpha:
        cyclers["alpha"] = alpha
    if linestyle:
        cyclers["linestyle"] = linestyle
    if marker:
        cyclers["marker"] = marker
    if linewidth:
        cyclers["linewidth"] = linewidth
    return cyclers


def benchmark_plot(
    x: Union[Arr[f64], Arr[i32]],
    *data: Arr[f64],
    x_lim: Union[List[float], None] = None,
    y_lim: Union[List[float], None] = None,
    figsize: Tuple[float, float] = (4, 3),
    dpi: int = 150,
    x_label: Union[str, None] = None,
    y_label: Union[str, None] = None,
    curve_labels: Union[List[str], None] = None,
    color: Union[List[str], None] = ["k", "r", "b", "g", "c", "m"],
    alpha: Union[List[float], None] = None,
    linestyle: Union[List[str], None] = None,
    linewidth: Union[List[float], None] = None,
    marker: Union[List[str], None] = None,
    markersize: Union[int, float] = 4,
    markerskip: Union[int, List[int], float, List[float], None] = None,
    markeredgewidth: float = 0.3,
    legendlabelcols: int = 4,
    fillstyle: str = "full",
    transparency: bool = False,
    fout: Union[str, None] = None,
    **kwargs,
) -> None:
    styles = {
        "markersize": markersize,
        "markevery": markerskip,
        "fillstyle": fillstyle,
        "markeredgewidth": markeredgewidth,
    }
    cyclers = _create_cyclers(color, alpha, linestyle, linewidth, marker)
    fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=figsize, layout="constrained")
    ax.set_prop_cycle(**cyclers)
    ax.invert_xaxis()
    for y in data:
        ax.loglog(x, y, **styles, **kwargs)
    if x_label:
        ax.set_xlabel(x_label, fontsize=12)
    if y_label:
        ax.set_ylabel(y_label, fontsize=12)
    if curve_labels:
        fig.legend(curve_labels, ncols=legendlabelcols, **_LEGEND_KWARGS)
    if x_lim:
        plt.setp(fig.axes, xlim=x_lim)
    if y_lim:
        plt.setp(fig.axes, ylim=y_lim)
    if fout:
        fig.savefig(fout, transparent=transparency)
    else:
        plt.show()
    plt.close(fig)


def plot_scalar(
    *data: Union[Tuple[Arr[f64], Arr[f64]], List[Arr[f64]]],
    x_lim: Union[List[float], None] = None,
    y_lim: Union[List[float], None] = None,
    figsize: Tuple[float, float] = (4, 3),
    dpi: int = 150,
    x_label: Union[str, None] = None,
    y_label: Union[str, None] = None,
    curve_labels: Union[List[str], None] = None,
    color: Union[List[str], None] = ["k", "r", "b", "g", "c", "m"],
    alpha: Union[List[float], None] = None,
    linestyle: Union[List[str], None] = None,
    linewidth: Union[List[float], None] = None,
    marker: Union[List[str], None] = None,
    markersize: Union[int, float] = 4,
    markerskip: Union[int, List[int], float, List[float], None] = None,
    markeredgewidth: float = 0.3,
    legendlabelcols: int = 4,
    fillstyle: str = "full",
    transparency: bool = False,
    fout: Union[str, None] = None,
    **kwargs,
) -> None:
    styles = {
        "markersize": markersize,
        "markevery": markerskip,
        "fillstyle": fillstyle,
        "markeredgewidth": markeredgewidth,
    }
    cyclers = _create_cyclers(color, alpha, linestyle, linewidth, marker)

    fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=figsize, layout="constrained")
    ax.set_prop_cycle(**cyclers)
    for x, y in data:
        ax.plot(x, y, **styles, **kwargs)
    if curve_labels:
        fig.legend(curve_labels, ncols=legendlabelcols, **_LEGEND_KWARGS)
    if x_label:
        ax.set_xlabel(x_label, fontsize=12)
    if y_label:
        ax.set_ylabel(y_label, fontsize=12)
    if x_lim:
        plt.setp(fig.axes, xlim=x_lim)
    if y_lim:
        plt.setp(fig.axes, ylim=y_lim)
    if fout:
        fig.savefig(fout, transparent=transparency)
    else:
        plt.show()
    plt.close(fig)


def plot_stress_vs_strain_1D(
    *data: Union[Tuple[Arr[f64], Arr[f64]], List[Arr[f64]]],
    x_lim: Union[List[float], None] = None,
    y_lim: Union[List[float], None] = None,
    figsize: Tuple[float, float] = (4, 3),
    dpi: int = 150,
    x_label: str = r"$E$",
    y_label: str = r"$S$ (kPa)",
    curve_labels: Union[List[str], None] = None,
    color: Union[List[str], None] = ["k", "r", "b", "g", "c", "m"],
    alpha: Union[List[float], None] = None,
    linestyle: Union[List[str], None] = None,
    linewidth: Union[List[float], None] = None,
    marker: Union[List[str], None] = None,
    markersize: Union[int, float] = 4,
    markerskip: Union[int, List[int], float, List[float], None] = None,
    markeredgewidth: float = 0.3,
    legendlabelcols: int = 4,
    fillstyle: str = "full",
    transparency: bool = False,
    fout: Union[str, None] = None,
    **kwargs,
) -> None:
    styles = {
        "markersize": markersize,
        "markevery": markerskip,
        "fillstyle": fillstyle,
        "markeredgewidth": markeredgewidth,
    }
    cyclers = _create_cyclers(color, alpha, linestyle, linewidth, marker)
    fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=figsize, layout="constrained")
    ax.set_prop_cycle(**cyclers)
    for x, y in data:
        ax.plot(x[:, 0, 0], y[:, 0, 0], **styles, **kwargs)
    if curve_labels:
        fig.legend(curve_labels, ncols=legendlabelcols, **_LEGEND_KWARGS)
    if x_label:
        ax.set_xlabel(x_label, fontsize=12)
    if y_label:
        ax.set_ylabel(y_label, fontsize=12)
    if x_lim:
        plt.setp(fig.axes, xlim=x_lim)
    if y_lim:
        plt.setp(fig.axes, ylim=y_lim)
    if fout:
        fig.savefig(fout, transparent=transparency)
    else:
        plt.show()
    plt.close(fig)


def plot_stress_vs_strain_2D(
    *data: Union[Tuple[Arr[f64], Arr[f64]], List[Arr[f64]]],
    x_lim: Union[List[float], None] = None,
    y_lim: Union[List[float], None] = None,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 150,
    x_label: List[str] = [r"$E_{11}$", r"$E_{12}$", r"$E_{21}$", r"$E_{22}$"],
    y_label: List[str] = [
        r"$S_{11}$ (kPa)",
        r"$S_{12}$ (kPa)",
        r"$S_{21}$ (kPa)",
        r"$S_{22}$ (kPa)",
    ],
    curve_labels: Union[List[str], None] = None,
    color: Union[List[str], None] = ["k", "r", "b", "g", "c", "m"],
    alpha: Union[List[float], None] = None,
    linestyle: Union[List[str], None] = None,
    linewidth: Union[List[float], None] = None,
    marker: Union[List[str], None] = None,
    markersize: Union[int, float] = 4,
    markerskip: Union[int, List[int], float, List[float], None] = None,
    markeredgewidth: float = 0.3,
    legendlabelcols: int = 4,
    fillstyle: str = "full",
    transparency: bool = False,
    fout: Union[str, None] = None,
    **kwargs,
) -> None:
    styles = {
        "markersize": markersize,
        "markevery": markerskip,
        "fillstyle": fillstyle,
        "markeredgewidth": markeredgewidth,
    }
    cyclers = _create_cyclers(color, alpha, linestyle, linewidth, marker)
    fig, axs = plt.subplots(2, 2, dpi=dpi, figsize=figsize, layout="constrained")

    for k in range(4):
        i, j = _TENSOR_VECTOR_MAP[k]
        axs[i, j].set_prop_cycle(**cyclers)
        for x, y in data:
            axs[i, j].plot(x[:, i, j], y[:, i, j], **styles, **kwargs)
        if x_label:
            axs[i, j].set_xlabel(x_label[k], fontsize=12)
        if y_label:
            axs[i, j].set_ylabel(y_label[k], fontsize=12)
    if curve_labels:
        fig.legend(curve_labels, ncols=legendlabelcols, **_LEGEND_KWARGS)
    if x_lim:
        plt.setp(fig.axes, xlim=x_lim)
    if y_lim:
        plt.setp(fig.axes, ylim=y_lim)
    if fout:
        fig.savefig(fout, transparent=transparency)
    else:
        plt.show()
    plt.close(fig)


def plot_strain_vs_time_1D(
    time: Arr[f64],
    *data: Arr[f64],
    x_lim: Union[List[float], None] = None,
    y_lim: Union[List[float], None] = None,
    figsize: Tuple[float, float] = (4, 3),
    dpi: int = 150,
    x_label: str = r"time (s)",
    y_label: str = r"$E$",
    curve_labels: Union[List[str], None] = None,
    color: Union[List[str], None] = ["k", "r", "b", "g", "c", "m"],
    alpha: Union[List[float], None] = None,
    linestyle: Union[List[str], None] = None,
    linewidth: Union[List[float], None] = None,
    marker: Union[List[str], None] = None,
    markersize: Union[int, float] = 4,
    markerskip: Union[int, List[int], float, List[float], None] = None,
    markeredgewidth: float = 0.3,
    legendlabelcols: int = 4,
    fillstyle: str = "full",
    transparency: bool = False,
    fout: Union[str, None] = None,
    **kwargs,
) -> None:
    styles = {
        "markersize": markersize,
        "markevery": markerskip,
        "fillstyle": fillstyle,
        "markeredgewidth": markeredgewidth,
    }
    cyclers = _create_cyclers(color, alpha, linestyle, linewidth, marker)
    fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=figsize, layout="constrained")
    ax.set_prop_cycle(**cyclers)
    for y in data:
        ax.plot(time, y[:, 0, 0], **styles, **kwargs)
    if x_label:
        ax.set_xlabel(x_label, fontsize=12)
    if y_label:
        ax.set_ylabel(y_label, fontsize=12)
    if curve_labels:
        fig.legend(curve_labels, ncols=legendlabelcols, **_LEGEND_KWARGS)
    if x_lim:
        plt.setp(fig.axes, xlim=x_lim)
    if y_lim:
        plt.setp(fig.axes, ylim=y_lim)
    if fout:
        fig.savefig(fout, transparent=transparency)
    else:
        plt.show()
    plt.close(fig)


def plot_strain_vs_time_2D(
    time: Arr[f64],
    *data: Arr[f64],
    x_lim: Union[List[float], None] = None,
    y_lim: Union[List[float], None] = None,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 150,
    x_label: str = r"time (s)",
    y_label: List[str] = [
        r"$E_{11}$",
        r"$E_{12}$",
        r"$E_{21}$",
        r"$E_{22}$",
    ],
    curve_labels: Union[List[str], None] = None,
    color: Union[List[str], None] = ["k", "r", "b", "g", "c", "m"],
    alpha: Union[List[float], None] = None,
    linestyle: Union[List[str], None] = None,
    linewidth: Union[List[float], None] = None,
    marker: Union[List[str], None] = None,
    markersize: Union[int, float] = 4,
    markerskip: Union[int, List[int], float, List[float], None] = None,
    markeredgewidth: float = 0.3,
    legendlabelcols: int = 4,
    fillstyle: str = "full",
    transparency: bool = False,
    fout: Union[str, None] = None,
    **kwargs,
) -> None:
    styles = {
        "markersize": markersize,
        "markevery": markerskip,
        "fillstyle": fillstyle,
        "markeredgewidth": markeredgewidth,
    }
    cyclers = _create_cyclers(color, alpha, linestyle, linewidth, marker)
    fig, axs = plt.subplots(4, 1, dpi=dpi, figsize=figsize, layout="constrained")
    for k in range(4):
        i, j = _TENSOR_VECTOR_MAP[k]
        axs[k].set_prop_cycle(**cyclers)
        for y in data:
            axs[k].plot(time, y[:, i, j], **styles, **kwargs)
        if x_label:
            axs[k].set_xlabel(x_label, fontsize=12)
        if y_label:
            axs[k].set_ylabel(y_label[k], fontsize=12)
    if curve_labels:
        fig.legend(curve_labels, ncols=legendlabelcols, **_LEGEND_KWARGS)
    if x_lim:
        plt.setp(fig.axes, xlim=x_lim)
    if y_lim:
        plt.setp(fig.axes, ylim=y_lim)
    if fout:
        fig.savefig(fout, transparent=transparency)
    else:
        plt.show()
    plt.close(fig)


def plot_stress_vs_time_1D(
    time: Arr[f64],
    *data: Arr[f64],
    x_lim: Union[List[float], None] = None,
    y_lim: Union[List[float], None] = None,
    figsize: Tuple[float, float] = (4, 3),
    dpi: int = 150,
    x_label: str = r"time (s)",
    y_label: str = r"$S$ (kPa)",
    curve_labels: Union[List[str], None] = None,
    color: Union[List[str], None] = ["k", "r", "b", "g", "c", "m"],
    alpha: Union[List[float], None] = None,
    linestyle: Union[List[str], None] = None,
    linewidth: Union[List[float], None] = None,
    marker: Union[List[str], None] = None,
    markersize: Union[int, float] = 4,
    markerskip: Union[int, List[int], float, List[float], None] = None,
    markeredgewidth: float = 0.3,
    legendlabelcols: int = 4,
    fillstyle: str = "full",
    transparency: bool = False,
    fout: Union[str, None] = None,
    **kwargs,
) -> None:
    styles = {
        "markersize": markersize,
        "markevery": markerskip,
        "fillstyle": fillstyle,
        "markeredgewidth": markeredgewidth,
    }
    cyclers = _create_cyclers(color, alpha, linestyle, linewidth, marker)
    fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=figsize, layout="constrained")
    ax.set_prop_cycle(**cyclers)
    for y in data:
        ax.plot(time, y[:, 0, 0], **styles, **kwargs)
    if x_label:
        ax.set_xlabel(x_label, fontsize=12)
    if y_label:
        ax.set_ylabel(y_label, fontsize=12)
    if curve_labels:
        fig.legend(curve_labels, ncols=legendlabelcols, **_LEGEND_KWARGS)
    if x_lim:
        plt.setp(fig.axes, xlim=x_lim)
    if y_lim:
        plt.setp(fig.axes, ylim=y_lim)
    if fout:
        fig.savefig(fout, transparent=transparency)
    else:
        plt.show()
    plt.close(fig)


def plot_stress_vs_time_2D(
    time: Arr[f64],
    *data: Arr[f64],
    x_lim: Union[List[float], None] = None,
    y_lim: Union[List[float], None] = None,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 150,
    x_label: str = r"time (s)",
    y_label: List[str] = [
        r"$S_{11}$ (kPa)",
        r"$S_{12}$ (kPa)",
        r"$S_{21}$ (kPa)",
        r"$S_{22}$ (kPa)",
    ],
    curve_labels: Union[List[str], None] = None,
    color: Union[List[str], None] = ["k", "r", "b", "g", "c", "m"],
    alpha: Union[List[float], None] = None,
    linestyle: Union[List[str], None] = None,
    linewidth: Union[List[float], None] = None,
    marker: Union[List[str], None] = None,
    markersize: Union[int, float] = 4,
    markerskip: Union[int, List[int], float, List[float], None] = None,
    markeredgewidth: float = 0.3,
    legendlabelcols: int = 4,
    fillstyle: str = "full",
    transparency: bool = False,
    fout: Union[str, None] = None,
    **kwargs,
) -> None:
    styles = {
        "markersize": markersize,
        "markevery": markerskip,
        "fillstyle": fillstyle,
        "markeredgewidth": markeredgewidth,
    }
    cyclers = _create_cyclers(color, alpha, linestyle, linewidth, marker)
    fig, axs = plt.subplots(4, 1, dpi=dpi, figsize=figsize, layout="constrained")
    for k in range(4):
        i, j = _TENSOR_VECTOR_MAP[k]
        axs[k].set_prop_cycle(**cyclers)
        for y in data:
            axs[k].plot(time, y[:, i, j], **styles, **kwargs)
        if x_label:
            axs[k].set_xlabel(x_label, fontsize=12)
        if y_label:
            axs[k].set_ylabel(y_label[k], fontsize=12)
    if curve_labels:
        fig.legend(curve_labels, ncols=legendlabelcols, **_LEGEND_KWARGS)
    if x_lim:
        plt.setp(fig.axes, xlim=x_lim)
    if y_lim:
        plt.setp(fig.axes, ylim=y_lim)
    if fout:
        fig.savefig(fout, transparent=transparency)
    else:
        plt.show()
    plt.close(fig)
