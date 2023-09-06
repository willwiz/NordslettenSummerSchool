import numpy as np
from biomechanics import *


def benchmarkit(pars: np.ndarray, alpha: float, dt: float, Np: int) -> float:
    nt = round(5.0 / dt) + 1
    time = np.linspace(0, 5, nt, dtype=f64)
    poly = polynomial_function(pars, time)
    analytic = analytical_polynomial_fractionalderiv_function(0.1, pars, time)
    carp = CaputoInitialize(alpha, 5.0, Np=Np)
    delta_t = np.diff(time, prepend=-1)
    approximation = caputo_derivative_linear(carp, poly, delta_t)
    residual = approximation - analytic
    return np.sqrt(dt * (residual @ residual))


def main():
    # Define time domain
    time = np.linspace(0, 5, 501, dtype=f64)
    # Wave forms, period 1s
    linear = polynomial_function([1 / time[-1]], time)
    quadratic = polynomial_function([0, 1 / time[-1] ** 2], time)
    cubic = polynomial_function([0, 0, 1 / time[-1] ** 3], time)
    plot_scalar(
        (time, linear),
        (time, quadratic),
        (time, cubic),
        x_label="time (s)",
        curve_labels=["linear", "quadratic", "cubic"],
    )
    linear_analytic = analytical_polynomial_fractionalderiv_function(
        0.1, [1 / time[-1]], time
    )
    quadratic_analytic = analytical_polynomial_fractionalderiv_function(
        0.1, [0, 1 / time[-1] ** 2], time
    )
    cubic_analytic = analytical_polynomial_fractionalderiv_function(
        0.1, [0, 0, 1 / time[-1] ** 3], time
    )
    plot_scalar(
        (time, linear_analytic),
        (time, quadratic_analytic),
        (time, cubic_analytic),
        x_label="time (s)",
        curve_labels=["linear", "quadratic", "cubic"],
    )
    # Create deformation gradient
    carp = CaputoInitialize(0.1, 5.0, Np=3)
    dt = np.diff(time, prepend=-1)
    linear_approximation = caputo_derivative_linear(carp, linear, dt)
    quadratic_approximation = caputo_derivative_linear(carp, quadratic, dt)
    cubic_approximation = caputo_derivative_linear(carp, cubic, dt)
    plot_scalar(
        (time, linear_analytic),
        (time, quadratic_analytic),
        (time, cubic_analytic),
        (time, linear_approximation),
        (time, quadratic_approximation),
        (time, cubic_approximation),
        color=["k", "r", "b", "k", "r", "b"],
        alpha=[0.2, 0.2, 0.2, 1.0, 1.0, 1.0],
        linewidth=[4, 4, 4, 0.75, 0.75, 0.75],
        linestyle=["-", "-", "-", "-", "-", "-"],
        x_label="time (s)",
        curve_labels=[
            "linear (Analytic)",
            "quad (Analytic)",
            "cubic (Analytic)",
            "linear (Approx)",
            "quad (Approx)",
            "cubic (Approx)",
        ],
        legendlabelcols=3,
    )
    # Benchmark one
    benchmarks = [
        [
            benchmarkit([0, 0, 1 / 5.0**3], 0.1, d, n)
            for d in [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4]
        ]
        for n in [3, 6, 9, 12, 15]
    ]
    benchmark_plot(
        np.array([1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4]),
        *benchmarks,
        x_label="dt (s)",
        y_label="$l^2$-norm",
        curve_labels=[r"Np=3", r"Np=6", r"Np=9", r"Np=12", r"Np=15"],
        legendlabelcols=5,
    )


if __name__ == "__main__":
    main()
