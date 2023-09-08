import numpy as np
from biomechanics import *


def main():
    """
    First, we need to define time domain of the simulation.
    Numpy's linspace function is convenient. The first parameter is the staring time,
    the second parameter is the end time, the third parameter is the number of time
    steps + 1.
    Lets start by considering an experiment taken over the course of 1110s, with a
    dt of 0.1s.
    """
    period: float = 30.0
    time = np.linspace(0, 1110, 11101, dtype=f64)
    """
    Next we need to construct the individual components of the deformation gradient.
    Here are 3 examples of wave functions.
    sin^2(pi*t) is a good choice for a sinusoidal loading curve as it have zero velocity
    at time = 0s
    A triangle wave which start at can be created by taking the arccos of a cos function
    A ramping curve can be created using the clip function from numpy, here we clip all
    points between 0 and 1
    """
    sinusoid_wave = np.sin(np.pi * time / period, dtype=f64) ** 2.0
    triangle_wave = np.arccos(np.cos(2 * np.pi * time / period)) / np.pi
    ramping_wave = np.clip(time / period, 0, 1)
    plot_scalar(
        (time, sinusoid_wave),
        (time, triangle_wave),
        (time, ramping_wave),
        figsize=(8, 3),
        x_label="time (s)",
        curve_labels=["sinusoid", "triangle", "ramping"],
    )
    """
    Let us consider sinusoidal loading with a period of 30s, it would look like this
    """
    loading_curve = np.sin(np.pi * time / period) ** 2.0
    plot_scalar(
        (time, loading_curve),
        figsize=(8, 3),
        x_label="time (s)",
    )
    """
    Let us create the following testing protocol,
        5 cycles of preconditioning at max strain, t = 0
        1 cycle of loading at max strain, t = 150s
        1 cycle at 80% strain t = 180
        1 cycle at 60%, t = 210
        1 cycle at 40%, t = 240
        1 cycle at 20%, t = 270
        1 cycle of loadding at max strain, t = 300
        1 cycle at 80% strain, t = 330
        1 cycle at 60%, t = 360
        1 cycle at 40%, t = 390
        1 cycle at 20%, t = 420
        1 cycle of loading at max strain, t = 450
        15s ramp to maximum strain, t = 480
        600s hold at maximum strain, t = 495
        15s unloading, 1095
    """
    loading_curve[1800:2100] = 0.8 * loading_curve[1800:2100]
    loading_curve[2100:2400] = 0.6 * loading_curve[2100:2400]
    loading_curve[2400:2700] = 0.4 * loading_curve[2400:2700]
    loading_curve[2700:3000] = 0.2 * loading_curve[2700:3000]
    loading_curve[3300:3600] = 0.8 * loading_curve[3300:3600]
    loading_curve[3600:3900] = 0.6 * loading_curve[3600:3900]
    loading_curve[3900:4200] = 0.4 * loading_curve[3900:4200]
    loading_curve[4200:4500] = 0.2 * loading_curve[4200:4500]
    loading_curve[4800:10950] = np.clip((time[4800:10950] - time[4800]) / 15.0, 0, 1)
    loading_curve[10950:] = 1.0 - (time[10950:] - time[10950]) / 15.0
    plot_scalar(
        (time, loading_curve),
        figsize=(8, 3),
        x_label="time (s)",
    )
    r"""
    Now let us create the deformation gradient tensor for this experiment
    assuming a maximum strain applied is 40%, i.e. a stretch of 1.4. Afterwards, let us
    calculate the green lagrange strain and plot that
    """
    stretch = 1.0 + 0.4 * loading_curve
    F = construct_tensor_uniaxial(stretch)
    E = compute_green_lagrange_strain(F)
    plot_strain_vs_time_1D(time, E)
    """
    Now let us create a biaxial testing protocol,
        5 cycles of preconditioning at max strain, t = 0
        1 cycle of loading at max strain, t = 150s
        1 cycle at 80% strain for axis 1 t = 180
        1 cycle at 60% strain for axis 1, t = 210
        1 cycle at 40% strain for axis 1, t = 240
        1 cycle at 20% strain for axis 1, t = 270
        1 cycle of loadding at max strain, t = 300
        1 cycle at 80% strain for axis 2, t = 330
        1 cycle at 60% strain for axis 2, t = 360
        1 cycle at 40% strain for axis 2, t = 390
        1 cycle at 20% strain for axis 2, t = 420
        1 cycle of loading at max strain, t = 450
        15s ramp to maximum strain, t = 480
        600s hold at maximum strain, t = 495
        15s unloading, 1095
    """
    loading_curve_1 = np.sin(np.pi * time / period) ** 2
    loading_curve_1[1800:2100] = 0.8 * loading_curve_1[1800:2100]
    loading_curve_1[2100:2400] = 0.6 * loading_curve_1[2100:2400]
    loading_curve_1[2400:2700] = 0.4 * loading_curve_1[2400:2700]
    loading_curve_1[2700:3000] = 0.2 * loading_curve_1[2700:3000]
    loading_curve_1[4800:10950] = np.clip((time[4800:10950] - time[4800]) / 15.0, 0, 1)
    loading_curve_1[10950:] = 1.0 - (time[10950:] - time[10950]) / 15.0
    loading_curve_2 = np.sin(np.pi * time / period) ** 2
    loading_curve_2[3300:3600] = 0.8 * loading_curve_2[3300:3600]
    loading_curve_2[3600:3900] = 0.6 * loading_curve_2[3600:3900]
    loading_curve_2[3900:4200] = 0.4 * loading_curve_2[3900:4200]
    loading_curve_2[4200:4500] = 0.2 * loading_curve_2[4200:4500]
    loading_curve_2[4800:10950] = np.clip((time[4800:10950] - time[4800]) / 15.0, 0, 1)
    loading_curve_2[10950:] = 1.0 - (time[10950:] - time[10950]) / 15.0
    """
    Similarly calculate the Green's strain, assuming no shear, and plot
    """
    stretch1 = 1.0 + 0.4 * loading_curve_1
    stretch2 = 1.0 + 0.4 * loading_curve_2
    F = construct_tensor_biaxial(stretch1, 0, 0, stretch2)
    E = compute_green_lagrange_strain(F)
    plot_strain_vs_time_2D(time, E)


if __name__ == "__main__":
    main()
