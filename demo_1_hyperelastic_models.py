import numpy as np
from biomechanics import *


def main():
    """
    Let us create a relatively simple biaxial testing experiment. See demo 0
    """
    # Define time domain
    time = np.linspace(0, 4, 401, dtype=f64)
    # Wave forms, period 1s
    sinusoid_wave = np.sin(np.pi * time) ** 2
    triangle_wave = np.arccos(np.cos(2 * np.pi * time)) / np.pi
    ramping_wave = np.clip(0.5 * time, 0, 1)
    # Create deformation gradient
    stretch1 = 1.0 + 0.1 * sinusoid_wave
    shear12 = 0.2 * triangle_wave
    stretch2 = 1.0 + 0.1 * ramping_wave
    biaxial_deformation = construct_tensor_biaxial(stretch1, shear12, 0, stretch2)
    plot_strain_vs_time_2D(time, biaxial_deformation)
    """
    Let us create some models to test
    """
    # Make fibers directions
    fiber_direction = np.array([1, 0, 0], dtype=f64)
    sheet_direction = np.array([0, 1, 0], dtype=f64)
    # Models are class instances creates with given material parameters
    # To calculate the stress, call the pk2 method
    neo_model = NeoHookeanModel(0.5)
    gucci_model = GuccioneModel(
        0.05, 0.0, 33.27, 12.92, 11.99, fiber_direction, sheet_direction
    )
    costa_model = CostaModel(
        0.13, 33.27, 20.83, 2.63, 12.92, 11.99, 11.46, fiber_direction, sheet_direction
    )
    holzapfel_model = HolzapfelOgdenModel(
        0.61,
        7.5,
        4.56,
        35.31,
        0.46,
        5.09,
        3.70,
        33.24,
        fiber_direction,
        sheet_direction,
    )
    nordsletten_model = NordslettenModel(
        10.02,
        1.158,
        1.640,
        0.897,
        0.409,
        6.175,
        3.520,
        2.895,
        fiber_direction,
        sheet_direction,
    )
    # Compute some stresses
    # E.g. call the pk2 method to calculate the 2nd PK stress
    gucci_stress = gucci_model.pk2(biaxial_deformation)
    # Add the hydrostatic pressure by solving the incompressible problem
    gucci_stress = add_hydrostatic_pressure(gucci_stress, biaxial_deformation)
    # You can chain this into the same line
    costa_stress = add_hydrostatic_pressure(
        costa_model.pk2(biaxial_deformation), biaxial_deformation
    )
    holzapfel_stress = add_hydrostatic_pressure(
        holzapfel_model.pk2(biaxial_deformation), biaxial_deformation
    )
    nordsletten_stress = add_hydrostatic_pressure(
        nordsletten_model.pk2(biaxial_deformation), biaxial_deformation
    )
    """
    Plot the results, by default, stress vs strain plot assumes the x-axis is the
    green lagrange strain
    """
    green_strain = compute_green_lagrange_strain(biaxial_deformation)
    plot_stress_vs_strain_2D(
        (green_strain, gucci_stress),
        (green_strain, costa_stress),
        (green_strain, holzapfel_stress),
        (green_strain, nordsletten_stress),
        curve_labels=["Guccioni", "Costa", "Holzapfel", "Nordsletten"],
    )
    """
    Now let us try the full experiment from demo 0, with a maximum strain of 0.1
    """
    period: float = 30.0
    time = np.linspace(0, 1110, 11101, dtype=f64)
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
    stretch1 = 1.0 + 0.1 * loading_curve_1
    stretch2 = 1.0 + 0.1 * loading_curve_2
    F = construct_tensor_biaxial(stretch1, 0, 0, stretch2)
    # Calculate the stresses
    gucci_stress = add_hydrostatic_pressure(gucci_model.pk2(F), F)
    costa_stress = add_hydrostatic_pressure(costa_model.pk2(F), F)
    holzapfel_stress = add_hydrostatic_pressure(holzapfel_model.pk2(F), F)
    nordsletten_stress = add_hydrostatic_pressure(nordsletten_model.pk2(F), F)
    E = compute_green_lagrange_strain(F)
    plot_stress_vs_time_2D(
        time,
        gucci_stress,
        costa_stress,
        holzapfel_stress,
        nordsletten_stress,
        curve_labels=["Guccioni", "Costa", "Holzapfel", "Nordsletten"],
    )
    # We can also try plotting the cycles between preconditioning and relaxation
    # which were between 150s and 480s
    plot_stress_vs_strain_2D(
        (E[1500:4800], gucci_stress[1500:4800]),
        (E[1500:4800], costa_stress[1500:4800]),
        (E[1500:4800], holzapfel_stress[1500:4800]),
        (E[1500:4800], nordsletten_stress[1500:4800]),
        curve_labels=["Guccioni", "Costa", "Holzapfel", "Nordsletten"],
    )


if __name__ == "__main__":
    main()
