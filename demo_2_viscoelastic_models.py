import numpy as np
from biomechanics import *


def main():
    """
    demo 1 but with viscoelastic models
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
    """
    Let us create some models to test
    """
    # Make fibers directions
    fiber_direction = np.array([1, 0, 0], dtype=f64)
    sheet_direction = np.array([0, 1, 0], dtype=f64)
    # Models are class instances creates with given material parameters
    # To calculate the stress, call the pk2 method
    # Make some models
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
    green_strain = compute_green_lagrange_strain(biaxial_deformation)
    # make some viscoelastic models
    frac_holzapfel_model = FractionalVEModel(0.2, 2.0, 9, models=[holzapfel_model])
    diff_holzapfel_model = FractionalDiffEqModel(
        0.2, 0.02, 2.0, 9, viscoelastic_models=[frac_holzapfel_model]
    )
    kevin_holzapfel_model = KelvinVoigtModel(1.0, models=[holzapfel_model])
    maxwell_holzapfel_model = MaxwellModel(0.5, models=[holzapfel_model])
    zener_holzapfel_model = ZenerModel(0.5, 1.0, viscoelastic_models=[holzapfel_model])
    # Compute some viscoelastic stresses, requires an extra argument of time
    # Some multiplier are added for visual purposes
    holzapfel_stress = add_hydrostatic_pressure(
        holzapfel_model.pk2(biaxial_deformation), biaxial_deformation
    )
    frac_holzapfel_stress = add_hydrostatic_pressure(
        frac_holzapfel_model.pk2(biaxial_deformation, time), biaxial_deformation
    )
    diff_holzapfel_stress = add_hydrostatic_pressure(
        diff_holzapfel_model.pk2(biaxial_deformation, time), biaxial_deformation
    )
    kevin_holzapfel_stress = (1 / 3) * add_hydrostatic_pressure(
        kevin_holzapfel_model.pk2(biaxial_deformation, time), biaxial_deformation
    )
    maxwell_holzapfel_stress = 15 * add_hydrostatic_pressure(
        maxwell_holzapfel_model.pk2(biaxial_deformation, time), biaxial_deformation
    )
    zener_holzapfel_stress = 3 * add_hydrostatic_pressure(
        zener_holzapfel_model.pk2(biaxial_deformation, time), biaxial_deformation
    )
    """
    Plot the results, by default, stress vs strain plot assumes the x-axis is the
    green lagrange strain
    """
    green_strain = compute_green_lagrange_strain(biaxial_deformation)
    plot_stress_vs_strain_2D(
        (green_strain, holzapfel_stress),
        (green_strain, frac_holzapfel_stress),
        (green_strain, diff_holzapfel_stress),
        (green_strain, kevin_holzapfel_stress),
        (green_strain, maxwell_holzapfel_stress),
        (green_strain, zener_holzapfel_stress),
        curve_labels=[
            "Hyperelastic",
            "Fractional",
            "Frac. Diffeq",
            "Kelvin Voigt",
            "Maxwell",
            "Zener",
        ],
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
    # Some multiplier are added for visual purposes
    holzapfel_stress = add_hydrostatic_pressure(holzapfel_model.pk2(F), F)
    frac_holzapfel_stress = add_hydrostatic_pressure(
        frac_holzapfel_model.pk2(F, time), F
    )
    diff_holzapfel_stress = add_hydrostatic_pressure(
        diff_holzapfel_model.pk2(F, time), F
    )
    kevin_holzapfel_stress = 5 * add_hydrostatic_pressure(
        kevin_holzapfel_model.pk2(F, time), F
    )
    maxwell_holzapfel_stress = add_hydrostatic_pressure(
        maxwell_holzapfel_model.pk2(F, time), F
    )
    zener_holzapfel_stress = 5 * add_hydrostatic_pressure(
        zener_holzapfel_model.pk2(F, time), F
    )
    E = compute_green_lagrange_strain(F)
    plot_stress_vs_time_2D(
        time,
        holzapfel_stress,
        frac_holzapfel_stress,
        diff_holzapfel_stress,
        kevin_holzapfel_stress,
        maxwell_holzapfel_stress,
        zener_holzapfel_stress,
        curve_labels=[
            "Hyperelastic",
            "Fractional",
            "Frac. Diffeq",
            "Kelvin Voigt",
            "Maxwell",
            "Zener",
        ],
    )
    # We can also try plotting the cycles between preconditioning and relaxation
    # which were between 150s and 480s
    plot_stress_vs_strain_2D(
        (E[1500:4800], holzapfel_stress[1500:4800]),
        (E[1500:4800], frac_holzapfel_stress[1500:4800]),
        (E[1500:4800], diff_holzapfel_stress[1500:4800]),
        (E[1500:4800], kevin_holzapfel_stress[1500:4800]),
        (E[1500:4800], maxwell_holzapfel_stress[1500:4800]),
        (E[1500:4800], zener_holzapfel_stress[1500:4800]),
        curve_labels=[
            "Hyperelastic",
            "Fractional",
            "Frac. Diffeq",
            "Kelvin Voigt",
            "Maxwell",
            "Zener",
        ],
    )


if __name__ == "__main__":
    main()
