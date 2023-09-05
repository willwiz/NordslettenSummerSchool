import numpy as np
from biomechanics import *


def main():
    # Define time domain
    time = np.linspace(0, 5, 501, dtype=f64)
    # Wave forms, period 1s
    sinusoid_wave = np.sin(np.pi * time) ** 2
    triangle_wave = np.arccos(np.cos(2 * np.pi * time)) / np.pi
    ramping_wave = np.clip(0.5 * time, 0, 1)
    plot_scalar(
        (time, sinusoid_wave),
        (time, triangle_wave),
        (time, ramping_wave),
        x_label="time (s)",
        curve_labels=["sinusoid", "triangle", "ramping"],
        fout="loading_waves.png",
    )
    # Create deformation gradient
    stretch1 = 1.0 + 0.1 * sinusoid_wave
    shear12 = 0.2 * ramping_wave
    stretch2 = 1.0 + 0.1 * triangle_wave
    biaxial_deformation = construct_tensor_biaxial(stretch1, shear12, 0, stretch2)
    plot_strain_vs_time_2D(time, biaxial_deformation, fout="strain_vs_time.png")
    # Make fibers
    fiber_direction = np.array([1, 0, 0], dtype=f64)
    sheet_direction = np.array([0, 1, 0], dtype=f64)
    # Make some models
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
    # Compute some stresses
    gucci_stress = add_hydrostatic_pressure(
        gucci_model.pk2(biaxial_deformation), biaxial_deformation
    )
    costa_stress = add_hydrostatic_pressure(
        costa_model.pk2(biaxial_deformation), biaxial_deformation
    )
    holzapfel_stress = add_hydrostatic_pressure(
        holzapfel_model.pk2(biaxial_deformation), biaxial_deformation
    )
    green_strain = compute_green_lagrange_strain(biaxial_deformation)
    plot_stress_vs_strain_2D(
        (green_strain, gucci_stress),
        (green_strain, costa_stress),
        (green_strain, holzapfel_stress),
        curve_labels=["Guccioni", "Costa", "Holzapfel"],
        fout="hyperelastic_stresses.png",
    )
    # make some viscoelastic models
    frac_holzapfel_model = FractionalVEModel(0.2, 2.0, 9, models=[holzapfel_model])
    diff_holzapfel_model = FractionalDiffEqModel(
        0.2, 0.02, 2.0, 9, viscoelastic_models=[frac_holzapfel_model]
    )
    kevin_holzapfel_model = KelvinVoigtModel(1.0, models=[holzapfel_model])
    maxwell_holzapfel_model = MaxwellModel(0.1, models=[holzapfel_model])
    zener_holzapfel_model = ZenerModel(0.1, 1.0, viscoelastic_models=[holzapfel_model])
    # Compute some viscoelastic stresses
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
    plot_stress_vs_strain_2D(
        (green_strain, holzapfel_stress),
        (green_strain, frac_holzapfel_stress),
        (green_strain, diff_holzapfel_stress),
        (green_strain, kevin_holzapfel_stress),
        (green_strain, maxwell_holzapfel_stress),
        (green_strain, zener_holzapfel_stress),
        curve_labels=[
            "Holzapfel",
            "Frac Holzapfel",
            "Diffeq Holzapfel",
            "Kelvin Voigt",
            "Maxwell",
            "Zener",
        ],
        fout="viscoelastic_stresses.png",
    )
    plot_stress_vs_time_2D(
        time,
        holzapfel_stress,
        frac_holzapfel_stress,
        diff_holzapfel_stress,
        kevin_holzapfel_stress,
        maxwell_holzapfel_stress,
        zener_holzapfel_stress,
        curve_labels=[
            "Holzapfel",
            "Frac Holzapfel",
            "Diffeq Holzapfel",
            "Kelvin Voigt",
            "Maxwell",
            "Zener",
        ],
        fout="viscoelastic_vs_time.png",
    )


if __name__ == "__main__":
    main()
