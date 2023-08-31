# NordslettenSummerSchool
For Graz Biomechanical Summer School

### Before Getting Started

Before you get started, make sure to install python. To avoid contaminating your default Python setup, please use a virtual environment.

To create a virtual environment, e.g. in .venv of the current working directory, use the command
```console
python -m venv ./.venv
```
Activate your virtual environment. If you are using Windows, use
```console
.\.venv\Scripts\activate
```
If you are using Linux, macOS, or other Unix systems, use
```console
source ./venv/bin/activate
```
You can exit the virtual environment at any time by executing the command `deactivate`

## Getting Started

The NordslettenSummerSchool python module and all dependencies can be installed with `pip`.

For example, after you git clone or download the `nordsletten_summer_school` directory, you can install it with

```console
pip install ./nordsletten_summer_school
```
In case your `pip` executable is not bound to the same `python` executable, you call pip from python. I.e., you can alternatively call
```console
python -m pip install ./nordsletten_summer_school
```

If your current directory is inside `nordsletten_summer_school`, you can install the module via
```console
python -m pip install .
```

Once this is done, you are ready to go.

## Content

### Kinematics

This module contains the following for generating deformation gradient data:
```python
def construct_tensor_uniaxial(stretch: NDArray[f64]) -> NDArray[f64]: pass

def construct_tensor_biaxial(
    F11: NDArray[f64] | float = 1.0,
    F12: NDArray[f64] | float = 0.0,
    F21: NDArray[f64] | float = 0.0,
    F22: NDArray[f64] | float = 1.0,
) -> NDArray[f64]: pass
```

This module also contains the following for converting the deformation gradient to other strain tensor types:
```python
def compute_right_cauchy_green(F: Arr[f64]) -> Arr[f64]: pass

def compute_left_cauchy_green(F: Arr[f64]) -> Arr[f64]: pass

def compute_green_lagrange_strain(F: Arr[f64]) -> Arr[f64]: pass
```

### Constitutive Models

All constitutive models are Python classes that instantiate with their material parameters and provide a method `pk2` that returns the second Piola Kirchhoff stress tensor.

This module also contains the following functions for converting to other stress tensor types:

```python
def compute_pk1_from_pk2(S: Arr[f64], F: Arr[f64]) -> Arr[f64]: pass

def compute_cauchy_from_pk2(S: Arr[f64], F: Arr[f64]) -> Arr[f64]: pass
```

#### Hyperelastic Constitutive Models
All hyperelastic models inherit from
```python
class HyperelasticModel(abc.ABC):

    @abc.abstractmethod
    def pk2(self, F: Arr[f64]) -> Arr[f64]:
        pass
```
i.e., they all have the method `pk2` for calculating the second Piola Kirchhoff stress tensor.



The list of models includes:
```python
class NeoHookeanModel(HyperelasticModel):
    mu: float # Bulk Modulus

    def __init__(self,
        mu:float # Bulk Modulus
    ) -> None: pass
```


```python
class GuccioneModel(HyperelasticModel):
    mu: float  # Bulk Modulus
    b1: float  # Isotropic Exponent
    b2: Arr[f64]  # Array of fiber Exponents
    H: Arr[f64]  # Structural tensor for isotropic part, i.e. identity
    fiber: Arr[f64]  # Fiber orientation array

    def __init__(self,
        mu: float, # Bulk Modulus
        b_1: float, # Isotopic Exponent
        b_ff: float, # Fiber Exponent
        b_fs: float, # Fiber Shear Exponent
        b_sn: float, # Off-fiber interaction exponent
        v_f: Arr[f64], # Unit vector for fiber direction
        v_s: Arr[f64], # Unit vector for fiber sheet direction
    ) -> None: pass
```


```python
class CostaModel(HyperelasticModel):
    mu: float  # Bulk Modulus
    b: Arr[f64]  # Array of fiber Exponents
    fiber: Arr[f64]  # Fiber orientation array

    def __init__(self,
        mu: float, # Bulk modulus
        b_ff: float, # Fiber direction exponent
        b_ss: float, # Sheet direction exponent
        b_nn: float, # Normal direction exponent
        b_fs: float, # Fiber-sheet interation exponent
        b_fn: float, # Fiber-normal interation exponent
        b_sn: float, # Sheet-normal interation exponent
        v_f: Arr[f64], # Unit vector for fiber direction
        v_s: Arr[f64], # Unit vector for fiber sheet direction
    ) -> None: pass
```

```python
class HolzapfelOgdenModel(HyperelasticModel):
    k_iso: float # Isotropic part modulus
    b_iso: float # Isotropic part exponent
    k_fiber: Arr[f64] # Anisotropic modulus array
    b_fiber: Arr[f64] # Anisotropic exponent array
    fiber: Arr[f64]  # Fiber orientation array

    def __init__(self,
        k_iso: float, # Isotropic part modulus
        b_iso: float, # Isotropic part exponent
        k_ff: float, # Fiber modulus
        b_ff: float, # Fiber exponent
        k_fs: float, # Fiber-sheet interaction modulus
        b_fs: float, # Fiber-sheet interaction exponent
        k_ss: float, # Sheet modulus
        b_ss: float, # Sheet exponent
        v_f: Arr[f64], # Unit vector for fiber direction
        v_s: Arr[f64], # Unit vector for fiber sheet direction
    ) -> None: pass
```



#### Viscoelastic Constitutive Models


#### Hydrostatic Pressure
You can add a hydrostatic pressure with

```python
def add_hydrostatic_pressure(S: Arr[f64], F: Arr[f64]) -> Arr[f64]
```

### Plotting

```python
def plot_stress_vs_strain_1D(
    *data: tuple[Arr[f64], Arr[f64]] | list[Arr[f64]], ...
) -> None: pass

def plot_stress_vs_strain_2D(
    *data: tuple[Arr[f64], Arr[f64]] | list[Arr[f64]], ...
) -> None: pass

def plot_strain_vs_time_1D(
    time: Arr[f64], *data: Arr[f64], ...
) -> None: pass

def plot_strain_vs_time_2D(
    time: Arr[f64], *data: Arr[f64], ...
) -> None: pass

def plot_stress_vs_time_1D(
    time: Arr[f64], *data: Arr[f64], ...
) -> None: pass

def plot_stress_vs_time_2D(
    time: Arr[f64], *data: Arr[f64], ...
) -> None: pass
```


