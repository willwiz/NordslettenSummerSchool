# NordslettenSummerSchool
For Graz Biomechanical Summer School

### Before Getting Started

Before you get started, make sure to install Python. To avoid contaminating your default Python setup, please use a virtual environment.

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

Once this is done, you are ready to go. All functions can be imported from the `biomechanics` module, i.e.,
```python
from biomechanics import *
```

See `example.py` for example.

## Content

### Kinematics

The package contains tools for describing incompressible deformations in 1D and 2D. Specifically, 1D deformations are of the form
$$
\begin{equation}
\mathbf{F}(t) = \begin{bmatrix}
\lambda(t) & 0 & 0\\
0 & \frac{1}{\sqrt{\lambda(t)}} & 0 \\
0& 0& \frac{1}{\sqrt{\lambda(t)}}
\end{bmatrix},
\end{equation}
$$
whereas 2D deformations are of the form
$$
\begin{equation}
\mathbf{F}(t) = \begin{bmatrix}
F_{11}(t) & F_{12}(t) & 0\\
F_{21}(t) & F_{22}(t) & 0 \\
0& 0& \frac{1}{F_{11}(t) F_{22}(t) - F_{12}(t)F_{21}(t)}
\end{bmatrix}.
\end{equation}
$$
The main components of the tensors are up to the users to define, and the following functions are provided to create the deformation gradient tensors.
```python
def construct_tensor_uniaxial(stretch: NDArray[f64]) -> NDArray[f64]: pass

def construct_tensor_biaxial(
    F11: NDArray[f64] | float = 1.0,
    F12: NDArray[f64] | float = 0.0,
    F21: NDArray[f64] | float = 0.0,
    F22: NDArray[f64] | float = 1.0,
) -> NDArray[f64]: pass
```




This module also contains the following for converting the deformation gradient to other strain tensor types, right Cauchy Green strain ($\mathbf{C} = \mathbf{F}^T\mathbf{F}$), left Cauchy Green strain ($\mathbf{B}= \mathbf{F}\mathbf{F}^T$), and Green-Lagrange strain ($\mathbf{E} = \frac{1}{2}\left(\mathbf{C} - \mathbf{I}\right)$).
```python
def compute_right_cauchy_green(F: NDArray[f64]) -> NDArray[f64]: pass

def compute_left_cauchy_green(F: NDArray[f64]) -> NDArray[f64]: pass

def compute_green_lagrange_strain(F: NDArray[f64]) -> NDArray[f64]: pass
```



Similarly, this package also contains the following functions for converting to other stress tensor types from the 2nd Piola Kirchhoff stress: 1st Piola Kirchhoff stress ($\mathbf{P}$) and the Cauchy stress ($\mathbf{\sigma})

```python
def compute_pk1_from_pk2(S: NDArray[f64], F: NDArray[f64]) -> NDArray[f64]: pass

def compute_cauchy_from_pk2(S: NDArray[f64], F: NDArray[f64]) -> NDArray[f64]: pass
```

### Constitutive Models

All constitutive models are Python classes that instantiate with their material parameters and provide a method `pk2` that returns the 2nd Piola Kirchhoff stress tensor.



#### Hyperelastic Constitutive Models


All hyperelastic models inherit from
```python
class HyperelasticModel(abc.ABC):

    @abc.abstractmethod
    def pk2(self, F: NDArray[f64]) -> NDArray[f64]:
        pass
```
i.e., they all have the method `pk2` for calculating the second Piola Kirchhoff stress tensor.



The list of models includes:
```python
class NeoHookeanModel(HyperelasticModel):

    def __init__(self,
        mu:float # Bulk Modulus
    ) -> None: pass
```


```python
class GuccioneModel(HyperelasticModel):

    def __init__(self,
        mu: float, # Bulk Modulus
        b_1: float, # Isotopic Exponent
        b_ff: float, # Fiber Exponent
        b_fs: float, # Fiber Shear Exponent
        b_sn: float, # Off-fiber interaction exponent
        v_f: NDArray[f64], # Unit vector for fiber direction
        v_s: NDArray[f64], # Unit vector for fiber sheet direction
    ) -> None: pass
```


```python
class CostaModel(HyperelasticModel):

    def __init__(self,
        mu: float, # Bulk modulus
        b_ff: float, # Fiber direction exponent
        b_ss: float, # Sheet direction exponent
        b_nn: float, # Normal direction exponent
        b_fs: float, # Fiber-sheet interation exponent
        b_fn: float, # Fiber-normal interation exponent
        b_sn: float, # Sheet-normal interation exponent
        v_f: NDArray[f64], # Unit vector for fiber direction
        v_s: NDArray[f64], # Unit vector for fiber sheet direction
    ) -> None: pass
```

```python
class HolzapfelOgdenModel(HyperelasticModel):

    def __init__(self,
        k_iso: float, # Isotropic part modulus
        b_iso: float, # Isotropic part exponent
        k_ff: float, # Fiber modulus
        b_ff: float, # Fiber exponent
        k_fs: float, # Fiber-sheet interaction modulus
        b_fs: float, # Fiber-sheet interaction exponent
        k_ss: float, # Sheet modulus
        b_ss: float, # Sheet exponent
        v_f: NDArray[f64], # Unit vector for fiber direction
        v_s: NDArray[f64], # Unit vector for fiber sheet direction
    ) -> None: pass
```



#### Viscoelastic Constitutive Models
All viscoelastic models inherit from
```python
class ViscoelasticModel(abc.ABC):

    @abc.abstractmethod
    def pk2(self, F: NDArray[f64], time: NDArray[f64]) -> NDArray[f64]:
        pass
```
thus needing an additional time argument.


Three classic viscoelastic models are provided, they operate on hyperelastic laws.
```python
class KelvinVoigtModel(ViscoelasticModel):

    def __init__(self,
        weight: float = 1.0, # multiplier on the stress
        models: list[HyperelasticModel] | None = None, # Models being differentiated
    ) -> None:
        self.w = weight
        self.laws = models if models else list()
```

```python
class MaxwellModel(ViscoelasticModel):

    def __init__(self,
        weight: float = 0.0, # weight on the derivative on the left hand side
        models: list[HyperelasticModel] | None = None, # Models on the right handside
    ) -> None:
        self.w = weight
        self.hlaws = models
```
```python
class ZenerModel(ViscoelasticModel):

    def __init__(self,
        weight_LHS: float = 0.0, # weight on the derivative on the left hand side
        weight_RHS: float = 1.0, # multiplier on the stress under time derivative on the RHS
        hyperelastic_models: list[HyperelasticModel] | None = None,
        viscoelastic_models: list[ViscoelasticModel] | None = None, # taken time derivative of
    ) -> None:
        self.w_right = weight_RHS
        self.w_left = weight_LHS
        self.hlaws = hyperelastic_models if hyperelastic_models else list()
        self.vlaws = viscoelastic_models if viscoelastic_models else list()
```

Two fractional viscoelastic models are provided, they operate on hyperelastic laws.

```python
class FractionalVEModel(ViscoelasticModel):

    def __init__(self,
        alpha: float, # Fractional order
        Tf: float, # Periodicity
        Np: int = 9, # Number of Prony terms
        models: list[HyperelasticModel] | None = None, # Models being differentiated
    ) -> None: pass
```

```python
class FractionalDiffEqModel(ViscoelasticModel):

    def __init__(self,
        alpha: float, # Fractional order
        delta: float, # Fractional term weight
        Tf: float, # Periodicity
        Np: int = 9, # Number of Prony terms
        hyperelastic_models: list[HyperelasticModel] | None = None, # Hyperelastic Models on RHS
        viscoelastic_models: list[ViscoelasticModel] | None = None, # Viscoelastic Models on RHS
    ) -> None: pass
```


#### Hydrostatic Pressure
You can add a hydrostatic pressure, i.e. the term $p(J - 1)$ in the constitutive equations, using

```python
def add_hydrostatic_pressure(S: NDArray[f64], F: NDArray[f64]) -> NDArray[f64]: pass
```


### Composing Models
The following two classes are provided for composing models:
```python
class CompositeHyperelasticModel(HyperelasticModel):

    def __init__(self,
        models: list[HyperelasticModel]
    ) -> None: pass
```
```python
class CompositeViscoelasticModel(ViscoelasticModel):

    def __init__(self,
        hyperelastic_models: list[HyperelasticModel] | None = None,
        viscoelastic_models: list[ViscoelasticModel] | None = None,
    ) -> None: pass
```

For example,
```python
    fractional_holzapfel_model = FractionalVEModel(0.15, 10.0, 9, [
        HolzapfelOgdenModel(1.0, .5, 1.0, 1.0, 1.0, 0.25, 1.0, 0.5, np.array([1,0,0]), np.array([0,1,0]))
    ])
    composite_model = CompositeViscoelasticModel(
        hyperelastic_models = [
            NeoHookean(1.0),
            GuccioneModel(1.0, 0.0., 1.0, 0.5, 0.5, np.array([1,0,0]), np.array([0,1,0])),
        ],
        viscoelastic_models = [
            fractional_holzapfel_model,
        ],
    )

    stress = composite_model.pk2(F_tensor, time)
```

### Benchmarking using polynomial functions

This package also provides some tools for benchmarking fractional derivatives on polynomial functions. Polynomial data can be generated with the function:
```python
def polynomial_function(pars: Arr[f64] | tuple[Arr[f64], Arr[i32]], time: Arr[f64]) -> Arr[f64]: pass
```
`pars` are the constant in front of each term of the polynomial in order starting from 1, i.e.,
$$
\begin{equation}
f(t) = \mathrm{pars}[0] t + \mathrm{pars}[1] t^2 + \mathrm{pars}[2] t^3 + \ldots
\end{equation}
$$
Alternatively, the exponents can be passed in as well in a tuple, i.e.,
$$
\begin{equation}
f(t) = \mathrm{pars}[0][0] t^{\mathrm{pars}[1][0]} + \mathrm{pars}[0][1] t^{\mathrm{pars}[1][1]}
+ \mathrm{pars}[0][2] t^{\mathrm{pars}[1][2]} + \ldots
\end{equation}
$$

The analytical solution to the fractional derivative of polynomials is given by the function
```python
def analytical_polynomial_fractionalderiv_function(
    alpha: float, pars: Arr[f64] | tuple[Arr[f64], Arr[i32]], time: Arr[f64]
) -> Arr[f64]: pass
```
This function has an additional parameter `alpha: float`, which is the order of the fractional derivative.


Two functions are provided to approximate the fractional derivative,
```python
def caputo_derivative_linear(carp: CaputoInitialize, S: Arr[f64], dt: Arr[64]): pass
def caputo_diffeq_linear(
    delta: float, carp: CaputoInitialize, S_HE: Arr[f64], S_VE: Arr[f64], dt: Arr[64]
): pass
```
Here, `alpha` is the order of the derivative, `S` is the array of data being differentiated, and `dt` is the array of time steps. For `caputo_diffeq_linear`, which is a full fractional differential equation, `delta` is the weight on the fractional part of the left-hand side, `S_VE` will be differentiated on the right-hand side, whereas `S_HE` will not.


In both functions, `carp` contains the pre-computed Prony series parameters for the Caputo fractional derivative. It is instantiated from the class
```python
@dataclass(slots=True, init=False)
class CaputoInitialize:
    Np: int
    beta0: float
    betas: Arr[f64]
    taus: Arr[f64]

    def __init__(self,
        alpha: float,
        Tf: float,
        Np: int = 9,
    ) -> None: pass
```
Here, `alpha` is order, `Tf` is the periodicity of the curve, and `Np` is the number of Prony terms.

### Plotting

The following plot functions, with self-documenting names, are provided:

```python

def benchmark_plot(
    x: Arr[f64] | Arr[i32], *data: Arr[f64], ...
) -> None: pass

def plot_scalar(
    *data: tuple[Arr[f64], Arr[f64]] | list[Arr[f64]], ...
) -> None: pass

def plot_stress_vs_strain_1D(
    *data: tuple[NDArray[f64], NDArray[f64]] | list[NDArray[f64]], ...
) -> None: pass

def plot_stress_vs_strain_2D(
    *data: tuple[NDArray[f64], NDArray[f64]] | list[NDArray[f64]], ...
) -> None: pass

def plot_strain_vs_time_1D(
    time: NDArray[f64], *data: NDArray[f64], ...
) -> None: pass

def plot_strain_vs_time_2D(
    time: NDArray[f64], *data: NDArray[f64], ...
) -> None: pass

def plot_stress_vs_time_1D(
    time: NDArray[f64], *data: NDArray[f64], ...
) -> None: pass

def plot_stress_vs_time_2D(
    time: NDArray[f64], *data: NDArray[f64], ...
) -> None: pass
```

All plot functions have the following options:
```python
x_lim: list[float] | None = None # X plot range
y_lim: list[float] | None = None # Y plot range
figsize: tuple[float, float] = (4, 3) # figure size by inches
dpi: int = 150 # DPI
x_label: str|list[str] = r"$E$"
y_label: str|list[str] = r"$S$ (kPa)"
curve_labels: list[str] | None = None # in order create a legend with labels for each data passed in
color: list[str] | None = ["k", "r", "b", "g", "c", "m"],
alpha: list[float] | None = None,
linestyle: list[str] | None = None, # "-" for lines, "none" for no lines
linewidth: list[float] | None = None,
marker: list[str] | None = None, # "o" for markers, "none" for no markers
markersize: int | float = 4,
markerskip: int | list[int] | float | list[float] | None = None,
markeredgewidth: float = 0.3,
legendlabelcols: int = 4,
fillstyle: str = "full", # "full", "none", "top", "bottom", "left", "right"
transparency: bool = False, # Transparency of the exported figure
fout: str | None = None,# "if None then figure will be displayed, if given a string, then attempt to save to str
**kwargs,# other kwargs will be pass to ax.plot
```

