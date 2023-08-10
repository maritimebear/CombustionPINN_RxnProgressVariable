"""
Premixed laminar flame  modelling using reaction progress variable

Governing equation from Eq. (2.36), Theoretical and Numerical Combustion, Poinsot and Veynante

Using 1-step methane-oxygen reaction mechanism as reference:
1S_CH4_MP1 from CERFACS: https://www.cerfacs.fr/cantera/mechanisms/meth.php#1S

CH4 + 2O2 -> CO2 + 2H2O

-----------------------

Reference data was obtained by using this mechanism in Cantera with the system specified below:

Inlet state:
------------
T1 = 298K
P = 101325 Pa, isobaric system
rho1 = 1.130 kg/m^3 (inlet density)
u1 = 0.393 m/s (inlet velocity, == laminar flame speed)

Gas properties at inlet state, assumed to be constant:
------------
*** Units assumed to be SI, not specified in Cantera results ***

Thermal conductivity = 0.027 W/m-K
Isobaric specific heat capacity = 1076.858 J/kg-K

Arrhenius coefficients:
------------

Activation energy = 83680 J/mol
Pre-exponential factor = 347850542.619 (from Cantera, units unspecified)
Temperature exponent = 0
"""

import network

import torch
import matplotlib.pyplot as plt
import numpy as np

from typing import Sequence, TypeAlias, Union
Tensor: TypeAlias = Union[np.ndarray, torch.Tensor]

torch.manual_seed(7673345)
torch.set_default_dtype(torch.float64)  # Double precision required for L-BFGS? https://stackoverflow.com/questions/73019486/torch-optim-lbfgs-does-not-change-parameters
plt.ioff()
R = 8.314  # Universal Gas Constant, J/K-mol

# --- Parameters --- #
n_collocation_points = 1_000
n_data_points = 10

# Learning rate and decay control
lr_Adam = 1e-3
lr_decay_exp = 1 - 1e-4  # Exponent for exponential learning rate decay

# Domain definition
extents_x = (0.0, 1e-1)

# Loss weights
weight_dataloss = 0.0
weight_residualloss = 1.0

# Training loop control
n_epochs = 10_000  # Use with for-loop
converged = False  # Use with while-loop with convergence control
convergence_threshold = 1e-1
convergence_sustain_duration = 10
n_converged = 0


