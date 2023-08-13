"""
Premixed laminar flame  modelling using reaction progress variable

Governing equation from Eq. (2.36), Theoretical and Numerical Combustion, Poinsot and Veynante

Using 1-step methane-oxygen reaction mechanism as reference:
1S_CH4_MP1 from CERFACS: https://www.cerfacs.fr/cantera/mechanisms/meth.php#1S

CH4 + 2O2 -> CO2 + 2H2O

-----------------------

Reference data was obtained by using this mechanism in Cantera with the FreeFlame solver.
"""


import torch
import matplotlib.pyplot as plt

import training
import network

from typing import TypeAlias
Tensor: TypeAlias = torch.Tensor

# --- Parameters --- #

# Training parameters
datafile = "./data/c_eqn_solution.csv"
batch_size = 64
learning_rate = 1e-4
lr_decay_exp = 1 - 1e-4  # Exponential learning rate decay
num_epochs = 10_000

loss_weights = {"data": 1.0, "residual": 1.0}

torch.manual_seed(7673345)

# Residuals and domain
n_residual_points = 1000
extents_x = (0.0, 2e-2)

# Test step and error calculation
n_test_points = 101

# Thermodynamic and chemical parameters
rho_0 = 1.130
u_0 = 0.3788
T_0 = 298
T_end = 2315.733
k = 0.02715  # Thermal conductivity
c_p = 1076.858  # Isobaric specific heat
T_act = 10064.951  # Activation temperature
A = 347850542  # Arrhenius pre-exponential factor

# --- end of parameters --- #


def c_equation(y: Tensor, x: Tensor) -> Tensor:
    # Calculates residual D(y; x)
    # Residual == 0 when y(x) satisfies equation system D(y; x)

    y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graphs=True)[0]  # First derivative
    y_xx = torch.autograd.grad(y_x, x, grad_outputs=torch.ones_like(y_x), create_graphs=True)[0]  # Second derivative

    return ((rho_0 * u_0 * y_x)
            - (k / c_p * y_xx)
            - (A * (1 - y)
               * (rho_0 * T_0 / (T_0 + y * (T_end - T_0)))
               * torch.exp(-T_act / (T_0 + y * (T_end - T_0)))
               )
            )


torch.set_default_dtype(torch.float64)

# Load data
dataset = training.PINN_Dataset(datafile, ["x"], ["reaction_progress"])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Test grid to plot change of prediction over training
testgrid = torch.linspace(*extents_x, n_test_points).reshape(-1, 1)

# Set up network
network = network.FCN(1, 1, 64, 4)
optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=lr_decay_exp)

# Set up losses
loss_fns = {key: training.WeightedScalarLoss(torch.nn.MSELoss(), weight=value) for
            key, value in loss_weights.items()}


# Training loop
loss_history = {key: list() for key, _ in loss_weights.item()}  # Mean losses per epoch

for epoch in range(num_epochs):
    losses_epoch.append(includes.train(dataloader, network, loss, optimiser))

    y_test = network(testgrid)
    print(f"Epoch: {epoch}, Epoch loss: {losses_epoch[-1]}")

    if not (epoch + 1) % 10:
        fig, axs = plt.subplots(2, 1, figsize=(4,8))
        axs[0].semilogy(losses_epoch)
        axs[1].plot(testgrid, y_test.detach().numpy())

        plt.show()

