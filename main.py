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
import plotters

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
loss_history = {key: list() for key, _ in loss_weights.items()}  # Losses per iteration

# Sampler to randomly sample collocation points in each training iteration
residual_sampler = training.UniformRandomSampler(n_points=n_residual_points, extents=[extents_x])
residual_norm = {"l2": list(), "max": list()}  # Tracks norms of residual vector per test iteration

# Training loop
for epoch in range(num_epochs):
    # Train both data and residual losses concurrently
    for batch in dataloader:
        optimiser.zero_grad()
        x_data, y_data = batch
        yh_data = network(x_data)  # Data prediction
        x_res = residual_sampler()
        yh_res = network(x_res)  # Prediction at collocation points
        residual = c_equation(yh_res, x_res)  # Residuals at collocation points
        loss_data = loss_fns["data"](yh_data, y_data)
        loss_res = loss_fns["residual"](residual, torch.zeros_like(residual))
        loss_total = loss_data + loss_res
        loss_total.backward()
        optimiser.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        # Save losses for plotting
        loss_history["data"].append(loss_data.detach.item())
        loss_history["residual"].append(loss_res.detach.item())
    # After each training epoch, do a test iteration on testgrid
    yh_test = network(testgrid)
    residual_test = c_equation(yh_test, testgrid)
    residual_norm["l2"].append(torch.linalg.norm(residual_test.detach()))
    residual_norm["max"].append(torch.linalg.norm(residual_test.detach()), ord=float('inf'))

    print(f"Epoch: {epoch}")

    if not (epoch + 1) % 100:
        # Plot losses
        _, ax_loss = plt.subplots(1, 1, figsize=(4, 4))
        for _label, _list in loss_history.items():
            ax_loss = plotters.semilogy_plot(ax_loss, _list, label=_label,
                                             ylabel="Loss", xlabel="Iteration", title="Loss curves")

        # Plot test-iteration residual norms
        _, ax_resnorms = plt.subplots(1, 1, figsize=(4, 4))
        for _label, _list in residual_norm.items():
            ax_resnorms = plotters.semilogy_plot(ax_resnorms, _list, label=_label,
                                                 ylabel="||r||", xlabel=f"Iterations * {num_epochs}",
                                                 title="Residual norms, test iteration")

        # Plot prediction on testgrid
        _, ax_pred = plt.subplots(1, 1, figsize=(4, 4))
        ax_pred = plotters.xy_plot(ax_pred, yh_test.detach().to_numpy(), testgrid.to_numpy(),
                                   ylabel="c", xlabel="x (m)", title="Reaction progress variable")

        plt.show()
