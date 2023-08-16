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
from datetime import datetime

import training
import network
import plotters
import logger

# CIP pool runs Python 3.9, TypeAlias in typing for >= 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

Tensor: TypeAlias = torch.Tensor

# --- Parameters --- #

# Training parameters
run_name = "bs_64_lr_1e-3_lrd_1e-8_nres_1000"
datafile = "./data/c_eqn_solution.csv"
batch_size = 64
learning_rate = 1e-3
lr_decay_exp = 1 - 1e-8  # Exponential learning rate decay
num_epochs = 10_000

loss_weights = {"data": 1.0, "residual": 1.0}

torch.manual_seed(7673345)
use_gpu = True

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

    y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]  # First derivative
    y_xx = torch.autograd.grad(y_x, x, grad_outputs=torch.ones_like(y_x), create_graph=True)[0]  # Second derivative

    return ((rho_0 * u_0 * y_x)
            - (k / c_p * y_xx)
            - (A * (1 - y)
               * (rho_0 * T_0 / (T_0 + y * (T_end - T_0)))
               * torch.exp(-T_act / (T_0 + y * (T_end - T_0)))
               )
            )


torch.set_default_dtype(torch.float64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
dataset = training.PINN_Dataset(datafile, ["x"], ["reaction_progress"])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Test grid to plot change of prediction over training
testgrid = torch.linspace(*extents_x, n_test_points).reshape(-1, 1).requires_grad_(True)
testgrid = testgrid.to(device)

# Set up network
network = network.FCN(1, 1, 64, 9).to(device)
optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=lr_decay_exp)

# Set up losses
loss_fns = {key: training.WeightedScalarLoss(torch.nn.MSELoss(), weight=value) for
            key, value in loss_weights.items()}
loss_history = {key: list() for key, _ in loss_weights.items()}  # Losses per iteration

# Sampler to randomly sample collocation points in each training iteration
residual_sampler = training.UniformRandomSampler(n_points=n_residual_points, extents=[extents_x])
residual_norm = {"l2": list(), "max": list()}  # Tracks norms of residual vector per test iteration

# Set up loggers to track loss and residual history as .csv files
logger_loss = logger.DictLogger(loss_history, f"runs/log_{run_name}_loss.csv", "Iteration")
logger_residual = logger.DictLogger(residual_norm, f"runs/log_{run_name}_residual.csv", "Epoch")


def train_iteration() -> None:
    # Train both data and residual losses concurrently
    for batch in dataloader:
        optimiser.zero_grad()
        x_data, y_data = [t.to(device) for t in batch]
        yh_data = network(x_data)  # Data prediction
        x_res = residual_sampler().to(device)
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
        loss_history["data"].append(loss_data.detach().item())
        loss_history["residual"].append(loss_res.detach().item())


def test_residuals(threshold) -> bool:
    # Returns bool: if converged, for training-while loop
    yh_test = network(testgrid)
    residual_test = c_equation(yh_test, testgrid)
    residual_norm["l2"].append(torch.linalg.norm(residual_test.detach()))
    residual_norm["max"].append(torch.linalg.norm(residual_test.detach(), ord=float('inf')))

    convergence_control = residual_norm["max"][-1]
    return convergence_control <= threshold


# Training loop
epoch = 0
converged = False
convergence_threshold = 1e-5
convergence_duration = 10
converged_ctr = 0

# # While loop
while not converged:
    print(f"Epoch: {epoch}")
    train_iteration()
    epoch += 1
    # After each training epoch, do a test iteration on testgrid
    converged_ctr = converged_ctr + 1 if test_residuals(convergence_threshold) else 0
    if converged_ctr >= convergence_duration:
        print(f"Training converged in {epoch} epochs")
        converged = True

    if not (epoch + 1) % 100:
        # Update loggers
        _ = [_logger.update() for _logger in (logger_loss, logger_residual)]


# # For loop
# for epoch in range(num_epochs):
#     print(f"Epoch: {epoch}")
#     train_iteration()
#     # After each training epoch, do a test iteration on testgrid
#     _ = test_residuals(0.0)

#     if not (epoch + 1) % 100:
#         # Update loggers
#         _ = [_logger.update() for _logger in (logger_loss, logger_residual)]


torch.save(network.state_dict(),
           f"model_state_dict_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")


# Plotting, unused?

        # # Plot losses
        # _, ax_loss = plt.subplots(1, 1, figsize=(4, 4))
        # for _label, _list in loss_history.items():
        #     ax_loss = plotters.semilogy_plot(ax_loss, _list, label=_label,
        #                                      ylabel="Loss", xlabel="Iteration", title="Loss curves")

        # # Plot test-iteration residual norms
        # _, ax_resnorms = plt.subplots(1, 1, figsize=(4, 4))
        # for _label, _list in residual_norm.items():
        #     ax_resnorms = plotters.semilogy_plot(ax_resnorms, _list, label=_label,
        #                                          ylabel="||r||", xlabel="Epoch",
        #                                          title="Residual norms, test iteration")

        # # Plot prediction on testgrid
        # _, ax_pred = plt.subplots(1, 1, figsize=(4, 4))
        # ax_pred = plotters.xy_plot(ax_pred, yh_test.detach().numpy(), testgrid.detach().numpy(),
        #                            ylabel="c", xlabel="x (m)", title="Reaction progress variable")

        # plt.show()
