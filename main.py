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
import loss
import sampler
import dataset
import trainers
import physics
import plot_logger
import plotters

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Sequence, TypeAlias, Union
Tensor: TypeAlias = Union[np.ndarray, torch.Tensor]

torch.manual_seed(7673345)
torch.set_default_dtype(torch.float64)  # Double precision required for L-BFGS? https://stackoverflow.com/questions/73019486/torch-optim-lbfgs-does-not-change-parameters
plt.ioff()
R = 8.314  # Universal Gas Constant, J/K-mol


# --- Parameters --- #

cantera_data = "./freely_propagating.csv"
reaction_progress_data = "./rxn_progress_data.csv"
# reaction_progress_data = "./test.csv"

activation_energy = 83680
A_arrhenius = 347850542.619  # Arrhenius pre-exponential factor

n_residual_points = 1000
batch_size = 64  # Number of Cantera data points per batch

# Learning rate and decay control
lr_Adam = 1e-4
lr_decay_exp = 1 - 1e-4  # Exponent for exponential learning rate decay

# Domain definition
extents_x = (0.0, 2e-2)
# extents_x = (0.0, 1.0)

# Loss weights
weight_dataloss = 1.0
weight_residualloss = 1.0

# Training loop control
n_epochs = 10_000  # Use with for-loop
converged = False  # Use with while-loop with convergence control
convergence_threshold = 1e-5
convergence_sustain_duration = 10
n_converged = 0

testgrid_n_points = 100

# --- end of parameters --- #


# Read parameters from Cantera data
cantera_df = pd.read_csv(cantera_data)  # Used for C equation parameters
rho_in, u_in, T_in, thermal_cond, spec_heat = [cantera_df[key].iloc[0] for
                                               key in ["density", "velocity", "T", "thermal_conductivity", "cp"]]
T_out = cantera_df["T"].iloc[-1]
T_act = activation_energy / R

c_reference = pd.read_csv(reaction_progress_data)["reaction_progress"].to_numpy()  # Used to calculate error

# Grid for plotting residuals and fields during testing
testgrid = torch.linspace(*extents_x, testgrid_n_points, requires_grad=True).reshape(-1,1)
# testgrid = torch.from_numpy(cantera_df["grid"].to_numpy()).requires_grad_(True).reshape(-1, 1)

# Set up model
model = network.FCN(1,  # inputs: x
                    1,  # outputs: c
                    64,  # number of neurons per hidden layer
                    4)  # number of hidden layers

# Set up optimiser
optimiser_Adam = torch.optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler_Adam = torch.optim.lr_scheduler.ExponentialLR(optimiser_Adam, gamma=lr_decay_exp)

# Set up losses
lossfn_data = loss.WeightedScalarLoss(torch.nn.MSELoss(), weight=weight_dataloss)
lossfn_residual = loss.WeightedScalarLoss(torch.nn.MSELoss(), weight=weight_residualloss)

# Set up trainers
# Data trainer
ds = dataset.PINN_Dataset(reaction_progress_data, ["x"], ["reaction_progress"])
dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
trainer_data = trainers.DataTrainer(model, loss_fn=lossfn_data)

# Residual trainer
sampler_residual = sampler.UniformRandomSampler(n_points=n_residual_points, extents=[extents_x])
residual_fn = physics.CEquation(rho_in, u_in, T_in, T_out, thermal_cond, spec_heat, T_act, A_arrhenius)
trainer_residual = trainers.ResidualTrainer(sampler_residual, model, residual_fn, lossfn_residual)

# Setup plot-loggers for loss and error curves
# Track losses and norms of error and residual through dicts
losses_dict = {key: list() for key in ("data", "residual", "total")}
errors_dict = {key: list() for key in ("l2", "max")}
residuals_dict = {key: list() for key in ("l2", "max")}

logger_loss = plot_logger.Plot_and_Log_Scalar("losses", losses_dict,
                                              plot_xlabel="Iteration", plot_ylabel="Loss", plot_title="Loss curves")
logger_error = plot_logger.Plot_and_Log_Scalar("error", errors_dict,
                                               plot_xlabel="Epoch", plot_ylabel="||error||", plot_title="Error at test points")
logger_residual = plot_logger.Plot_and_Log_Scalar("residuals", residuals_dict,
                                                  plot_xlabel="Epoch", plot_ylabel="||residual||", plot_title="Residuals at test points")


def train_iteration(optimiser, step: bool, lr_scheduler=None) -> torch.Tensor:
    # Can be used as closure function for L-BFGS
    # Returns total loss (scalar)
    # step: whether or not to optimiser.step()
    for batch in dataloader:
        optimiser.zero_grad()
        # Data loss
        x, y = [tensor.to(torch.get_default_dtype()) for tensor in batch]  # Network weights have dtype torch.float32
        loss_data = trainer_data(x, y)
        # Residual loss
        loss_residual = trainer_residual()
        # Total loss
        loss_total = loss_data + loss_residual
        loss_total.backward()
        if step:
            optimiser.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Append losses to corresponding lists in dict
        for key, _loss in zip(["data", "residual", "total"],
                              [loss_data, loss_residual, loss_total]):
            losses_dict[key].append(_loss.detach())
        logger_loss.update_log()

    return loss_total  # For future L-BFGS compatibility


def test() -> tuple[Sequence[Tensor], float]:
    # Calculate error and residual fields over a fixed set of points in the domain
    # Returns a tuple of resulting tensors and a metric of the error/residual (a norm or a mean),
    # to determine solution convergence

    # Calculate errors, update error logger
    error = c_reference - model(testgrid).detach().numpy()
    errors_dict["l2"].append(np.linalg.norm(error.flatten()))
    errors_dict["max"].append(np.linalg.norm(error.flatten(), ord=np.inf))
    logger_error.update_log()

    # Residuals
    u_h = model(testgrid)  # testgrid from main namespace
    residuals = residual_fn(u_h, testgrid)
    residuals_dict["l2"].append(np.linalg.norm(residuals.detach().numpy()))
    residuals_dict["max"].append(np.linalg.norm(residuals.detach().numpy(), ord=np.inf))
    logger_residual.update_log()

    # convergence_control = residuals_dict["max"][-1]  # Using inf-norm of residuals for convergence control
    convergence_control = errors_dict["max"][-1]

    return ([u_h, error, residuals], convergence_control)


def plot(u_h, error, residuals) -> None:
    _ = [logger.update_plot() for logger in (logger_loss, logger_error, logger_residual)]
    # Plot test grid
    fig_pred = plt.figure(figsize=(8, 8))
    ax_pred = fig_pred.add_subplot(1, 1, 1)
    ax_pred = plotters.xy_plot(ax=ax_pred, y=u_h.detach().numpy(), x=testgrid.detach().numpy(), label=None, ylabel="c", xlabel="x", title="Reaction progress variable -- prediction")

    plt.show()


# for-loop to train for a specified number of epochs
# epochs wrt dataset size and batch size of ground truth data
for i in range(n_epochs):
    print(f"Epoch: {i}")
    _ = train_iteration(optimiser_Adam, step=True, lr_scheduler=lr_scheduler_Adam)  # Discard return value, losses appended to lists
    test_tensors, _ = test()  # Discard convergence control in for-loop
    if not (i+1) % 100:
        plot(*test_tensors)

# # while-loop to train until converged wrt convergence control returned by test()
# epoch_ctr = 0
# while not converged:
#     _ = train_iteration(optimiser_Adam, step=True, lr_scheduler=None)  # Discard return value, losses appended to lists
#     test_tensors, convergence_control = test()

#     epoch_ctr += 1
#     if not epoch_ctr % 10:
#         plot(*test_tensors)

#     if convergence_control <= convergence_threshold:  # threshold defined in main namespace
#         n_converged += 1
#         print(f"Epoch: {epoch_ctr}\t" +
#               f"Convergence control: {convergence_control}\t" +
#               f"Threshold: {convergence_threshold}\t" +
#               f"Remaining: {convergence_sustain_duration - n_converged}")
#     else:
#         n_converged = 0
#         print(f"Epoch: {epoch_ctr}\t" +
#               f"Convergence control: {convergence_control}\t" +
#               f"Threshold: {convergence_threshold}")

#     if n_converged >= convergence_sustain_duration:
#         converged = True
#         print(f"Training converged in {epoch_ctr} epochs")

# Plot final results
final_test, _ = test()
plot(*final_test)
