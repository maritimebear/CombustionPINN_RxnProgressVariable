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
import physics
import trainer

from typing import TypeAlias
Tensor: TypeAlias = torch.Tensor

# --- Parameters --- #

# Training parameters
# saved_state_path = "Logistic_600_epochs_thinflame.pt"
saved_state_path = None
save_name = "test.pt"
# saved_state_path = None
datafile = "./data/c_eqn_solution.csv"
batchsize_data = 64
batchsize_residual = batchsize_data
# learning_rate = 1e-6
learning_rate = 1e-6
lr_decay_exp = 1 - 1e-8  # Exponential learning rate decay
n_epochs = 50_000

loss_weights = {"data": 1e0, "residual": 1e0}
# grad_clip_limit = 1e-6  # Maximum value for gradient clipping
grad_clip_limit = 1e-6

torch.manual_seed(7673345)

# Residuals and domain
n_residual_points = 10_000
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


def warmstart(num_epochs: int, loadfile: str = None):
    torch.set_default_dtype(torch.float64)

    # Load data
    data_ds = training.PINN_Dataset(datafile, ["x"], ["reaction_progress"])
    data_dl = torch.utils.data.DataLoader(data_ds, batch_size=batchsize_data, shuffle=True, pin_memory=True)

    # Collocation points (fixed over all training iterations)
    collocation_pts = torch.Tensor(n_residual_points, 1).uniform_(*extents_x).requires_grad_(True)
    residual_ds = training.SampledDataset(collocation_pts, torch.zeros_like(collocation_pts))
    residual_dl = torch.utils.data.DataLoader(residual_ds, batch_size=batchsize_residual, shuffle=True, pin_memory=True)
    residual_norm = {"l2": list(), "max": list()}  # Tracks norms of residual vector per test iteration

    # Residual equation
    c_equation = physics.ReactionProgress(rho_0, u_0, T_0, T_end, k, c_p, T_act, A)

    # Test grid to plot change of prediction over training
    testgrid = torch.linspace(*extents_x, n_test_points).reshape(-1, 1).requires_grad_(True)

    # Set up network
    model = network.FCN(1, 1, 64, 9)
    if loadfile is not None:
        print("Loading saved model state")
        model.load_state_dict(torch.load(loadfile))

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=lr_decay_exp)

    # Set up losses
    loss_fns = {key: training.WeightedScalarLoss(torch.nn.MSELoss(), weight=value) for
                key, value in loss_weights.items()}
    loss_history = {key: list() for key, _ in loss_weights.items()}  # Losses per iteration

    # Set up trainer object
    combined_trainer = trainer.Trainer_lists(dataloaders=[data_dl, residual_dl],
                                             model=model,
                                             optimiser=optimiser,
                                             loss_fns=[loss_fns["data"], loss_fns["residual"]],
                                             lr_scheduler=lr_scheduler,
                                             grad_norm_limit=grad_clip_limit)

    # Training loop
    for epoch in range(num_epochs):
        # Train both data and residual losses concurrently
        mean_losses = combined_trainer.train_epoch()
        _ = [loss_history[key].append(mean_losses[i]) for
             key, i in zip(("data", "residual"), (0, 1))]

        # After each training epoch, do a test iteration on testgrid
        yh_test = model(testgrid)
        residual_test = c_equation(yh_test, testgrid)
        residual_norm["l2"].append(torch.linalg.norm(residual_test.detach()))
        residual_norm["max"].append(torch.linalg.norm(residual_test.detach(), ord=float('inf')))

        print(f"Epoch: {epoch}")

        if not (epoch + 1) % 100:
            # Plot losses
            _, ax_loss = plt.subplots(1, 1, figsize=(4, 4))
            for _label, _list in loss_history.items():
                ax_loss = plotters.semilogy_plot(ax_loss, _list, label=_label,
                                                 ylabel="Loss", xlabel="Epoch", title="Mean Loss per Epoch")

            # Plot test-iteration residual norms
            _, ax_resnorms = plt.subplots(1, 1, figsize=(4, 4))
            for _label, _list in residual_norm.items():
                ax_resnorms = plotters.semilogy_plot(ax_resnorms, _list, label=_label,
                                                     ylabel="||r||", xlabel="Epoch",
                                                     title="Residual norms, test iteration")

            # Plot prediction on testgrid
            _, ax_pred = plt.subplots(1, 1, figsize=(4, 4))
            ax_pred = plotters.xy_plot(ax_pred, yh_test.detach().numpy(), testgrid.detach().numpy(),
                                       ylabel="c", xlabel="x (m)", title="Reaction progress variable")

            plt.show()

            # Remove after testing
            torch.save(model.state_dict(), save_name)

    torch.save(model.state_dict(), save_name)


if __name__ == "__main__":
    warmstart(n_epochs, saved_state_path)
