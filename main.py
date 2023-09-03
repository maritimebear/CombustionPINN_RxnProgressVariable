"""
Premixed methane-oxygen laminar flame modelling using reaction progress variable and physics-informed
neural networks.

Transport equation for reaction progress variable: Poinsot and Veynante, "Theoretical and Numerical
Combustion", Eq. (2.36).

Thermodynamic and chemical parameters obtained by using 1-step methane-oxygen reaction mechanism
from CERFACS in Cantera with the FreeFlame solver:
1S_CH4_MP1 from CERFACS: https://www.cerfacs.fr/cantera/mechanisms/meth.php#1S

CH4 + 2O2 -> CO2 + 2H2O

-----------------------

The network tends to diverge during training if it is directly trained on residuals. To prevent this,
training is carried out in two phases:

    1. Pre-training on a logistic function, due to the similarity of
    the logistic function to the reaction progress variable over the spatial domain.

    2. Main phase: training on data from the numerical solution of the governing transport equation,
    and the residuals of the governing equation. During this phase, the norm of the gradient is
    clipped to prevent divergence.

"""


import torch

import network
import physics
import utils
import training

from typing import TypeAlias
Tensor: TypeAlias = torch.Tensor

# --- Parameters --- #

# Common parameters
default_dtype = torch.float64
rng_seed = 7673345
extents_x = (0.0, 2e-2)  # Domain definition
n_testpoints = 101  # Number of points in testgrid

n_hidden_layers = 9
neurons_per_hidden_layer = 64

# Pretraining parameters
savename_pretrain = "Logistic_pretrained.pt"
flame_location = 7e-3
flame_width_parameter = 1e3
bs_pretrain = 64
lr_pretrain = 1e-3
n_epochs_pretrain = 600
loss_weights_pretrain = {"data": 1.0}
n_x_pretrain = 10_000  # Number of data points for pretraining


# Main training phase parameters
# Thermodynamic and chemical parameters
rho_0 = 1.130
u_0 = 0.3788
T_0 = 298
T_end = 2315.733
k = 0.02715  # Thermal conductivity
c_p = 1076.858  # Isobaric specific heat
T_act = 10064.951  # Activation temperature
A = 347850542  # Arrhenius pre-exponential factor

loadname_main = savename_pretrain
savename_main = "main_utils.pt"
datafile_ceqn = "./data/c_eqn_solution.csv"  # Numerical solution of the governing equation
bs_ceqn = 64
bs_residual = bs_ceqn  # Two dataloaders - one with solution data, another with collocation points for residuals
lr_main = 1e-6
lr_decay_exp = 1 - 1e-5  # Exponential learning-rate decay
n_epochs_main = 100_000
loss_weights_main = {"data": 1e2, "residual": 1e0}
n_x_residual = 20_000
grad_clip_limit = 1e-4

# --- end of parameters --- #


# Set up components from parameters
model = network.FCN(1, 1, neurons_per_hidden_layer, n_hidden_layers)
testgrid = torch.linspace(*extents_x, n_testpoints).reshape(-1, 1).requires_grad_(True)  # PyTorch complains about shape mismatch without reshape

# Pretraining
x_pretrain = utils.UniformRandomSampler(n_points=n_x_pretrain, extents=[extents_x], requires_grad=False)()
y_pretrain = utils.logistic_fn(x_pretrain, flame_location, flame_width_parameter)
ds_pretrain = utils.SampledDataset(x_pretrain, y_pretrain)
dl_pretrain = torch.utils.data.DataLoader(ds_pretrain, batch_size=bs_pretrain, shuffle=True, pin_memory=True)
optim_pretrain = torch.optim.Adam(model.parameters(), lr=lr_pretrain)

# Main training phase

# Dataloaders
collocation_pts = torch.Tensor(n_x_residual, 1).uniform_(*extents_x).requires_grad_(True)
residual_eqn = physics.ReactionProgress(rho_0, u_0, T_0, T_end, k, c_p, T_act, A)
# Numerical solution dataloader
ds_ceqn = utils.PINN_Dataset(datafile_ceqn, ["x"], ["reaction_progress"])
dl_ceqn = torch.utils.data.DataLoader(ds_ceqn, batch_size=bs_ceqn, shuffle=True, pin_memory=True)
# Residual dataloader
ds_residual = utils.SampledDataset(collocation_pts, torch.zeros_like(collocation_pts))
dl_residual = torch.utils.data.DataLoader(ds_residual, batch_size=bs_residual, shuffle=True, pin_memory=True)

# Optimiser
optim_main = torch.optim.Adam(model.parameters(), lr=lr_main)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim_main, gamma=lr_decay_exp)

# Training
torch.set_default_dtype(default_dtype)

# Perform pretraining
torch.manual_seed(rng_seed)
print("Pretraining")
training.pretrain(dl_pretrain,
               model,
               optim_pretrain,
               loss_weights_pretrain,
               num_epochs=n_epochs_pretrain,
               savename=savename_pretrain,
               testgrid=testgrid,
               test_interval=100
               )

# Main training phase
torch.manual_seed(rng_seed)
print("Main training phase")
training.warmstart([dl_ceqn, dl_residual],
                model,
                optim_main,
                loss_weights_main,
                n_epochs_main,
                savename_main,
                loadname_main,
                lr_scheduler,
                grad_clip_limit,
                testgrid,
                test_interval=100
                )
