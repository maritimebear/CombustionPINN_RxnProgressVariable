# Script to generate plots for presentation

import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import numpy as np
import torch

import seaborn as sns
sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("Set2"))

import plotters
import network
import utils

torch.set_default_dtype(torch.float64)
plt.rcParams.update({"font.size": 16})

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

# Reference data -- C-equation solution
ref_data = pd.read_csv("./data/c_eqn_solution.csv")
x = torch.from_numpy(ref_data["x"].to_numpy()).reshape(-1, 1)
y = torch.from_numpy(ref_data["reaction_progress"].to_numpy()).reshape(-1, 1)
y_pretrain = utils.logistic_fn(x.detach(), 7e-3, 1e3)

_, ax_ref = plt.subplots(1, 1, figsize=(8, 8))
ax_ref.xaxis.set_major_formatter(formatter)
ax_ref = plotters.xy_plot(ax_ref, y.detach(), x.detach(),
                            ylabel="c", xlabel="x (m)", title="Reaction progress variable")
# p_ref = plt.plot(ref_data["x"], ref_data["reaction_progress"])

# Pretrain vs reference data
_, ax_logistic = plt.subplots(1, 1, figsize=(8, 8))
ax_logistic.xaxis.set_major_formatter(formatter)
ax_logistic = plotters.xy_plot(ax_logistic, y.detach(), x.detach(), label="Ground truth",
                            ylabel="c", xlabel="x (m)", title="Pretraining function")
ax_logistic = plotters.xy_plot(ax_logistic, y_pretrain.detach(), x.detach(), label="Logistic function",
                            ylabel="c", xlabel="x (m)", title="Pretraining function")

# Network prediction vs reference
loadname = "trained_network.pt"
model = network.model = network.FCN(1, 1, 64, 9)
model.load_state_dict(torch.load(loadname))
y_hat = model(x)
error = y - y_hat

_, ax_pred = plt.subplots(1, 1, figsize=(8, 8))
ax_pred.xaxis.set_major_formatter(formatter)
ax_pred = plotters.xy_plot(ax_pred, y.detach(), x.detach(), label="Ground truth", ylabel="c",
                           xlabel="x (m)", title="Reaction progress variable")
ax_pred = plotters.xy_plot(ax_pred, y_hat.detach(), x.detach(), label="Prediction", ylabel="c",
                           xlabel="x (m)", title="Reaction progress variable")


_, ax_error = plt.subplots(1, 1, figsize=(8, 8))
ax_error.xaxis.set_major_formatter(formatter)
ax_error = plotters.xy_plot(ax_error, error.detach(), x.detach(), xlabel="x (m)",
                            ylabel="Error", title="Error in prediction")

plt.show()