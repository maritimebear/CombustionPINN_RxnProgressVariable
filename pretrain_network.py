# Functions to pre-train network
import torch
from datetime import datetime
import matplotlib.pyplot as plt

import training
import network
import plotters


def logistic_fn(x, x0, k, L=1.0) -> None:
    # Function to be pretrained against
    return L / (1.0 + torch.exp(-k * (x - x0)))


# --- Parameters --- #

save_path = "Logistic_600_epochs.pt"
flame_location = 7e-3
flame_width_parameter = 1e3

# Training parameters
pretrain_fn = logistic_fn
pretrain_fn_args = [flame_location, flame_width_parameter]

batch_size = 64
learning_rate = 1e-5
n_epochs = 600

loss_weights = {"data": 1.0}

torch.manual_seed(7673345)

# Residuals and domain
n_data_points = 10_000
extents_x = (0.0, 2e-2)

# Test step and error calculation
# n_test_points = 101
n_test_points = n_data_points

# --- end of parameters --- #

def pretrain(num_epochs: int, savename: str) -> None:
    torch.set_default_dtype(torch.float64)

    # Create data for pretraining
    sampler = training.UniformRandomSampler(n_points=n_data_points, extents=[extents_x], requires_grad=False)
    data_x = sampler()
    data_y = pretrain_fn(data_x, *pretrain_fn_args)

    dataset = training.SampledDataset(data_x, data_y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Test grid to plot change of prediction over training
    testgrid = torch.linspace(*extents_x, n_test_points).reshape(-1, 1)

    # Set up network
    model = network.FCN(1, 1, 64, 9)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=lr_decay_exp)

    # Set up losses
    loss_fns = {key: training.WeightedScalarLoss(torch.nn.MSELoss(), weight=value) for
                key, value in loss_weights.items()}
    loss_history = {key: list() for key, _ in loss_weights.items()}  # Losses per iteration

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")

        for batch in dataloader:
            optimiser.zero_grad()
            x, y = batch
            y_h, loss = training.chain_callables(x, [y], [model, loss_fns["data"]])
            loss.backward()
            optimiser.step()
            loss_history["data"].append(loss.detach().item())

        # Test step after each epoch
        yh_test = model(testgrid)

        if not (epoch + 1) % 100:
            # Plot losses
            _, ax_loss = plt.subplots(1, 1, figsize=(4, 4))
            for _label, _list in loss_history.items():
                ax_loss = plotters.semilogy_plot(ax_loss, _list, label=_label,
                                                 ylabel="Loss", xlabel="Iteration", title="Loss curves")

            # Plot prediction on testgrid
            _, ax_pred = plt.subplots(1, 1, figsize=(4, 4))
            ax_pred = plotters.xy_plot(ax_pred, yh_test.detach().numpy(), testgrid.detach().numpy(),
                                       ylabel="y", xlabel="x (m)", title="Prediction")

            plt.show()

    # Save model after training
    torch.save(model.state_dict(), savename)


if __name__ == "__main__":
    # pretrain(n_epochs, f"model_state_dict_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    pretrain(n_epochs, save_path)
