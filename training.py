import torch
import matplotlib.pyplot as plt

import utils
import plotters

from typing import TypeAlias
Tensor: TypeAlias = torch.Tensor


def pretrain(dataloader,
             model,
             optim,
             loss_weights,
             num_epochs: int,
             savename: str,
             testgrid: Tensor = None,
             test_interval: int = 0,
             ) -> None:

    # Set up losses
    loss_fns = {key: utils.WeightedScalarLoss(torch.nn.MSELoss(), weight=value) for
                key, value in loss_weights.items()}
    loss_history = {key: list() for key, _ in loss_weights.items()}  # Losses per iteration

    # Set up trainer
    trainer_data = utils.EpochTrainer(dataloaders=[dataloader],
                                      model=model,
                                      optimiser=optim,
                                      callables=[lambda x, y_h, y: loss_fns["data"](y_h, y)]
                                      )

    # Training loop
    for epoch in range(num_epochs):
        mean_losses = trainer_data.train_epoch()
        loss_history["data"].append(mean_losses[0])

        if test_interval > 0 and not ((epoch + 1) % test_interval):
            print(f"Epoch: {epoch}")
            # Test step after each epoch
            yh_test = model(testgrid)
            # Plot losses
            _, ax_loss = plt.subplots(1, 1, figsize=(8, 8))
            for _label, _list in loss_history.items():
                ax_loss = plotters.semilogy_plot(ax_loss, _list, label=_label,
                                                 ylabel="Loss", xlabel="Iteration", title="Loss curves")
            # Plot prediction on testgrid
            _, ax_pred = plt.subplots(1, 1, figsize=(8, 8))
            ax_pred = plotters.xy_plot(ax_pred, yh_test.detach().numpy(), testgrid.detach().numpy(),
                                       ylabel="y", xlabel="x (m)", title="Prediction")
            plt.show()

    # Save model after training
    torch.save(model.state_dict(), savename)


def warmstart(dataloaders,
              model,
              optim,
              loss_weights,
              collocation_pts: Tensor,
              residual_eqn: Callable[[Tensor, Tensor], Tensor],
              num_epochs: int,
              savename: str,
              loadname: str = None,
              lr_scheduler = None,
              grad_clip_limit = None,
              testgrid: Tensor = None,
              test_interval: int: None
              ) -> None:

    # Load network parameters if specified
    if loadfile is not None:
        print("Loading saved model state")
        model.load_state_dict(torch.load(loadfile))

    # Set up losses and residual tracking
    loss_fns = {key: utils.WeightedScalarLoss(torch.nn.MSELoss(), weight=value) for
                key, value in loss_weights.items()}
    loss_history = {key: list() for key, _ in loss_weights.items()}  # Losses per iteration
    residual_norm = {"l2": list(), "max": list()}  # Tracks norms of residual vector per test iteration

    # Set up trainer object
    combined_trainer = utils.EpochTrainer(dataloaders=dataloaders,
                                          model=model,
                                          optimiser=optim,
                                          callables=[lambda x, y_h, y: loss_fns["data"](y_h, y),
                                                     lambda x, y_h, y: loss_fns["residual"](residual_eqn(y_h, x), y)],
                                          lr_scheduler=lr_scheduler,
                                          grad_norm_limit=grad_clip_limit
                                          )

    # Training loop
    for epoch in range(num_epochs):
        # Train both data and residual losses concurrently
        mean_losses = combined_trainer.train_epoch()
        _ = [loss_history[key].append(mean_losses[i]) for
             key, i in zip(("data", "residual"), (0, 1))]

        # After each training epoch, do a test iteration on testgrid
        # Evaluating residuals after each training epoch to see if residuals explode
        # during training
        yh_test = model(testgrid)
        residual_test = residual_eqn(yh_test, testgrid)
        residual_norm["l2"].append(torch.linalg.norm(residual_test.detach()))
        residual_norm["max"].append(torch.linalg.norm(residual_test.detach(), ord=float('inf')))

        if test_interval > 0 and not ((epoch + 1) % test_interval):
            print(f"Epoch: {epoch}")
            # Plot losses
            _, ax_loss = plt.subplots(1, 1, figsize=(8, 8))
            for _label, _list in loss_history.items():
                ax_loss = plotters.semilogy_plot(ax_loss, _list, label=_label,
                                                 ylabel="Loss", xlabel="Epoch", title="Mean Loss per Epoch")

            # Plot test-iteration residual norms
            _, ax_resnorms = plt.subplots(1, 1, figsize=(8, 8))
            for _label, _list in residual_norm.items():
                ax_resnorms = plotters.semilogy_plot(ax_resnorms, _list, label=_label,
                                                     ylabel="||r||", xlabel="Epoch",
                                                     title="Residual norms, test iteration")

            # Plot prediction on testgrid
            _, ax_pred = plt.subplots(1, 1, figsize=(8, 8))
            ax_pred = plotters.xy_plot(ax_pred, yh_test.detach().numpy(), testgrid.detach().numpy(),
                                       ylabel="c", xlabel="x (m)", title="Reaction progress variable")

            plt.show()

    torch.save(model.state_dict(), save_name)
