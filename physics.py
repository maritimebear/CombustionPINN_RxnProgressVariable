import torch

from typing import TypeAlias, Callable

Tensor: TypeAlias = torch.Tensor


def _grad(f: Tensor, x: Tensor) -> Tensor:
    return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]


def CEquation(density_inlet: float,
              velocity_inlet: float,
              temp_inlet: float,
              temp_outlet: float,
              thermal_conductivity: float,
              spec_heat_capacity: float,
              activation_temp: float,
              pre_exp_factor: float) -> Callable[[Tensor, Tensor], Tensor]:

    # Returns function with parameters captured

    def residual(prediction: Tensor, domain: Tensor) -> Tensor:
        # Calculates residual
        c_x = _grad(prediction, domain)  # First derivative
        c_xx = _grad(c_x, domain)  # Second derivative
        # Reused variables
        temperature: Tensor = temp_inlet + prediction * (temp_outlet - temp_inlet)  # Definition of progress variable
        density: Tensor = density_inlet * temp_inlet / temperature  # From ideal gas law
        return ((density_inlet * velocity_inlet * c_x)
                - (thermal_conductivity / spec_heat_capacity * c_xx)
                - (pre_exp_factor * density * (1 - prediction) * torch.exp(-activation_temp / temperature))
                )

    return residual
