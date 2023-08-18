import torch
from dataclasses import dataclass

from typing import TypeAlias
Tensor: TypeAlias = torch.Tensor


@dataclass(slots=True)
class ReactionProgress():
    # Class to store equation parameters, __call__(y, x) returns residual
    rho_0: float  # Inlet density
    u_0: float  # Inlet velocity
    T_0: float  # Inlet temperature
    T_end: float  # Outlet temperature (assumed to be the adiabatic flame temperature)
    k: float  # Thermal conductivity (assumed constant)
    c_p: float  # Specific heat capacity (assumed constant)
    T_act: float  # Arrhenius activation temperature
    A: float  # Arrhenius pre-exponential factor

    def __call__(self, y: Tensor, x: Tensor) -> Tensor:
        # Calculates residual D(y; x)
        # Residual == 0 when y(x) satisfies equation system D(y; x)

        # Calling torch.zeros_like() each time to accommodate different input sizes during training and testing
        y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]  # First derivative
        y_xx = torch.autograd.grad(y_x, x, grad_outputs=torch.ones_like(y_x), create_graph=True)[0]  # Second derivative

        T: Tensor = self.T_0 + y * (self.T_end - self.T_0)  # Local temperature

        return ((self.rho_0 * self.u_0 * y_x)
                - (self.k / self.c_p * y_xx)
                - (self.A * (1.0 - y)
                   * (self.rho_0 * self.T_0 / T)
                   * torch.exp(-self.T_act / T)
                   )
                )
