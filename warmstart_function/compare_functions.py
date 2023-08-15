# Script to compare pretraining functions for warm-start
# Comparison on the basis of residual wrt C-equation
# Derivatives in residual by difference quotients

import torch
import matplotlib.pyplot as plt


x_0 = 0.0
x_end = 2e-2

n = 251

# System parameters
rho_0 = 1.130
u_0 = 0.3788
T_0 = 298
T_end = 2315.733
k = 0.02715  # Thermal conductivity
c_p = 1076.858  # Isobaric specific heat
T_act = 10064.951  # Activation temperature
A = 347850542  # Arrhenius pre-exponential factor

flame_location = 7e-3
flame_width_parameter = 1e3


def logistic(x, x0=flame_location, k=flame_width_parameter, L=1.0):
    return L / (1.0 + torch.exp(-k * (x - x0)))


def c_equation(y, x):
    # Evaluate c_eqn residual
    # Taken from PINN script

    y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]  # First derivative
    y_xx = torch.autograd.grad(y_x, x, grad_outputs=torch.ones_like(y_x), create_graph=True)[0]  # Second derivative

    return ((rho_0 * u_0 * y_x)
            - (k / c_p * y_xx)
            - (A * (1 - y)
               * (rho_0 * T_0 / (T_0 + y * (T_end - T_0)))
               * torch.exp(-T_act / (T_0 + y * (T_end - T_0)))
               )
            )


# Test
x = torch.linspace(x_0, x_end, n).requires_grad_(True)
y = logistic(x)

_, axs = plt.subplots(2, 1, figsize=(4, 8))
axs[0].plot(*[t.detach().numpy() for t in (x, y)])
axs[1].plot(*[t.detach().numpy() for t in (x, c_equation(y, x))])

plt.show()
