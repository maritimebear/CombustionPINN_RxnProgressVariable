import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd

# Numerical parameters
x_0 = 0.0
x_end = 2e-2

c_0 = 0.0
c_end = 1.0

n = 251
x = np.linspace(x_0, x_end, n)
h = (x_end - x_0) / (n - 1)

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


def c_xx(x, c, c_x):
    return ((rho_0 * u_0 * c_x)
            - (A * (1 - c)
               * (rho_0 * T_0 / (T_0 + c * (T_end - T_0)))
               * np.exp(-T_act / (T_0 + c * (T_end - T_0)))
               )
            ) * (c_p / k)

# def c_xx(x, c, c_x):
#     return ((rho_0 * u_0 * c_x) - (A * (1 - c) * (rho_0 * T_0 / (T_0 + c * (T_end - T_0))) * np.exp(-T_act / (T_0 + c * (T_end - T_0)))))



def residual(c):
    # residual(solution) == 0
    res = np.zeros(c.shape)
    # BCs
    res[0] = c[0] - c_0
    res[-1] = c[-1] - c_end

    c_xx_h  = (c[0:-2] - 2 * c[1:-1] + c[2:]) / (h**2)  # Approx. c_xx by central difference
    c_x_h = (c[2:] - c[0:-2]) / (2 * h)  # c_x by central differences

    res[1:-1] = c_xx_h - c_xx(x, c[1:-1], c_x_h)

    return res


# c_initial = c_0 + (c_end - c_0) / (x_end - x_0) * x
c_initial = np.ones_like(x) * (x >= flame_location)
# plt.plot(c_initial)
# plt.show()

solution = scipy.optimize.fsolve(residual, c_initial)

pd.DataFrame(np.c_[x, solution], columns=["x", "reaction_progress"]).to_csv("c_eqn_solution.csv", index=False)

plt.plot(x, solution)
plt.show()
