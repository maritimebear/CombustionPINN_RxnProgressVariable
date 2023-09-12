#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Freely propagating 1-D premixed
Methane/air, global 1-step mechanism from CERFACS
CH4_MP1 mechanism: CH4 + 2 O2 => CO2 + 2 H2O
https://www.cerfacs.fr/cantera/mechanisms/meth.php#1S
"""

import cantera as ct
import pandas as pd
import matplotlib.pyplot as plt
import Utility as util

import numpy as np

# plt.rcParams['text.usetex'] = True # LaTeX in plots

## Simulation

# Parameters
# Standard conditions: 300K, 101325 Pa, phi = 1
# CH4_MP1 mechanism: CH4 + 2 O2 => CO2 + 2 H2O
P = 101325
T_in = 298
reactants = {'CH4': 1.0, 'O2': 2.0, 'N2': 3.762*2}
width = 2e-2 # From CERFACS reference result
loglevel = 1

# Initialise gas object at upstream conditions
gas = ct.Solution('ch4.yaml') # ch4.yaml provides 1-step mechanism
gas.TPX = T_in, P, reactants

# Initialise Flame object with gas
# f = ct.FreeFlame(gas, width=width)
f = ct.FreeFlame(gas, grid=list(np.linspace(0, width, 1001)))
# f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12) # Refinement criteria from Cantera example for freely-propagating flame

# Start with mixture-averaged transport model
f.transport_model = 'Mix'
# f.solve(loglevel=loglevel, auto=True) 
f.solve(loglevel=loglevel, refine_grid=False)

# Save after Flame::solve() to restore later if needed
f.save('adiabatic_flame.yaml', 'mix',
       'solution with mixture-averaged transport')

# # # Continue with multicomponent diffusion
# f.transport_model = 'Multi'
# f.solve(loglevel)  # don't use 'auto' on subsequent solves
# f.save('adiabatic_flame.yaml', 'multi',
#         'solution with multicomponent transport')

## Write results to .csv and plot results, comparisons with reference data

# Add additional fields that are not included by Cantera to the result
result_df = f.to_pandas(species='Y')
omega = f.net_production_rates
util.add_species_fields(result_df, f.X.T, "X", gas.species_names) # Mole fractions, X.T to get each species columnwise
util.add_species_fields(result_df, omega.T, "omega", gas.species_names) # Molar production rates

# Adding thermal conductivity and specific heat capacity for PINN
for key in ("thermal_conductivity", "cp"):
    result_df[key] = f.__getattribute__(key)

result_df.to_csv('freely_propagating.csv', index=False) # Write results to csv via pandas (instead of Cantera) to include Y and omega columns

# Plot results, results vs reference
reference_df = pd.read_csv('./Solcan2av-CM1_P-101325-T-300.0-Phi-1.0.csv', skiprows=1) # Skip first row -- file header, not actual data
# Variables used later on in plotting for convenience
result_x = result_df['grid']
reference_x = reference_df['x_axis']
xlabel = "x (m)"

with plt.ioff(): # Interactive plotting off, display all plots at the end
    # Temperature, density, flamespeed
    _, axs = plt.subplots(3,1, figsize=(9, 12), layout='tight')
    util.comparison_plot(axs[0], result_x, result_df['T'], reference_x, reference_df['Temperature'], "Temperature vs x", xlabel, "T (K)")
    util.comparison_plot(axs[1], result_x, result_df['density'], reference_x, reference_df['rho'], "Density vs x", xlabel, "rho (kg/m^3)")
    util.comparison_plot(axs[2], result_x, result_df['velocity'], reference_x, reference_df['u'], "Velocity vs x", xlabel, "u (m/s)")
    
    # Mass fractions
    fig, axs = plt.subplots(5, 1, figsize=(9, 15), layout='tight')
    [util.comparison_plot(ax, result_x, result_df[f'Y_{species}'], reference_x, reference_df[f'{species}'], title=f'{species}')
         for ax, species in zip(axs, gas.species_names)]
    axs[-1].set_xlabel(xlabel)
    fig.suptitle("Mass fractions")
    
    # Molar rates of production
    fig, axs = plt.subplots(5, 1, figsize=(9, 15), layout='tight')
    [ax.plot(result_x, result_df[f'omega_{species}']) for ax, species in zip(axs, gas.species_names)]
    [ax.set_title(title) for ax, title in zip(axs, gas.species_names)]
    axs[-1].set_xlabel(xlabel)
    fig.suptitle("Molar rates of production (mol/m^3 - s)")

plt.show()
