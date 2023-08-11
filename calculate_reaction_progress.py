# Script to calculate reaction progress variable from Cantera temperature results

import pandas as pd
import numpy as np

cantera_data = "./freely_propagating.csv"
x_column = "grid"
temp_column = "T"

# Assuming inlet temperature is minimum, outlet temperature is maximum
# reaction progress variable c(x) := (T(x) - T_unburnt) / (T_flametemp_adiabatic - T_unburnt)

df = pd.read_csv(cantera_data)
x, T = [df[key].to_numpy() for key in (x_column, temp_column)]

c = (T - T[0]) / (T[-1] - T[0]) # reaction progress variable

# Write to csv
pd.DataFrame(np.c_[x, c], columns=["x", "reaction_progress"]).to_csv("rxn_progress_data.csv", index=False)
