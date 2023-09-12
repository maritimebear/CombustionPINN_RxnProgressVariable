#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 20:14:26 2023

@author: polarbear

Convenience functions and classes for Cantera simulations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def add_species_fields(result: pd.DataFrame,
                       fields: np.ndarray,
                       field_name: str,
                       species_names: List[str]) -> None:
    """
    Function to add species-wise fields which are not included by Cantera in the result
    (like molar production rate) to the result.
    
    Parameters
    ----------
    result : pd.DataFrame
        pandas dataframe containing Cantera results
    fields : np.ndarray
        Fields to be added to result, shape must agree with shape of result dataframe
    field_name: str
        Name of the field to appear in the result dataframe
    species_names : List[str]
        List of species names in the correct order, obtained by Solution::species_names

    Returns
    -------
    None, result dataframe modified in-place
    """
    assert(result.shape[0] == fields.shape[0])
    for idx, sp_name in enumerate(species_names):
        result[f"{field_name}_{sp_name}"] = pd.Series(fields[:,idx])

def comparison_plot(axes: plt.Axes,
                    data_x: np.ndarray,
                    data_y: np.ndarray,
                    reference_x: np.ndarray,
                    reference_y: np.ndarray,
                    title: str = None,
                    xlabel: str = None,
                    ylabel: str = None) -> None:
    """
    Function to plot data vs reference onto an existing matplotlib Axes object.
    """
    with plt.ioff(): # Suppress interactive plotting -- else the figure gets plotted immediately
        axes.plot(data_x, data_y, label='Obtained')
        axes.plot(reference_x, reference_y, label='Reference')
        axes.legend()
        axes.set(title=title, xlabel=xlabel, ylabel=ylabel)    
