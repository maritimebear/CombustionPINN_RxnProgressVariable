#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 20:19:02 2023

@author: polarbear
"""
import torch.nn as nn


class FCN(nn.Module):
    """ Fully Connected Network"""
    def __init__(self,
                 input_neurons: int,
                 output_neurons: int,
                 hidden_neurons: int,
                 hidden_layers: int):
        """
        input_neurons, output_neurons, hidden_neurons: numbers of neurons
        hidden_layers: number of hidden layers
        """
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_neurons, hidden_neurons), nn.Tanh(),                       # Input layer
                                     *[nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()] * (hidden_layers),    # Hidden layers
                                     nn.Linear(hidden_neurons, output_neurons)                                  # Output layer
                                     )
     
    def forward(self, x):
        return self.network(x)
