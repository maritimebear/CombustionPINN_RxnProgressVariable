#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 11:55:02 2023

@author: polarbear
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt

import includes
import network

# datafile = "./rxn_progress_data.csv"
datafile = "./c_eqn_solution.csv"
batch_size = 64
learning_rate = 1e-4
num_epochs = 10_000

torch.manual_seed(7673345)
torch.set_default_dtype(torch.float64)

df = pd.read_csv(datafile)

dataset = includes.Dataset(df)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
network = network.FCN(1, 1, 64, 4)
loss = torch.nn.MSELoss()
optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate)

# Test grid
testgrid = torch.linspace(0, 2e-2, 101).reshape(-1, 1)

# Training loop
losses_epoch = list()
for epoch in range(num_epochs):
    losses_epoch.append(includes.train(dataloader, network, loss, optimiser))
    
    y_test = network(testgrid)
    print(f"Epoch: {epoch}, Epoch loss: {losses_epoch[-1]}")
    
    if not (epoch + 1) % 10:
        fig, axs = plt.subplots(2, 1, figsize=(4,8))
        axs[0].semilogy(losses_epoch)
        axs[1].plot(testgrid, y_test.detach().numpy())
    
        plt.show()
    