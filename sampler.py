#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:00:21 2023

@author: polarbear
"""

import torch

from typing import TypeAlias, Protocol, Sequence

Tensor: TypeAlias = torch.Tensor


class CustomSampler(Protocol):
    def __call__(self) -> Tensor: ...


class UniformRandomSampler():
    """
    Callable class, returns tensors with uniform-random entries, in the shape (n_points, n_dims).
    Each column corresponds to a space or time dimension (x, y, z, t) in the governing equations.

    n_dims inferred from number of 'extents' arguments passed to constructor.

    Each sequence in 'extents' defines the sampling interval in the corresponding coordinate,
    i.e. the corresponding column of the returned tensor.
    """
    def __init__(self,
                 n_points: int,
                 extents: Sequence[Sequence[float]],
                 *,
                 requires_grad=True):
        """
        'extents' must be a sequence of sequences,
            eg. [(0.0, 1.0), (0.0, 0.0)] for e1: [0,1], e2: [0,0]
        """
        self.n_points = n_points
        self.extents = extents
        self.requires_grad = requires_grad
        # Lambdas used to generate tensors
        self.generate = lambda _range: (torch.Tensor(n_points, 1).uniform_(*_range).requires_grad_(requires_grad))

    def __call__(self) -> Tensor:
        return torch.hstack([self.generate(coord) for coord in self.extents])
