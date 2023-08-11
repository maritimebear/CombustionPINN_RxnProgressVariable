#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 22:45:10 2023

@author: polarbear

Classes to decouple loss calculations from other parts of neural-network code.
Intended to be AI-library agnostic.
"""

from typing import Callable, TypeVar


class WeightedScalarLoss():
    """
    Callable class to calculate a generic scalar loss, multiplied by a weight.
    """
    Tensor = TypeVar('Tensor')  # template <typename Tensor> class WeightedLoss;

    def __init__(self,
                 loss_fn: Callable[Tensor, float],
                 weight: float = 1.0):
        self.loss_fn = loss_fn  # Function/callable class pointer
        self.weight = weight
    # end __init__()

    def __call__(self,
                 prediction: Tensor,
                 target: Tensor) -> float:
        return ( self.weight * self.loss_fn(prediction, target) )
    # end __call__()
# end class WeightedScalarLoss
