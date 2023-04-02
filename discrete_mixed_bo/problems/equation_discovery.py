#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Welded Beam problem from https://link.springer.com/content/pdf/10.1007/s00158-018-2182-1.pdf
"""
from copy import deepcopy
from math import sqrt
from typing import Optional, Tuple

import numpy as np
import torch
from botorch.test_functions.base import ConstrainedBaseTestProblem
from torch import Tensor

from discrete_mixed_bo.problems.base import DiscreteTestProblem


class EquationDiscovery(DiscreteTestProblem, ConstrainedBaseTestProblem):

    def __init__(self, model) -> None:
        
        noise_std = None
        self.model = model
        self.continuous = False
        self.dim = len(model.product_space)
        self.num_constraints = 1
        self._bounds = model.product_space.bounds
        self._orig_cont_bounds_list = deepcopy(self._bounds)
        integer_indices = np.arange(self.dim).tolist()

        super().__init__(
            negate=False,
            noise_std=noise_std,
            integer_indices=integer_indices,
            categorical_indices=[],
        )
        self.register_buffer(
            "_orig_cont_bounds_tensor",
            torch.tensor(self._orig_cont_bounds_list).t(),
        )

    def evaluate_true(self, X: Tensor) -> Tensor:
        y,c,l = self.model.evaluate(X=X.numpy())
        return -torch.as_tensor(y)

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        y,c,l = self.model.evaluate(X=X.numpy())
        return -torch.as_tensor(c)
