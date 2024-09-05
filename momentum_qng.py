# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantum natural gradient optimizer with momentum"""
import pennylane as qml

# pylint: disable=too-many-branches
# pylint: disable=too-many-arguments
from pennylane import numpy as pnp
from pennylane.utils import _flatten, unflatten

from .qng import QNGOptimizer

class MomentumQNGOptimizer(QNGOptimizer):
    r"""Very cool docstring"""

    def __init__(self, stepsize=0.01, momentum=0.9, approx="block-diag", lam=0):
        super().__init__(stepsize)
        self.momentum = momentum
        self.accumulation = None

    def apply_grad(self, grad, args):
        r"""Update the parameter array :math:`x` for a single optimization step. Flattens and
        unflattens the inputs to maintain nested iterables as the parameters of the optimization.

        Args:
            grad (array): The gradient of the objective
                function at point :math:`x^{(t)}`: :math:`\nabla f(x^{(t)})`
            args (array): the current value of the variables :math:`x^{(t)}`

        Returns:
            array: the new values :math:`x^{(t+1)}`
        """
        args_new = list(args)
        mt = self.metric_tensor if isinstance(self.metric_tensor, tuple) else (self.metric_tensor,)

        trained_index = 0
        new_accumulation = []
        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", False):
                grad_flat = pnp.array(list(_flatten(grad[trained_index])))
                # self.metric_tensor has already been reshaped to 2D, matching flat gradient.
                qng_update = pnp.linalg.solve(mt[trained_index], grad_flat)

                if self.accumulation is None:
                    accum = 0 * grad_flat
                else:
                    accum = self.accumulation[trained_index]

                accum += self.stepsize * update
                new_accumulation.append(accum)
                args_new[index] = arg - unflatten(accum, grad[trained_index])

                trained_index += 1

        self.accumulation = new_accumulation

        return tuple(args_new)
