# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
import numpy as np
import torch


class RobustKernel:
    @abstractmethod
    def apply(self, x: torch.Tensor,alpha=None) -> torch.Tensor:
        pass


class HuberRobustKernel(RobustKernel):
    def apply(self, x: torch.Tensor, alpha = None) -> torch.Tensor:
        weights = torch.ones_like(x)
        s = x * x
        weights[s > 1] = 1 / torch.sqrt(s)[s > 1]
        return weights
class AdaptiveBarronRobustKernel(RobustKernel):
    def __init__(self, alpha_init: float = 2.0, c: float = 1.0,
                 alpha_range=(-8, 2), num_alpha=50, N=10, steps=2000):
        """
        Adaptive Barron kernel with alpha search.

        Args:
            alpha_init: starting alpha value
            c: scale parameter
            alpha_range: (min, max) range of alphas for lookup
            num_alpha: number of alpha samples in lookup table
            N: truncation bound for Z(alpha) integral [-N, N]
            steps: number of integration steps
        """
        self.alpha = alpha_init
        self.c = c
        self.alpha_range = alpha_range
        self.num_alpha = num_alpha
        self.N = N
        self.steps = steps

        # Precompute lookup table for Z(alpha)
        self.alpha_grid, self.Z_table = self._precompute_Z()

    def _rho(self, r: torch.Tensor, alpha: float, c: float, eps: float = 1e-8) -> torch.Tensor:
        """
        Barron loss (Eq. 5) with stable handling of alpha=0 and alpha=2 cases.
        r : residual tensor
        alpha : shape parameter
        c : scale parameter
        eps : small constant for stability
        """
        s = (r / (c + eps)) ** 2  # normalized squared residuals

        if abs(alpha) < eps:
            # α → 0 → Cauchy loss
            return torch.log1p(s / 2.0)

        elif abs(alpha - 2.0) < eps:
            # α → 2 → Least Squares loss
            return 0.5 * s

        else:
            abs_term = abs(alpha - 2.0)
            base = 1.0 + s / abs_term
            return (abs_term / (alpha)) * (base**(alpha / 2.0) - 1.0)

    def _precompute_Z(self):
        """Numerical integration of Z(alpha) over truncated [-N, N]."""
        alphas = np.linspace(self.alpha_range[0], self.alpha_range[1], self.num_alpha)
        r = np.linspace(-self.N, self.N, self.steps)
        dr = r[1] - r[0]
        Z_vals = []
        for alpha in alphas:
            rho_vals = self._rho(r, alpha, c=1.0)
            integrand = np.exp(-rho_vals)
            Z_vals.append(np.sum(integrand) * dr)
        return alphas, np.array(Z_vals)

    def _get_Z(self, alpha):
        """Interpolate Z(alpha) from lookup table."""
        return np.interp(alpha, self.alpha_grid, self.Z_table)

    def find_best_alpha(self, residuals: torch.Tensor):
        """Optimize alpha given residuals (grid search)."""
        residuals_np = residuals.detach().cpu().numpy()
        alphas = self.alpha_grid
        best_alpha, best_loss = None, float("inf")

        for alpha in alphas:
            rho_vals = self._rho(residuals_np, alpha, self.c)
            Z = self._get_Z(alpha)
            loss = np.sum(rho_vals) + len(residuals_np) * np.log(self.c * Z)
            if loss < best_loss:
                best_alpha, best_loss = alpha, loss

        self.alpha = best_alpha  # update alpha
        return best_alpha

    def apply(self, x: torch.Tensor, alpha: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute IRLS weights for residuals x.

        Supports both scalar alpha (single value for all residuals)
        and tensor alpha (per-residual adaptive alpha).
        Handles the special case alpha == 2 (L2 loss) stably.
        """
        if alpha is None:
            alpha = torch.tensor(self.alpha)  # scalar case

        s = (x / self.c) ** 2

        # numerical safety epsilon
        eps = 1e-8

        # mask for alpha == 2 (within small tolerance)
        alpha_is_2 = torch.isclose(alpha, torch.tensor(2.0, device=x.device, dtype=x.dtype), atol=1e-6)

        # compute abs_term safely
        abs_term = torch.abs(alpha - 2.0)
        abs_term = torch.where(alpha_is_2, torch.ones_like(abs_term), abs_term + eps)

        # main weight formula
        weights = (1.0 / self.c**2) * (s / abs_term + 1.0) ** (alpha / 2.0 - 1.0)

        # replace cases where alpha == 2 with constant 1/c^2
        weights = torch.where(alpha_is_2, torch.full_like(weights, 1.0 / self.c**2), weights)

        # an adjustment made to make the weight of the dynamic pixels zero,
        # comment it to keep the classical implementation of kernels.
        weights = torch.where(alpha == -10, torch.tensor(0), weights)
        return weights
