
"""Scheduler for training."""

from typing import List

import torch
from torch.optim.lr_scheduler import _LRScheduler


class AnnealingStepLR(_LRScheduler):
    """Anenealing scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer.
        mu_i (float, optional): Initial learning rate.
        mu_f (float, optional): Final learning rate.
        n (float, optional): Annealing steps.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, mu_i: float = 5e-4,
                 mu_f: float = 5e-5, n: float = 1.6e6) -> None:
        self.mu_i = mu_i
        self.mu_f = mu_f
        self.n = n

        super().__init__(optimizer)

    def get_lr(self) -> List[float]:

        return [max(self.mu_f + (self.mu_i - self.mu_f) *
                    (1.0 - self.last_epoch / self.n), self.mu_f)
                for base_lr in self.base_lrs]


class Annealer:
    """Annealer for training.

    Args:
        init (float): Initial value.
        final (float): Final value.
        steps (int): Number of annealing steps.
    """

    def __init__(self, init: float, final: float, steps: int) -> None:

        self.init = init
        self.final = final
        self.steps = steps

        # Current value
        self.t = 0
        self.current = init

    def __iter__(self):
        return self

    def __next__(self) -> float:
        self.t += 1
        value = max(
            self.final + (self.init - self.final) * (1 - self.t / self.steps),
            self.final
        )
        self.current = value

        return value


class VarianceAnnealer:
    """Annealer for variance used in Consistent GQN training."""

    def __init__(self) -> None:
        # Current value
        self.t = 0

    def __iter__(self):
        return self

    def __next__(self) -> float:
        self.t += 1

        if self.t <= 100000:
            value = 2.0
        elif 100000 < self.t <= 150000:
            value = 0.2
        elif 150000 < self.t <= 200000:
            value = 0.4
        else:
            value = 0.9

        # Return sigma
        return value ** 0.5
