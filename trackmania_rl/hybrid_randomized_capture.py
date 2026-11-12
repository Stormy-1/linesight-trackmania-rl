"""Seeded, state-independent action interventions for offline capture."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

SAFE_RANDOMIZED_ACTION_INDICES = tuple(range(9))


class SeededRandomizedActionPolicy:
    """Wrap a nominal policy with a predeclared local random intervention schedule.

    The intervention draw happens before the nominal policy is called.  It uses
    only the local PRNG state and therefore does not inspect observations,
    model outputs, or future telemetry when deciding whether to intervene.
    """

    def __init__(
        self,
        nominal_policy: Callable[[Any, Any], tuple[int, bool, float, np.ndarray]],
        *,
        seed: int,
        intervention_probability: float = 0.25,
        randomized_action_indices: Sequence[int] = SAFE_RANDOMIZED_ACTION_INDICES,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if not np.isfinite(intervention_probability) or not 0.0 <= intervention_probability <= 1.0:
            raise ValueError("intervention_probability must be finite and within [0, 1]")
        action_indices = tuple(randomized_action_indices)
        if not action_indices or any(
            isinstance(action, bool) or not isinstance(action, int) or action < 0 or action > 8
            for action in action_indices
        ):
            raise ValueError("randomized_action_indices must contain distinct integer actions in [0, 8]")
        if len(set(action_indices)) != len(action_indices):
            raise ValueError("randomized_action_indices must be distinct")
        self.nominal_policy = nominal_policy
        self.seed = seed
        self.intervention_probability = float(intervention_probability)
        self.randomized_action_indices = action_indices
        self._random = random.Random(seed)
        self.decision_count = 0
        self.last_action_is_randomized = False

    def __call__(self, image: Any, float_inputs: Any) -> tuple[int, bool, float, np.ndarray]:
        intervention = self._random.random() < self.intervention_probability
        nominal_action, nominal_greedy, q_value, q_values = self.nominal_policy(image, float_inputs)
        if intervention:
            action = self._random.choice(self.randomized_action_indices)
            q_array = np.asarray(q_values)
            if q_array.size == 0:
                raise ValueError("nominal policy must return non-empty q_values")
            greedy_action = int(np.argmax(q_array))
            action_was_greedy = action == greedy_action
        else:
            action = int(nominal_action)
            action_was_greedy = bool(nominal_greedy)
        self.last_action_is_randomized = intervention
        self.decision_count += 1
        return int(action), bool(action_was_greedy), float(q_value), q_values
