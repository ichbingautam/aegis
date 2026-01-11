"""Base algorithm interface for Aegis.

Defines the abstract interface that all RL algorithms must implement,
enabling algorithm-agnostic training and fair benchmarking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
import optax

from aegis.core.types import Batch, Metrics, TrainState


class BaseAlgorithm(ABC):
    """Abstract base class for RL algorithms.

    All algorithms must implement:
    - loss_fn: Compute loss given parameters and batch
    - update_step: Perform one optimization step

    Attributes:
        config: Algorithm configuration dictionary
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the algorithm.

        Args:
            config: Configuration dictionary with hyperparameters
        """
        self.config = config

        # Common hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)

    @abstractmethod
    def loss_fn(
        self,
        params: dict[str, Any],
        batch: Batch,
        network: Any,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute the loss function.

        Args:
            params: Network parameters
            batch: Batch of experience
            network: Neural network module

        Returns:
            Tuple of (scalar loss, auxiliary metrics dict)
        """
        pass

    @abstractmethod
    def update_step(
        self,
        state: TrainState,
        batch: Batch,
        network: Any,
    ) -> tuple[TrainState, Metrics]:
        """Perform one optimization step.

        Args:
            state: Current training state
            batch: Batch of experience
            network: Neural network module

        Returns:
            Tuple of (updated state, training metrics)
        """
        pass

    def compute_advantages(
        self,
        rewards: jax.Array,
        values: jax.Array,
        dones: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute GAE advantages and returns.

        Args:
            rewards: Reward array [T]
            values: Value estimates [T+1]
            dones: Done flags [T]

        Returns:
            Tuple of (advantages, returns)
        """
        T = len(rewards)
        advantages = jnp.zeros(T)

        def gae_step(gae, t):
            t_rev = T - 1 - t  # Reverse index
            delta = (
                rewards[t_rev] + self.gamma * values[t_rev + 1] * (1 - dones[t_rev]) - values[t_rev]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t_rev]) * gae
            return gae, gae

        _, advantages_rev = jax.lax.scan(gae_step, 0.0, jnp.arange(T))
        advantages = advantages_rev[::-1]
        returns = advantages + values[:-1]

        return advantages, returns

    def create_optimizer(self) -> optax.GradientTransformation:
        """Create optimizer from config.

        Returns:
            Optax optimizer
        """
        lr = self.config.get("learning_rate", 2.5e-4)
        schedule = self.config.get("lr_schedule", "constant")
        total_steps = self.config.get("total_steps", 1_000_000)

        # Learning rate schedule
        if schedule == "linear":
            lr_schedule = optax.linear_schedule(
                init_value=lr,
                end_value=0.0,
                transition_steps=total_steps,
            )
        elif schedule == "cosine":
            lr_schedule = optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=total_steps,
            )
        else:
            lr_schedule = lr

        # Optimizer with gradient clipping
        return optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(
                learning_rate=lr_schedule,
                b1=self.config.get("adam_beta1", 0.9),
                b2=self.config.get("adam_beta2", 0.999),
                eps=self.config.get("adam_epsilon", 1e-5),
            ),
        )


def create_algorithm(name: str, config: dict[str, Any]) -> BaseAlgorithm:
    """Factory function to create algorithm by name.

    Args:
        name: Algorithm name ('ppo' or 'vmpo')
        config: Configuration dictionary

    Returns:
        Algorithm instance

    Raises:
        ValueError: If algorithm name is unknown
    """
    from aegis.algorithms.ppo import PPOAlgorithm
    from aegis.algorithms.vmpo import VMPOAlgorithm

    algorithms = {
        "ppo": PPOAlgorithm,
        "vmpo": VMPOAlgorithm,
    }

    if name not in algorithms:
        raise ValueError(f"Unknown algorithm: {name}. " f"Available: {list(algorithms.keys())}")

    return algorithms[name](config)
