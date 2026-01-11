"""Utility functions for Aegis framework."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enable deterministic operations (slower)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Set environment variables for deterministic operations
    if deterministic:
        os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"


def create_prng_key(seed: int) -> jax.Array:
    """Create a JAX PRNG key from an integer seed.

    Args:
        seed: Integer seed value

    Returns:
        JAX PRNG key
    """
    return jax.random.PRNGKey(seed)


def split_key(key: jax.Array, num: int = 2) -> Tuple[jax.Array, ...]:
    """Split a PRNG key into multiple subkeys.

    Args:
        key: PRNG key to split
        num: Number of subkeys to generate

    Returns:
        Tuple of PRNG subkeys
    """
    return tuple(jax.random.split(key, num))


def explained_variance(y_pred: jax.Array, y_true: jax.Array) -> float:
    """Compute explained variance between predictions and targets.

    Explained variance measures how well the value function predicts returns:
    - 1.0: Perfect predictions
    - 0.0: Predictions are as good as the mean
    - <0: Predictions are worse than the mean

    Args:
        y_pred: Predicted values
        y_true: True values (returns)

    Returns:
        Explained variance (float between -inf and 1)
    """
    var_y = jnp.var(y_true)
    return 1.0 - jnp.var(y_true - y_pred) / (var_y + 1e-8)


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jax.Array, jax.Array]:
    """Compute Generalized Advantage Estimation.

    GAE provides a bias-variance tradeoff in advantage estimation:
    - lambda=0: One-step TD (low variance, high bias)
    - lambda=1: Monte Carlo (high variance, low bias)

    Args:
        rewards: Rewards array of shape [T]
        values: Value estimates of shape [T+1] (includes bootstrap)
        dones: Done flags of shape [T]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        Tuple of (advantages, returns) each of shape [T]
    """
    T = len(rewards)
    advantages = jnp.zeros(T)
    last_gae = 0.0

    # Scan backwards to compute GAE
    def gae_step(carry, t):
        last_gae = carry
        next_value = values[t + 1]
        done = dones[t]
        reward = rewards[t]
        value = values[t]

        # TD error: r + gamma * V(s') * (1 - done) - V(s)
        delta = reward + gamma * next_value * (1 - done) - value

        # GAE: delta + gamma * lambda * (1 - done) * GAE(t+1)
        gae = delta + gamma * gae_lambda * (1 - done) * last_gae

        return gae, gae

    # Reverse scan
    _, advantages = jax.lax.scan(
        gae_step,
        0.0,
        jnp.arange(T - 1, -1, -1),
    )
    advantages = advantages[::-1]  # Reverse back

    # Returns = Advantages + Values
    returns = advantages + values[:-1]

    return advantages, returns


def compute_gae_numpy(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Numpy implementation of GAE (for actors without JAX).

    Args:
        rewards: Rewards array of shape [T]
        values: Value estimates of shape [T+1]
        dones: Done flags of shape [T]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        Tuple of (advantages, returns) each of shape [T]
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_value = values[t + 1]
        done = dones[t]
        delta = rewards[t] + gamma * next_value * (1 - done) - values[t]
        last_gae = delta + gamma * gae_lambda * (1 - done) * last_gae
        advantages[t] = last_gae

    returns = advantages + values[:-1]
    return advantages, returns


def flatten_batch(batch_of_batches: list) -> Dict[str, jax.Array]:
    """Flatten a list of batches into a single batch.

    Args:
        batch_of_batches: List of Batch objects

    Returns:
        Dictionary with concatenated arrays
    """
    from aegis.core.types import Batch

    keys = [
        "observations",
        "actions",
        "rewards",
        "dones",
        "old_log_probs",
        "old_values",
        "advantages",
        "returns",
        "policy_versions",
    ]

    result = {}
    for key in keys:
        arrays = [getattr(b, key) for b in batch_of_batches]
        result[key] = jnp.concatenate(arrays, axis=0)

    # Handle optional fields
    if all(b.weights is not None for b in batch_of_batches):
        result["weights"] = jnp.concatenate([b.weights for b in batch_of_batches])
    if all(b.indices is not None for b in batch_of_batches):
        result["indices"] = jnp.concatenate([b.indices for b in batch_of_batches])

    return result


def shuffle_batch(batch: Any, key: jax.Array) -> Any:
    """Shuffle a batch along the first axis.

    Args:
        batch: Batch object or pytree to shuffle
        key: PRNG key for shuffling

    Returns:
        Shuffled batch
    """
    # Get batch size from first leaf
    leaves = jax.tree_util.tree_leaves(batch)
    batch_size = leaves[0].shape[0]

    # Generate random permutation
    perm = jax.random.permutation(key, batch_size)

    # Apply permutation to all leaves
    return jax.tree_util.tree_map(lambda x: x[perm], batch)


def minibatch_iterator(batch: Any, minibatch_size: int, key: jax.Array):
    """Iterate over minibatches of a batch.

    Args:
        batch: Batch object to iterate over
        minibatch_size: Size of each minibatch
        key: PRNG key for shuffling

    Yields:
        Minibatch slices
    """
    # Shuffle batch
    shuffled = shuffle_batch(batch, key)

    # Get batch size
    leaves = jax.tree_util.tree_leaves(shuffled)
    batch_size = leaves[0].shape[0]

    # Iterate over minibatches
    for start in range(0, batch_size, minibatch_size):
        end = min(start + minibatch_size, batch_size)
        yield jax.tree_util.tree_map(lambda x: x[start:end], shuffled)


def soft_update(
    params: Dict[str, Any],
    target_params: Dict[str, Any],
    tau: float = 0.005,
) -> Dict[str, Any]:
    """Soft update of target network parameters.

    target = (1 - tau) * target + tau * source

    Args:
        params: Source parameters
        target_params: Target parameters to update
        tau: Interpolation factor (0 = no update, 1 = hard update)

    Returns:
        Updated target parameters
    """
    return jax.tree_util.tree_map(
        lambda p, tp: (1 - tau) * tp + tau * p,
        params,
        target_params,
    )


def polyak_average(params_list: list, weights: Optional[list] = None) -> Dict[str, Any]:
    """Compute weighted average of multiple parameter sets.

    Args:
        params_list: List of parameter dictionaries
        weights: Optional weights (uniform if None)

    Returns:
        Averaged parameters
    """
    if weights is None:
        weights = [1.0 / len(params_list)] * len(params_list)

    def weighted_sum(*ps):
        return sum(w * p for w, p in zip(weights, ps))

    return jax.tree_util.tree_map(weighted_sum, *params_list)


class Timer:
    """Simple timer for profiling."""

    def __init__(self):
        self.times = {}
        self._start_times = {}

    def start(self, name: str) -> None:
        """Start timing a named section."""
        import time

        self._start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing and record duration."""
        import time

        duration = time.perf_counter() - self._start_times[name]
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(duration)
        return duration

    def mean(self, name: str) -> float:
        """Get mean time for a section."""
        return np.mean(self.times.get(name, [0.0]))

    def summary(self) -> Dict[str, float]:
        """Get summary of all timings."""
        return {name: self.mean(name) for name in self.times}
