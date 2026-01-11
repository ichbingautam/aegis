"""Core type definitions for Aegis.

This module defines the fundamental data structures used throughout
the framework, including Trajectory and Batch dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# JAX is optional for type definitions
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = np
    JAX_AVAILABLE = False


# Type aliases
Array = Union[np.ndarray, Any]  # jax.Array when available
Params = Dict[str, Any]
PRNGKey = Any  # jax.Array when available


@dataclass
class Trajectory:
    """A trajectory of experience collected by an actor.

    Contains a sequence of (observation, action, reward, done) tuples
    along with additional information needed for learning.

    Attributes:
        observations: Array of shape [T+1, *obs_shape] - includes bootstrap obs
        actions: Array of shape [T, *action_shape]
        rewards: Array of shape [T]
        dones: Array of shape [T] - episode termination flags
        log_probs: Array of shape [T] - log probabilities under behavior policy
        values: Array of shape [T+1] - value estimates (includes bootstrap)
        policy_version: Version of the policy used to collect this trajectory
        actor_id: ID of the actor that collected this trajectory
        env_id: ID of the environment within the actor
    """

    observations: Array
    actions: Array
    rewards: Array
    dones: Array
    log_probs: Array
    values: Array
    policy_version: int
    actor_id: int = 0
    env_id: int = 0

    def __post_init__(self):
        """Validate trajectory dimensions."""
        T = len(self.rewards)
        assert len(self.actions) == T, f"Expected {T} actions, got {len(self.actions)}"
        assert len(self.dones) == T, f"Expected {T} dones, got {len(self.dones)}"
        assert len(self.log_probs) == T, f"Expected {T} log_probs, got {len(self.log_probs)}"
        assert len(self.observations) == T + 1, f"Expected {T+1} obs, got {len(self.observations)}"
        assert len(self.values) == T + 1, f"Expected {T+1} values, got {len(self.values)}"

    @property
    def length(self) -> int:
        """Number of transitions in this trajectory."""
        return len(self.rewards)

    def to_numpy(self) -> Trajectory:
        """Convert all arrays to numpy."""
        return Trajectory(
            observations=np.asarray(self.observations),
            actions=np.asarray(self.actions),
            rewards=np.asarray(self.rewards),
            dones=np.asarray(self.dones),
            log_probs=np.asarray(self.log_probs),
            values=np.asarray(self.values),
            policy_version=self.policy_version,
            actor_id=self.actor_id,
            env_id=self.env_id,
        )

    def to_jax(self) -> Trajectory:
        """Convert all arrays to JAX arrays."""
        return Trajectory(
            observations=jnp.asarray(self.observations),
            actions=jnp.asarray(self.actions),
            rewards=jnp.asarray(self.rewards),
            dones=jnp.asarray(self.dones),
            log_probs=jnp.asarray(self.log_probs),
            values=jnp.asarray(self.values),
            policy_version=self.policy_version,
            actor_id=self.actor_id,
            env_id=self.env_id,
        )


@dataclass
class Batch:
    """A batch of experience for learning.

    Created by sampling from the replay buffer and computing
    advantages. All arrays have shape [batch_size, ...].

    Attributes:
        observations: Batch of observations [B, *obs_shape]
        actions: Batch of actions [B, *action_shape]
        rewards: Batch of rewards [B]
        dones: Batch of done flags [B]
        old_log_probs: Log probs under behavior policy [B]
        old_values: Value estimates from behavior policy [B]
        advantages: Computed advantage estimates [B]
        returns: Computed returns (targets for value function) [B]
        policy_versions: Policy version for each sample [B]
        weights: Importance sampling weights (for PER) [B]
        indices: Buffer indices (for priority updates) [B]
    """

    observations: Array
    actions: Array
    rewards: Array
    dones: Array
    old_log_probs: Array
    old_values: Array
    advantages: Array
    returns: Array
    policy_versions: Array
    weights: Optional[Array] = None
    indices: Optional[Array] = None

    @property
    def batch_size(self) -> int:
        """Number of samples in this batch."""
        return len(self.observations)

    def normalize_advantages(self) -> Batch:
        """Return new batch with normalized advantages."""
        mean = jnp.mean(self.advantages)
        std = jnp.std(self.advantages) + 1e-8
        normalized = (self.advantages - mean) / std
        return Batch(
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            old_log_probs=self.old_log_probs,
            old_values=self.old_values,
            advantages=normalized,
            returns=self.returns,
            policy_versions=self.policy_versions,
            weights=self.weights,
            indices=self.indices,
        )

    def slice(self, start: int, end: int) -> Batch:
        """Return a slice of this batch."""
        return Batch(
            observations=self.observations[start:end],
            actions=self.actions[start:end],
            rewards=self.rewards[start:end],
            dones=self.dones[start:end],
            old_log_probs=self.old_log_probs[start:end],
            old_values=self.old_values[start:end],
            advantages=self.advantages[start:end],
            returns=self.returns[start:end],
            policy_versions=self.policy_versions[start:end],
            weights=self.weights[start:end] if self.weights is not None else None,
            indices=self.indices[start:end] if self.indices is not None else None,
        )


@dataclass
class TrainState:
    """Training state for the learner.

    Encapsulates all state needed for training, enabling
    functional updates and easy checkpointing.

    Attributes:
        params: Neural network parameters
        opt_state: Optimizer state
        step: Current training step
        policy_version: Current policy version
        dual_params: Dual variables for V-MPO (eta, alpha)
    """

    params: Params
    opt_state: Any
    step: int = 0
    policy_version: int = 0
    dual_params: Optional[Dict[str, float]] = None

    def increment_step(self) -> TrainState:
        """Return new state with incremented step."""
        return TrainState(
            params=self.params,
            opt_state=self.opt_state,
            step=self.step + 1,
            policy_version=self.policy_version,
            dual_params=self.dual_params,
        )

    def increment_version(self) -> TrainState:
        """Return new state with incremented policy version."""
        return TrainState(
            params=self.params,
            opt_state=self.opt_state,
            step=self.step,
            policy_version=self.policy_version + 1,
            dual_params=self.dual_params,
        )


@dataclass
class Metrics:
    """Collection of training metrics.

    Attributes:
        loss: Total loss value
        policy_loss: Policy/actor loss component
        value_loss: Value/critic loss component
        entropy: Policy entropy
        kl_divergence: KL divergence from old policy
        clip_fraction: Fraction of samples clipped (PPO)
        explained_variance: How well value function explains returns
        grad_norm: Gradient norm before clipping
        extra: Additional algorithm-specific metrics
    """

    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    kl_divergence: float = 0.0
    clip_fraction: float = 0.0
    explained_variance: float = 0.0
    grad_norm: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        d = {
            "loss/total": self.loss,
            "loss/policy": self.policy_loss,
            "loss/value": self.value_loss,
            "policy/entropy": self.entropy,
            "policy/kl_divergence": self.kl_divergence,
            "policy/clip_fraction": self.clip_fraction,
            "value/explained_variance": self.explained_variance,
            "grad/norm": self.grad_norm,
        }
        for k, v in self.extra.items():
            d[f"extra/{k}"] = v
        return d


@dataclass
class ActorInfo:
    """Information about an actor's current state.

    Used for monitoring and debugging distributed training.
    """

    actor_id: int
    policy_version: int
    episodes_completed: int
    steps_collected: int
    mean_episode_return: float
    mean_episode_length: float
    steps_per_second: float


@dataclass
class LearnerInfo:
    """Information about the learner's current state."""

    learner_id: int
    policy_version: int
    updates_completed: int
    samples_processed: int
    updates_per_second: float
    gpu_utilization: float = 0.0


def tree_stack(trees: list) -> Any:
    """Stack a list of pytrees along the first axis.

    Args:
        trees: List of pytrees with the same structure

    Returns:
        Single pytree with arrays stacked
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for tree_stack")
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trees)


def tree_unstack(tree: Any) -> list:
    """Unstack a pytree along the first axis.

    Args:
        tree: Pytree with arrays that have a batch dimension

    Returns:
        List of pytrees, one per batch element
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for tree_unstack")
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    n = leaves[0].shape[0]
    return [treedef.unflatten([leaf[i] for leaf in leaves]) for i in range(n)]
