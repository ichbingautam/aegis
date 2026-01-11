"""Prioritized Experience Replay Buffer.

This module implements a distributed prioritized experience replay
buffer designed for the actor-learner architecture. It combines:
- SumTree for O(log N) priority-based sampling
- Ring buffer for fixed-memory trajectory storage
- Ray actor for distributed access
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from aegis.core.types import Batch, Trajectory
from aegis.replay.ring_buffer import TrajectoryBuffer
from aegis.replay.sum_tree import MinTree, SumTree


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer.

    Implements the PER algorithm from Schaul et al. (2016) with:
    - Proportional prioritization (p_i = |δ_i| + ε)^α
    - Importance sampling weights for bias correction
    - Stratified sampling for better coverage

    Attributes:
        capacity: Maximum number of transitions
        alpha: Prioritization exponent (0 = uniform, 1 = full priority)
        beta: Importance sampling exponent (annealed to 1)
        epsilon: Small constant to ensure non-zero priority
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        action_shape: tuple,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 1_000_000,
        epsilon: float = 1e-6,
        obs_dtype: np.dtype = np.float32,
        action_dtype: np.dtype = np.int32,
    ):
        """Initialize the prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions
            obs_shape: Shape of observations
            action_shape: Shape of actions
            alpha: Priority exponent
            beta_start: Initial importance sampling weight
            beta_end: Final importance sampling weight
            beta_frames: Frames over which to anneal beta
            epsilon: Minimum priority constant
            obs_dtype: Data type for observations
            action_dtype: Data type for actions
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.epsilon = epsilon

        # Data storage
        self.buffer = TrajectoryBuffer(
            capacity=capacity,
            obs_shape=obs_shape,
            action_shape=action_shape,
            obs_dtype=obs_dtype,
            action_dtype=action_dtype,
        )

        # Priority trees
        self.sum_tree = SumTree(capacity)
        self.min_tree = MinTree(capacity)

        # Track maximum priority for new samples
        self.max_priority = 1.0

        # Frame counter for beta annealing
        self.frame = 0

        # Random generator
        self.rng = np.random.default_rng()

    @property
    def beta(self) -> float:
        """Current importance sampling weight (annealed)."""
        fraction = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        policy_version: int,
        priority: float | None = None,
    ) -> int:
        """Add a transition to the buffer.

        Args:
            observation: Observation array
            action: Action array
            reward: Reward value
            done: Episode termination flag
            log_prob: Log probability of action
            value: Value estimate
            policy_version: Policy version used
            priority: Optional priority (uses max_priority if None)

        Returns:
            Index where transition was stored
        """
        # Add to data buffer
        idx = self.buffer.add(
            observation=observation,
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value,
            policy_version=policy_version,
        )

        # Compute priority
        if priority is None:
            priority = self.max_priority
        priority = (priority + self.epsilon) ** self.alpha

        # Update trees
        self.sum_tree.update_leaf(idx, priority)
        self.min_tree.update(idx, priority)

        return idx

    def add_trajectory(
        self, trajectory: Trajectory, priorities: np.ndarray | None = None
    ) -> np.ndarray:
        """Add a full trajectory to the buffer.

        Args:
            trajectory: Trajectory object to add
            priorities: Optional priority for each transition

        Returns:
            Array of indices where transitions were stored
        """
        T = trajectory.length

        # Convert trajectory to arrays
        observations = np.asarray(trajectory.observations[:-1])  # Exclude bootstrap
        actions = np.asarray(trajectory.actions)
        rewards = np.asarray(trajectory.rewards)
        dones = np.asarray(trajectory.dones)
        log_probs = np.asarray(trajectory.log_probs)
        values = np.asarray(trajectory.values[:-1])  # Exclude bootstrap
        policy_versions = np.full(T, trajectory.policy_version, dtype=np.int32)

        # Add to buffer
        indices = self.buffer.add_batch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            log_probs=log_probs,
            values=values,
            policy_versions=policy_versions,
        )

        # Update priorities
        if priorities is None:
            priorities = np.full(T, self.max_priority)

        for i, idx in enumerate(indices):
            p = (priorities[i] + self.epsilon) ** self.alpha
            self.sum_tree.update_leaf(idx, p)
            self.min_tree.update(idx, p)

        return indices

    def sample(self, batch_size: int) -> tuple[Batch, np.ndarray]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (Batch, tree_indices for priority updates)
        """
        self.frame += batch_size

        # Sample indices proportional to priority
        data_indices, priorities, tree_indices = self.sum_tree.batch_sample(batch_size, self.rng)

        # Get data
        data = self.buffer.get(data_indices)

        # Compute importance sampling weights
        total_priority = self.sum_tree.total()
        min_priority = self.min_tree.min()

        # P(i) = p_i / sum(p)
        sampling_probs = priorities / total_priority

        # w_i = (N * P(i))^(-beta) / max_j(w_j)
        # max_w = (N * min_P)^(-beta)
        weights = (len(self.buffer) * sampling_probs) ** (-self.beta)
        max_weight = (len(self.buffer) * min_priority / total_priority) ** (-self.beta)
        weights = weights / max_weight  # Normalize

        # TODO: Compute advantages here or in learner?
        # For now, return placeholder advantages/returns
        advantages = np.zeros(batch_size, dtype=np.float32)
        returns = np.zeros(batch_size, dtype=np.float32)

        batch = Batch(
            observations=data["observations"],
            actions=data["actions"],
            rewards=data["rewards"],
            dones=data["dones"],
            old_log_probs=data["log_probs"],
            old_values=data["values"],
            advantages=advantages,
            returns=returns,
            policy_versions=data["policy_versions"],
            weights=weights.astype(np.float32),
            indices=tree_indices,
        )

        return batch, tree_indices

    def update_priorities(self, tree_indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities after learning.

        Args:
            tree_indices: Indices from sample() call
            priorities: New priority values (typically |TD-error|)
        """
        for idx, priority in zip(tree_indices, priorities, strict=False):
            # Convert tree index to data index
            data_idx = idx - self.sum_tree._leaf_offset

            # Clamp priority to reasonable range
            priority = np.clip(priority, 1e-6, 1e6)

            # Update max priority
            self.max_priority = max(self.max_priority, priority)

            # Compute prioritized value
            p = (priority + self.epsilon) ** self.alpha

            # Update trees
            self.sum_tree.update(idx, p)
            self.min_tree.update(data_idx, p)

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        return (
            f"PrioritizedReplayBuffer(capacity={self.capacity}, "
            f"size={len(self)}, alpha={self.alpha}, beta={self.beta:.3f})"
        )


# Ray-based distributed version
if RAY_AVAILABLE:

    @ray.remote
    class DistributedReplayBuffer:
        """Ray actor wrapper for distributed prioritized replay.

        This actor manages a shard of the distributed replay buffer,
        handling concurrent access from multiple actors and learners.
        """

        def __init__(
            self,
            capacity: int,
            obs_shape: tuple,
            action_shape: tuple,
            alpha: float = 0.6,
            beta_start: float = 0.4,
            **kwargs,
        ):
            """Initialize the distributed buffer shard."""
            self.buffer = PrioritizedReplayBuffer(
                capacity=capacity,
                obs_shape=obs_shape,
                action_shape=action_shape,
                alpha=alpha,
                beta_start=beta_start,
                **kwargs,
            )
            self.shard_id = 0

        def add(self, *args, **kwargs) -> int:
            """Add a transition."""
            return self.buffer.add(*args, **kwargs)

        def add_trajectory(
            self, trajectory: Trajectory, priorities: np.ndarray | None = None
        ) -> np.ndarray:
            """Add a trajectory."""
            return self.buffer.add_trajectory(trajectory, priorities)

        def sample(self, batch_size: int) -> tuple[dict, np.ndarray]:
            """Sample a batch."""
            batch, indices = self.buffer.sample(batch_size)
            # Convert Batch to dict for Ray serialization
            return {
                "observations": batch.observations,
                "actions": batch.actions,
                "rewards": batch.rewards,
                "dones": batch.dones,
                "old_log_probs": batch.old_log_probs,
                "old_values": batch.old_values,
                "advantages": batch.advantages,
                "returns": batch.returns,
                "policy_versions": batch.policy_versions,
                "weights": batch.weights,
                "indices": batch.indices,
            }, indices

        def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
            """Update priorities."""
            self.buffer.update_priorities(indices, priorities)

        def size(self) -> int:
            """Get current size."""
            return len(self.buffer)

        def stats(self) -> dict[str, Any]:
            """Get buffer statistics."""
            return {
                "size": len(self.buffer),
                "capacity": self.buffer.capacity,
                "beta": self.buffer.beta,
                "max_priority": self.buffer.max_priority,
            }
