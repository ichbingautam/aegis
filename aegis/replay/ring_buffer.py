"""Lock-free ring buffer for high-throughput trajectory storage.

This module provides a high-performance ring buffer implementation
optimized for the SPSC (Single Producer Single Consumer) pattern
common in actor-learner architectures.

Note: For maximum performance, a Cython implementation is available
in ring_buffer.pyx. This pure Python version serves as a fallback.
"""

from __future__ import annotations

import threading
from typing import Generic, List, Optional, TypeVar

import numpy as np

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """Thread-safe ring buffer using Python's GIL.

    This is a pure Python implementation that leverages Python's GIL
    for basic thread safety. For true lock-free performance, use the
    Cython implementation.

    Attributes:
        capacity: Maximum number of elements
        buffer: Internal storage array
        write_idx: Current write position
        read_idx: Current read position
    """

    def __init__(self, capacity: int):
        """Initialize the ring buffer.

        Args:
            capacity: Maximum number of elements. Will be rounded
                     to the nearest power of 2.
        """
        # Round to power of 2 for efficient modulo
        self.capacity = 1 << ((capacity - 1).bit_length())
        self.mask = self.capacity - 1

        # Storage
        self.buffer: List[Optional[T]] = [None] * self.capacity

        # Indices (using atomic-like access via GIL)
        self._write_idx = 0
        self._read_idx = 0

        # Lock for multi-producer scenarios
        self._lock = threading.Lock()

    @property
    def write_idx(self) -> int:
        return self._write_idx

    @property
    def read_idx(self) -> int:
        return self._read_idx

    def push(self, item: T) -> bool:
        """Push an item to the buffer.

        Args:
            item: Item to push

        Returns:
            True if successful, False if buffer is full
        """
        with self._lock:
            next_write = (self._write_idx + 1) & self.mask

            # Check if buffer is full
            if next_write == self._read_idx:
                return False

            self.buffer[self._write_idx] = item
            self._write_idx = next_write
            return True

    def push_overwrite(self, item: T) -> Optional[T]:
        """Push an item, overwriting oldest if full.

        Args:
            item: Item to push

        Returns:
            Overwritten item if any, None otherwise
        """
        with self._lock:
            overwritten = None
            next_write = (self._write_idx + 1) & self.mask

            # If full, advance read pointer
            if next_write == self._read_idx:
                overwritten = self.buffer[self._read_idx]
                self._read_idx = (self._read_idx + 1) & self.mask

            self.buffer[self._write_idx] = item
            self._write_idx = next_write
            return overwritten

    def pop(self) -> Optional[T]:
        """Pop an item from the buffer.

        Returns:
            The oldest item, or None if buffer is empty
        """
        with self._lock:
            # Check if buffer is empty
            if self._read_idx == self._write_idx:
                return None

            item = self.buffer[self._read_idx]
            self.buffer[self._read_idx] = None  # Help GC
            self._read_idx = (self._read_idx + 1) & self.mask
            return item

    def peek(self) -> Optional[T]:
        """Peek at the oldest item without removing it.

        Returns:
            The oldest item, or None if buffer is empty
        """
        if self._read_idx == self._write_idx:
            return None
        return self.buffer[self._read_idx]

    def __len__(self) -> int:
        """Return current number of items."""
        write = self._write_idx
        read = self._read_idx
        if write >= read:
            return write - read
        return self.capacity - read + write

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._read_idx == self._write_idx

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return ((self._write_idx + 1) & self.mask) == self._read_idx

    def clear(self) -> None:
        """Clear all items from the buffer."""
        with self._lock:
            self.buffer = [None] * self.capacity
            self._write_idx = 0
            self._read_idx = 0

    def __repr__(self) -> str:
        return f"RingBuffer(capacity={self.capacity}, size={len(self)})"


class TrajectoryBuffer:
    """Efficient buffer for storing trajectory data.

    Stores trajectories as flat numpy arrays for efficient
    memory usage and fast access. Uses a ring buffer pattern
    for fixed memory footprint.

    Attributes:
        capacity: Maximum number of transitions
        obs_shape: Shape of observations
        action_shape: Shape of actions
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        action_shape: tuple,
        obs_dtype: np.dtype = np.float32,
        action_dtype: np.dtype = np.int32,
    ):
        """Initialize the trajectory buffer.

        Args:
            capacity: Maximum number of transitions
            obs_shape: Shape of a single observation
            action_shape: Shape of a single action
            obs_dtype: Data type for observations
            action_dtype: Data type for actions
        """
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        # Pre-allocate arrays
        self.observations = np.zeros((capacity,) + obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((capacity,) + action_shape, dtype=action_dtype)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.policy_versions = np.zeros(capacity, dtype=np.int32)

        # Ring buffer state
        self.write_idx = 0
        self.size = 0

        # Thread safety
        self._lock = threading.Lock()

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        policy_version: int,
    ) -> int:
        """Add a single transition.

        Args:
            observation: Observation array
            action: Action array
            reward: Reward value
            done: Episode termination flag
            log_prob: Log probability of action
            value: Value estimate
            policy_version: Policy version used

        Returns:
            Index where transition was stored
        """
        with self._lock:
            idx = self.write_idx

            self.observations[idx] = observation
            self.actions[idx] = action
            self.rewards[idx] = reward
            self.dones[idx] = done
            self.log_probs[idx] = log_prob
            self.values[idx] = value
            self.policy_versions[idx] = policy_version

            self.write_idx = (self.write_idx + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

            return idx

    def add_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        policy_versions: np.ndarray,
    ) -> np.ndarray:
        """Add a batch of transitions.

        Args:
            observations: Batch of observations [B, *obs_shape]
            actions: Batch of actions [B, *action_shape]
            rewards: Batch of rewards [B]
            dones: Batch of done flags [B]
            log_probs: Batch of log probabilities [B]
            values: Batch of value estimates [B]
            policy_versions: Batch of policy versions [B]

        Returns:
            Array of indices where transitions were stored
        """
        batch_size = len(observations)

        with self._lock:
            indices = np.arange(self.write_idx, self.write_idx + batch_size) % self.capacity

            for i, idx in enumerate(indices):
                self.observations[idx] = observations[i]
                self.actions[idx] = actions[i]
                self.rewards[idx] = rewards[i]
                self.dones[idx] = dones[i]
                self.log_probs[idx] = log_probs[i]
                self.values[idx] = values[i]
                self.policy_versions[idx] = policy_versions[i]

            self.write_idx = (self.write_idx + batch_size) % self.capacity
            self.size = min(self.size + batch_size, self.capacity)

            return indices

    def get(self, indices: np.ndarray) -> dict:
        """Get transitions at given indices.

        Args:
            indices: Array of indices to retrieve

        Returns:
            Dictionary of transition arrays
        """
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
            "log_probs": self.log_probs[indices],
            "values": self.values[indices],
            "policy_versions": self.policy_versions[indices],
        }

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"TrajectoryBuffer(capacity={self.capacity}, size={self.size})"
