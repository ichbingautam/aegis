"""SumTree data structure for Prioritized Experience Replay.

A binary tree where each node stores the sum of its children.
Leaf nodes store individual transition priorities, enabling
O(log N) sampling and updates.
"""

from __future__ import annotations

import math

import numpy as np


class SumTree:
    """Binary sum tree for efficient priority-based sampling.

    The tree structure enables:
    - O(log N) sampling proportional to priority
    - O(log N) priority updates
    - O(1) total priority access

    Memory layout:
    ```
    Tree:       [0]         <- root (sum of all)
               /   \\
             [1]   [2]      <- internal nodes
            / \\   / \\
          [3][4][5][6]      <- leaves (actual priorities)

    Array: [root, internal..., leaves...]
    Leaves start at index (capacity - 1)
    ```

    Attributes:
        capacity: Maximum number of elements (power of 2)
        tree: Array storing tree nodes
        data_pointer: Current write position (circular)
        size: Current number of stored elements
    """

    def __init__(self, capacity: int):
        """Initialize a SumTree.

        Args:
            capacity: Maximum number of elements. Will be rounded up
                     to the nearest power of 2 for efficient indexing.
        """
        # Round capacity to power of 2
        self.capacity = 1 << math.ceil(math.log2(max(capacity, 2)))

        # Tree has 2 * capacity - 1 nodes
        # Leaves are at indices [capacity-1, 2*capacity-2]
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)

        # Circular buffer pointer
        self.data_pointer = 0
        self.size = 0

        # Precompute leaf offset
        self._leaf_offset = self.capacity - 1

    def add(self, priority: float) -> int:
        """Add a new priority value.

        Args:
            priority: Priority value (must be > 0)

        Returns:
            Index where the priority was stored
        """
        # Get leaf index
        idx = self.data_pointer
        tree_idx = idx + self._leaf_offset

        # Update tree
        self.update(tree_idx, priority)

        # Advance pointer (circular)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        return idx

    def update(self, tree_idx: int, priority: float) -> None:
        """Update priority at a tree index.

        Args:
            tree_idx: Index in the tree array
            priority: New priority value
        """
        # Compute change
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # Propagate change up to root
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def update_leaf(self, data_idx: int, priority: float) -> None:
        """Update priority by data index.

        Args:
            data_idx: Index in the data buffer [0, capacity)
            priority: New priority value
        """
        tree_idx = data_idx + self._leaf_offset
        self.update(tree_idx, priority)

    def sample(self, value: float) -> tuple[int, float, int]:
        """Sample a leaf proportional to priority.

        Args:
            value: Random value in [0, total_priority)

        Returns:
            Tuple of (data_index, priority, tree_index)
        """
        tree_idx = 0  # Start at root

        while tree_idx < self._leaf_offset:  # While not a leaf
            left_child = 2 * tree_idx + 1
            right_child = left_child + 1

            # Go left if value < left sum, else go right
            if value < self.tree[left_child]:
                tree_idx = left_child
            else:
                value -= self.tree[left_child]
                tree_idx = right_child

        data_idx = tree_idx - self._leaf_offset
        priority = self.tree[tree_idx]

        return data_idx, priority, tree_idx

    def batch_sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of indices using stratified sampling.

        Divides the total priority into segments and samples
        uniformly within each segment for better coverage.

        Args:
            batch_size: Number of samples to draw
            rng: Numpy random generator

        Returns:
            Tuple of (data_indices, priorities, tree_indices)
        """
        total = self.total()
        if total == 0:
            raise ValueError("Cannot sample from empty tree")

        # Stratified sampling: divide priority range into segments
        segment = total / batch_size
        samples = rng.uniform(0, segment, size=batch_size)
        samples += np.arange(batch_size) * segment

        # Sample from each segment
        data_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        tree_indices = np.zeros(batch_size, dtype=np.int64)

        for i, value in enumerate(samples):
            data_idx, priority, tree_idx = self.sample(value)
            data_indices[i] = data_idx
            priorities[i] = priority
            tree_indices[i] = tree_idx

        return data_indices, priorities, tree_indices

    def total(self) -> float:
        """Get total priority (root node value).

        Returns:
            Sum of all priorities
        """
        return self.tree[0]

    def max(self) -> float:
        """Get maximum priority.

        Returns:
            Maximum priority among all leaves
        """
        return np.max(self.tree[self._leaf_offset : self._leaf_offset + self.size])

    def min(self) -> float:
        """Get minimum non-zero priority.

        Returns:
            Minimum positive priority
        """
        leaves = self.tree[self._leaf_offset : self._leaf_offset + self.size]
        positive = leaves[leaves > 0]
        if len(positive) == 0:
            return 0.0
        return np.min(positive)

    def __len__(self) -> int:
        """Return current number of stored elements."""
        return self.size

    def __repr__(self) -> str:
        return f"SumTree(capacity={self.capacity}, size={self.size}, total={self.total():.4f})"


class MinTree:
    """Binary min tree for efficient minimum queries.

    Used in conjunction with SumTree to compute importance
    sampling weights efficiently.
    """

    def __init__(self, capacity: int):
        """Initialize a MinTree.

        Args:
            capacity: Maximum number of elements
        """
        self.capacity = 1 << math.ceil(math.log2(max(capacity, 2)))
        self.tree = np.full(2 * self.capacity - 1, float("inf"), dtype=np.float64)
        self._leaf_offset = self.capacity - 1

    def update(self, data_idx: int, value: float) -> None:
        """Update value at data index.

        Args:
            data_idx: Index in data buffer
            value: New value
        """
        tree_idx = data_idx + self._leaf_offset
        self.tree[tree_idx] = value

        # Propagate minimum up
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            left = self.tree[2 * tree_idx + 1]
            right = self.tree[2 * tree_idx + 2]
            self.tree[tree_idx] = min(left, right)

    def min(self) -> float:
        """Get minimum value.

        Returns:
            Minimum value in the tree
        """
        return self.tree[0]
