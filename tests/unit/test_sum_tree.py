"""Unit tests for SumTree data structure."""

import numpy as np
import pytest

from aegis.replay.sum_tree import MinTree, SumTree


class TestSumTree:
    """Tests for SumTree priority sampling."""

    def test_init(self):
        """Test SumTree initialization."""
        tree = SumTree(100)
        assert tree.capacity == 128  # Rounded to power of 2
        assert len(tree) == 0
        assert tree.total() == 0.0

    def test_add_single(self):
        """Test adding a single priority."""
        tree = SumTree(16)
        idx = tree.add(1.0)

        assert idx == 0
        assert len(tree) == 1
        assert tree.total() == 1.0

    def test_add_multiple(self):
        """Test adding multiple priorities."""
        tree = SumTree(16)

        for i in range(10):
            tree.add(float(i + 1))

        assert len(tree) == 10
        assert tree.total() == 55.0  # 1 + 2 + ... + 10

    def test_update(self):
        """Test updating priorities."""
        tree = SumTree(16)

        # Add initial priorities
        for _i in range(5):
            tree.add(1.0)

        assert tree.total() == 5.0

        # Update one priority
        tree.update_leaf(2, 10.0)
        assert tree.total() == 14.0  # 1 + 1 + 10 + 1 + 1

    def test_sample(self):
        """Test priority-based sampling."""
        tree = SumTree(16)

        # Add priorities [1, 2, 3, 4]
        for i in range(4):
            tree.add(float(i + 1))

        total = tree.total()
        assert total == 10.0

        # Sample with value 0.5 should get index 0 (priority 1)
        idx, priority, _ = tree.sample(0.5)
        assert idx == 0

        # Sample with value 1.5 should get index 1 (priority 2)
        idx, priority, _ = tree.sample(1.5)
        assert idx == 1

        # Sample with value 9.5 should get index 3 (priority 4)
        idx, priority, _ = tree.sample(9.5)
        assert idx == 3

    def test_batch_sample(self):
        """Test stratified batch sampling."""
        tree = SumTree(100)
        rng = np.random.default_rng(42)

        # Add 50 priorities
        for _i in range(50):
            tree.add(rng.random())

        # Sample a batch
        indices, priorities, tree_indices = tree.batch_sample(10, rng)

        assert len(indices) == 10
        assert len(priorities) == 10
        assert len(tree_indices) == 10

        # All indices should be valid
        assert all(0 <= idx < 50 for idx in indices)

    def test_circular_overwrite(self):
        """Test circular buffer behavior."""
        tree = SumTree(8)  # Small capacity

        # Add more than capacity
        for _i in range(20):
            tree.add(1.0)

        assert len(tree) == 8  # Capped at capacity

    def test_max_min(self):
        """Test max and min priority queries."""
        tree = SumTree(16)

        for p in [1.0, 5.0, 3.0, 7.0, 2.0]:
            tree.add(p)

        assert tree.max() == 7.0
        assert tree.min() == 1.0


class TestMinTree:
    """Tests for MinTree."""

    def test_init(self):
        """Test MinTree initialization."""
        tree = MinTree(100)
        assert tree.capacity == 128
        assert tree.min() == float("inf")

    def test_update_and_min(self):
        """Test updating and querying minimum."""
        tree = MinTree(16)

        tree.update(0, 5.0)
        tree.update(1, 3.0)
        tree.update(2, 7.0)

        assert tree.min() == 3.0

        # Update minimum
        tree.update(1, 10.0)
        assert tree.min() == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
