"""Unit tests for ring buffer implementations."""

import numpy as np
import pytest

from aegis.replay.ring_buffer import RingBuffer, TrajectoryBuffer


class TestRingBuffer:
    """Tests for generic ring buffer."""

    def test_init(self):
        """Test ring buffer initialization."""
        buf = RingBuffer(100)
        assert buf.capacity == 128  # Rounded to power of 2
        assert len(buf) == 0
        assert buf.is_empty()

    def test_push_pop(self):
        """Test basic push and pop operations."""
        buf = RingBuffer(16)

        buf.push(1)
        buf.push(2)
        buf.push(3)

        assert len(buf) == 3
        assert buf.pop() == 1
        assert buf.pop() == 2
        assert buf.pop() == 3
        assert buf.is_empty()

    def test_full_buffer(self):
        """Test buffer when full."""
        buf = RingBuffer(4)  # Capacity becomes 4

        # Fill buffer (one slot reserved)
        assert buf.push(1)
        assert buf.push(2)
        assert buf.push(3)
        assert not buf.push(4)  # Should fail - buffer full

        assert buf.is_full()

    def test_push_overwrite(self):
        """Test push with overwrite."""
        buf = RingBuffer(4)

        buf.push_overwrite(1)
        buf.push_overwrite(2)
        buf.push_overwrite(3)
        overwritten = buf.push_overwrite(4)

        # First element should be overwritten
        assert overwritten == 1

    def test_peek(self):
        """Test peek without removing."""
        buf = RingBuffer(16)

        buf.push(1)
        buf.push(2)

        assert buf.peek() == 1
        assert len(buf) == 2  # Still has both elements

    def test_clear(self):
        """Test clearing buffer."""
        buf = RingBuffer(16)

        for i in range(10):
            buf.push(i)

        buf.clear()
        assert len(buf) == 0
        assert buf.is_empty()


class TestTrajectoryBuffer:
    """Tests for trajectory-specific buffer."""

    @pytest.fixture
    def buffer(self):
        """Create a trajectory buffer for testing."""
        return TrajectoryBuffer(
            capacity=100,
            obs_shape=(4,),
            action_shape=(),
            obs_dtype=np.float32,
            action_dtype=np.int32,
        )

    def test_init(self, buffer):
        """Test trajectory buffer initialization."""
        assert buffer.capacity == 100
        assert len(buffer) == 0

    def test_add_single(self, buffer):
        """Test adding a single transition."""
        idx = buffer.add(
            observation=np.ones(4),
            action=np.array(1),
            reward=1.0,
            done=False,
            log_prob=-0.5,
            value=0.5,
            policy_version=0,
        )

        assert idx == 0
        assert len(buffer) == 1

    def test_add_batch(self, buffer):
        """Test adding a batch of transitions."""
        batch_size = 10

        indices = buffer.add_batch(
            observations=np.ones((batch_size, 4)),
            actions=np.zeros(batch_size, dtype=np.int32),
            rewards=np.ones(batch_size),
            dones=np.zeros(batch_size, dtype=bool),
            log_probs=np.ones(batch_size) * -0.5,
            values=np.ones(batch_size) * 0.5,
            policy_versions=np.zeros(batch_size, dtype=np.int32),
        )

        assert len(indices) == batch_size
        assert len(buffer) == batch_size

    def test_get(self, buffer):
        """Test retrieving transitions by index."""
        # Add some data
        for i in range(5):
            buffer.add(
                observation=np.ones(4) * i,
                action=np.array(i),
                reward=float(i),
                done=False,
                log_prob=-0.5,
                value=0.5,
                policy_version=0,
            )

        # Get specific indices
        data = buffer.get(np.array([1, 3]))

        assert data['observations'].shape == (2, 4)
        assert np.allclose(data['observations'][0], np.ones(4) * 1)
        assert np.allclose(data['observations'][1], np.ones(4) * 3)

    def test_circular_behavior(self, buffer):
        """Test that buffer wraps around correctly."""
        # Overfill the buffer
        for i in range(150):
            buffer.add(
                observation=np.ones(4) * i,
                action=np.array(i % 10),
                reward=float(i),
                done=False,
                log_prob=-0.5,
                value=0.5,
                policy_version=0,
            )

        # Size should be capped at capacity
        assert len(buffer) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
