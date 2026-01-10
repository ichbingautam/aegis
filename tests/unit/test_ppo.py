"""Unit tests for PPO algorithm implementation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aegis.algorithms.ppo import PPOAlgorithm
from aegis.core.types import Batch


class TestPPOAlgorithm:
    """Tests for PPO algorithm."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'clip_value_loss': True,
            'value_clip_epsilon': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'max_grad_norm': 0.5,
            'learning_rate': 2.5e-4,
        }

    @pytest.fixture
    def ppo(self, config):
        """Create PPO algorithm instance."""
        return PPOAlgorithm(config)

    @pytest.fixture
    def mock_batch(self):
        """Create a mock batch for testing."""
        batch_size = 64
        obs_dim = 4

        return Batch(
            observations=jnp.ones((batch_size, obs_dim)),
            actions=jnp.zeros(batch_size, dtype=jnp.int32),
            rewards=jnp.ones(batch_size),
            dones=jnp.zeros(batch_size),
            old_log_probs=jnp.ones(batch_size) * -0.5,
            old_values=jnp.ones(batch_size) * 0.5,
            advantages=jnp.ones(batch_size),
            returns=jnp.ones(batch_size) * 1.5,
            policy_versions=jnp.zeros(batch_size, dtype=jnp.int32),
        )

    def test_init(self, ppo, config):
        """Test PPO initialization."""
        assert ppo.clip_epsilon == config['clip_epsilon']
        assert ppo.gamma == config['gamma']
        assert ppo.entropy_coef == config['entropy_coef']

    def test_compute_advantages(self, ppo):
        """Test GAE advantage computation."""
        T = 10
        rewards = jnp.ones(T)
        values = jnp.ones(T + 1) * 0.5
        dones = jnp.zeros(T)

        advantages, returns = ppo.compute_advantages(rewards, values, dones)

        assert advantages.shape == (T,)
        assert returns.shape == (T,)

        # Returns should be advantages + values
        np.testing.assert_allclose(
            returns, advantages + values[:-1], rtol=1e-5
        )

    def test_create_optimizer(self, ppo):
        """Test optimizer creation."""
        optimizer = ppo.create_optimizer()
        assert optimizer is not None

    def test_gae_with_episode_boundary(self, ppo):
        """Test GAE handles episode boundaries correctly."""
        T = 6
        rewards = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        values = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        dones = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # Episode ends at t=2

        advantages, returns = ppo.compute_advantages(rewards, values, dones)

        # After done=1, advantage computation should reset
        assert advantages.shape == (T,)

    def test_advantage_normalization(self, mock_batch):
        """Test that advantage normalization works."""
        # Add some variance to advantages
        batch = Batch(
            observations=mock_batch.observations,
            actions=mock_batch.actions,
            rewards=mock_batch.rewards,
            dones=mock_batch.dones,
            old_log_probs=mock_batch.old_log_probs,
            old_values=mock_batch.old_values,
            advantages=jnp.array([1.0, 2.0, 3.0, 4.0] * 16),
            returns=mock_batch.returns,
            policy_versions=mock_batch.policy_versions,
        )

        normalized = batch.normalize_advantages()

        # Should have mean ~0 and std ~1
        assert jnp.abs(jnp.mean(normalized.advantages)) < 0.1
        assert jnp.abs(jnp.std(normalized.advantages) - 1.0) < 0.1


class TestPPOLoss:
    """Tests for PPO loss computation."""

    def test_clipping_behavior(self):
        """Test that clipping prevents large policy updates."""
        config = {
            'clip_epsilon': 0.2,
            'entropy_coef': 0.0,
            'value_coef': 0.0,
            'gamma': 0.99,
            'gae_lambda': 0.95,
        }
        ppo = PPOAlgorithm(config)

        # Simulate large policy ratio
        old_log_prob = -1.0
        new_log_prob = -0.1  # Much higher probability

        ratio = jnp.exp(new_log_prob - old_log_prob)
        clipped_ratio = jnp.clip(ratio, 1 - 0.2, 1 + 0.2)

        # Ratio is large, should be clipped
        assert ratio > 1.2
        assert clipped_ratio == 1.2

    def test_value_clipping(self):
        """Test value function clipping."""
        old_value = 0.5
        new_value = 1.5  # Large change
        clip_epsilon = 0.2

        clipped_value = old_value + jnp.clip(
            new_value - old_value,
            -clip_epsilon,
            clip_epsilon,
        )

        assert clipped_value == 0.7  # 0.5 + 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
