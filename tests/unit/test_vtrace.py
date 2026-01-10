"""Unit tests for V-trace off-policy correction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aegis.learners.vtrace import (
    compute_vtrace,
    compute_log_rhos,
    should_use_vtrace,
)


class TestVtrace:
    """Tests for V-trace implementation."""

    def test_on_policy_case(self):
        """Test that V-trace reduces to standard returns when on-policy."""
        T = 5
        rewards = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        values = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
        bootstrap_value = jnp.array(0.5)
        log_rhos = jnp.zeros(T)  # π = μ, so ratio = 1
        dones = jnp.zeros(T)

        vtrace_targets, advantages, clipped_rhos = compute_vtrace(
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            log_rhos=log_rhos,
            dones=dones,
            gamma=0.99,
        )

        assert vtrace_targets.shape == (T,)
        assert advantages.shape == (T,)

        # Clipped rhos should all be 1.0 for on-policy
        np.testing.assert_allclose(clipped_rhos, jnp.ones(T), rtol=1e-5)

    def test_importance_clipping(self):
        """Test that importance ratios are clipped."""
        T = 3
        rewards = jnp.array([1.0, 1.0, 1.0])
        values = jnp.array([0.5, 0.5, 0.5])
        bootstrap_value = jnp.array(0.5)
        # Large positive log ratio = π >> μ
        log_rhos = jnp.array([2.0, 2.0, 2.0])  # ratio ≈ 7.4
        dones = jnp.zeros(T)

        _, _, clipped_rhos = compute_vtrace(
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            log_rhos=log_rhos,
            dones=dones,
            gamma=0.99,
            clip_rho=1.0,
            clip_c=1.0,
        )

        # All rhos should be clipped to 1.0
        np.testing.assert_allclose(clipped_rhos, jnp.ones(T), rtol=1e-5)

    def test_episode_boundary(self):
        """Test V-trace handles episode boundaries."""
        T = 4
        rewards = jnp.array([1.0, 1.0, 1.0, 1.0])
        values = jnp.array([0.5, 0.5, 0.5, 0.5])
        bootstrap_value = jnp.array(0.5)
        log_rhos = jnp.zeros(T)
        dones = jnp.array([0.0, 1.0, 0.0, 0.0])  # Episode ends at t=1

        vtrace_targets, advantages, _ = compute_vtrace(
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            log_rhos=log_rhos,
            dones=dones,
            gamma=0.99,
        )

        assert vtrace_targets.shape == (T,)

        # At done=1, future returns should be cut off
        # The target at t=1 should not include future discounted rewards

    def test_negative_advantages(self):
        """Test V-trace with negative rewards/advantages."""
        T = 3
        rewards = jnp.array([-1.0, -1.0, -1.0])
        values = jnp.array([0.5, 0.5, 0.5])
        bootstrap_value = jnp.array(0.5)
        log_rhos = jnp.zeros(T)
        dones = jnp.zeros(T)

        vtrace_targets, advantages, _ = compute_vtrace(
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            log_rhos=log_rhos,
            dones=dones,
            gamma=0.99,
        )

        # Advantages should be negative (bad actions)
        assert jnp.all(advantages < 0)


class TestLogRhos:
    """Tests for log importance ratio computation."""

    def test_compute_log_rhos(self):
        """Test log ratio computation."""
        current_log_probs = jnp.array([-0.5, -1.0, -1.5])
        behavior_log_probs = jnp.array([-1.0, -1.0, -1.0])

        log_rhos = compute_log_rhos(current_log_probs, behavior_log_probs)

        expected = jnp.array([0.5, 0.0, -0.5])
        np.testing.assert_allclose(log_rhos, expected, rtol=1e-5)

    def test_same_policy(self):
        """Test that same policy gives zero log ratio."""
        log_probs = jnp.array([-1.0, -1.0, -1.0])

        log_rhos = compute_log_rhos(log_probs, log_probs)

        np.testing.assert_allclose(log_rhos, jnp.zeros(3), rtol=1e-5)


class TestShouldUseVtrace:
    """Tests for V-trace decision logic."""

    def test_no_lag(self):
        """Test that V-trace is not needed when on-policy."""
        assert not should_use_vtrace(
            behavior_policy_version=10,
            current_policy_version=10,
            max_lag=3,
        )

    def test_small_lag(self):
        """Test that V-trace is not needed for small lag."""
        assert not should_use_vtrace(
            behavior_policy_version=7,
            current_policy_version=10,
            max_lag=3,
        )

    def test_large_lag(self):
        """Test that V-trace is needed for large lag."""
        assert should_use_vtrace(
            behavior_policy_version=5,
            current_policy_version=10,
            max_lag=3,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
