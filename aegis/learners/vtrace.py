"""V-trace off-policy correction for distributed RL.

Implements V-trace from Espeholt et al. (2018) IMPALA paper:
- Handles policy lag between actors and learner
- Truncated importance sampling for variance reduction
- Compatible with both PPO and V-MPO
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(5, 6))
def compute_vtrace(
    rewards: jax.Array,
    values: jax.Array,
    bootstrap_value: jax.Array,
    log_rhos: jax.Array,
    dones: jax.Array,
    gamma: float = 0.99,
    lambda_: float = 1.0,
    clip_rho: float = 1.0,
    clip_c: float = 1.0,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute V-trace targets and advantages.

    V-trace corrects for off-policy data by truncating importance
    sampling ratios, providing stable learning despite policy lag.

    The V-trace target is:
        v_s = V(s) + Σ_{t=s}^{s+n-1} γ^{t-s} (Π_{i=s}^{t-1} c_i) δ_t

    where:
        δ_t = ρ_t (r_t + γ V(s_{t+1}) - V(s_t))
        ρ_t = min(ρ̄, π(a_t|s_t) / μ(a_t|s_t))
        c_t = min(c̄, π(a_t|s_t) / μ(a_t|s_t))

    Args:
        rewards: Reward array [T]
        values: Value estimates [T] (not including bootstrap)
        bootstrap_value: Value for terminal state (scalar or [])
        log_rhos: Log importance ratios log(π/μ) [T]
        dones: Done flags [T]
        gamma: Discount factor
        lambda_: Trace decay (typically 1.0 for V-trace)
        clip_rho: ρ̄ clipping threshold
        clip_c: c̄ clipping threshold

    Returns:
        Tuple of (vtrace_targets, advantages, clipped_rhos)
    """
    T = len(rewards)

    # Compute importance sampling ratios
    rhos = jnp.exp(log_rhos)

    # Clip ratios
    clipped_rhos = jnp.minimum(rhos, clip_rho)
    clipped_cs = jnp.minimum(rhos, clip_c)

    # Compute TD errors: δ_t = ρ_t (r_t + γ V_{t+1} (1-d_t) - V_t)
    # Append bootstrap value for indexing
    values_plus_bootstrap = jnp.concatenate([values, jnp.array([bootstrap_value])])

    deltas = clipped_rhos * (
        rewards
        + gamma * values_plus_bootstrap[1:] * (1 - dones)
        - values
    )

    # Compute V-trace targets using reverse scan
    def vtrace_step(carry, t):
        """Single V-trace accumulation step (reverse order)."""
        next_vtrace = carry
        t_rev = T - 1 - t

        # Current V-trace increment
        # v_t = V_t + δ_t + γ c_t (1 - d_t) (v_{t+1} - V_{t+1})
        vtrace = (
            deltas[t_rev]
            + gamma * lambda_ * clipped_cs[t_rev] * (1 - dones[t_rev]) * next_vtrace
        )

        return vtrace, vtrace

    # Scan in reverse (from T-1 to 0)
    _, vtrace_increments_rev = jax.lax.scan(
        vtrace_step,
        0.0,  # Initial carry (next vtrace increment)
        jnp.arange(T),
    )

    # Reverse back to correct order
    vtrace_increments = vtrace_increments_rev[::-1]

    # V-trace targets = V + increments
    vtrace_targets = values + vtrace_increments

    # Advantages for policy gradient
    # A = ρ * (r + γ V_{t+1} - V_t) + γ c * (v_{t+1} - V_{t+1})
    # Simplified: A = v_t - V_t (where v_t is vtrace target)
    advantages = clipped_rhos * (vtrace_targets - values)

    return vtrace_targets, advantages, clipped_rhos


@partial(jax.jit, static_argnums=(5,))
def compute_vtrace_batch(
    rewards: jax.Array,
    values: jax.Array,
    bootstrap_values: jax.Array,
    log_rhos: jax.Array,
    dones: jax.Array,
    gamma: float = 0.99,
    clip_rho: float = 1.0,
    clip_c: float = 1.0,
) -> Tuple[jax.Array, jax.Array]:
    """Compute V-trace for a batch of trajectories.

    Vectorized version for processing multiple trajectories.

    Args:
        rewards: Reward array [B, T]
        values: Value estimates [B, T]
        bootstrap_values: Bootstrap values [B]
        log_rhos: Log importance ratios [B, T]
        dones: Done flags [B, T]
        gamma: Discount factor
        clip_rho: ρ̄ clipping threshold
        clip_c: c̄ clipping threshold

    Returns:
        Tuple of (vtrace_targets [B, T], advantages [B, T])
    """
    # vmap over batch dimension
    vtrace_fn = partial(
        compute_vtrace,
        gamma=gamma,
        clip_rho=clip_rho,
        clip_c=clip_c,
    )

    vtrace_targets, advantages, _ = jax.vmap(vtrace_fn)(
        rewards, values, bootstrap_values, log_rhos, dones
    )

    return vtrace_targets, advantages


def should_use_vtrace(
    behavior_policy_version: int,
    current_policy_version: int,
    max_lag: int = 3,
) -> bool:
    """Determine if V-trace correction should be applied.

    V-trace is needed when the policy that generated the data
    differs significantly from the current policy.

    Args:
        behavior_policy_version: Version that generated the data
        current_policy_version: Current learner policy version
        max_lag: Maximum allowed policy lag before V-trace kicks in

    Returns:
        True if V-trace correction should be applied
    """
    lag = current_policy_version - behavior_policy_version
    return lag > max_lag


@jax.jit
def compute_log_rhos(
    current_log_probs: jax.Array,
    behavior_log_probs: jax.Array,
) -> jax.Array:
    """Compute log importance sampling ratios.

    Args:
        current_log_probs: Log probs under current policy
        behavior_log_probs: Log probs under behavior policy

    Returns:
        Log ratios: log(π_current / π_behavior)
    """
    return current_log_probs - behavior_log_probs
