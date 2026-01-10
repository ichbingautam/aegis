"""Proximal Policy Optimization (PPO) algorithm.

Implements the PPO-Clip algorithm from Schulman et al. (2017) with:
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Value function clipping (optional)
- Entropy bonus for exploration
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from aegis.algorithms.base import BaseAlgorithm
from aegis.core.types import Batch, Metrics, TrainState


class PPOAlgorithm(BaseAlgorithm):
    """Proximal Policy Optimization algorithm.

    PPO prevents destructively large policy updates by clipping the
    probability ratio between new and old policies. This enables:
    - Multiple epochs of minibatch updates per data collection
    - Stable learning across diverse environments
    - Simple hyperparameter tuning

    Attributes:
        clip_epsilon: Clipping range for policy ratio
        clip_value_loss: Whether to clip value function loss
        value_clip_epsilon: Clipping range for value function
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize PPO algorithm.

        Args:
            config: Configuration with PPO hyperparameters:
                - clip_epsilon: Policy clipping range (default: 0.2)
                - clip_value_loss: Enable value clipping (default: True)
                - value_clip_epsilon: Value clipping range (default: 0.2)
        """
        super().__init__(config)

        # PPO-specific hyperparameters
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.clip_value_loss = config.get('clip_value_loss', True)
        self.value_clip_epsilon = config.get('value_clip_epsilon', 0.2)

    def loss_fn(
        self,
        params: Dict[str, Any],
        batch: Batch,
        network: Any,
    ) -> Tuple[jax.Array, Dict[str, Any]]:
        """Compute PPO loss.

        The loss has three components:
        1. Clipped policy loss (actor)
        2. Value function loss (critic)
        3. Entropy bonus (exploration)

        L = L_clip - c1 * L_vf + c2 * entropy

        Args:
            params: Network parameters
            batch: Batch of experience with advantages
            network: Actor-critic network

        Returns:
            Tuple of (total loss, auxiliary metrics)
        """
        # Get current policy outputs
        log_probs, entropy, values = network.apply(
            params,
            batch.observations,
            method=network.evaluate_actions,
            actions=batch.actions,
        )

        # Normalize advantages (per-minibatch)
        advantages = batch.advantages
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

        # Apply importance sampling weights if available
        if batch.weights is not None:
            weights = batch.weights
        else:
            weights = jnp.ones_like(advantages)

        # =====================
        # Policy Loss (Clipped)
        # =====================

        # Probability ratio: π(a|s) / π_old(a|s)
        log_ratio = log_probs - batch.old_log_probs
        ratio = jnp.exp(log_ratio)

        # Clipped ratio
        clipped_ratio = jnp.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        # Policy loss: min(ratio * A, clip(ratio) * A)
        policy_loss_unclipped = ratio * advantages
        policy_loss_clipped = clipped_ratio * advantages
        policy_loss = -jnp.mean(
            weights * jnp.minimum(policy_loss_unclipped, policy_loss_clipped)
        )

        # Clip fraction (for monitoring)
        clip_fraction = jnp.mean(jnp.abs(ratio - 1.0) > self.clip_epsilon)

        # KL divergence approximation
        approx_kl = jnp.mean((ratio - 1) - log_ratio)

        # ====================
        # Value Loss (MSE)
        # ====================

        if self.clip_value_loss:
            # Clipped value loss
            values_clipped = batch.old_values + jnp.clip(
                values - batch.old_values,
                -self.value_clip_epsilon,
                self.value_clip_epsilon,
            )
            value_loss_unclipped = jnp.square(values - batch.returns)
            value_loss_clipped = jnp.square(values_clipped - batch.returns)
            value_loss = 0.5 * jnp.mean(
                weights * jnp.maximum(value_loss_unclipped, value_loss_clipped)
            )
        else:
            value_loss = 0.5 * jnp.mean(weights * jnp.square(values - batch.returns))

        # ====================
        # Entropy Bonus
        # ====================

        entropy_loss = -jnp.mean(entropy)

        # ====================
        # Total Loss
        # ====================

        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        # Auxiliary metrics
        aux = {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': -entropy_loss,  # Positive entropy
            'kl_divergence': approx_kl,
            'clip_fraction': clip_fraction,
            'ratio_mean': jnp.mean(ratio),
            'ratio_std': jnp.std(ratio),
            'value_pred_mean': jnp.mean(values),
            'advantage_mean': jnp.mean(advantages),
        }

        return total_loss, aux

    @partial(jax.jit, static_argnums=(0, 3))
    def update_step(
        self,
        state: TrainState,
        batch: Batch,
        network: Any,
    ) -> Tuple[TrainState, Metrics]:
        """Perform one PPO optimization step.

        Args:
            state: Current training state
            batch: Batch of experience
            network: Actor-critic network

        Returns:
            Tuple of (updated state, training metrics)
        """
        # Compute gradients
        grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(state.params, batch, network)

        # Compute gradient norm before clipping
        grad_norm = optax.global_norm(grads)

        # Apply optimizer update
        updates, new_opt_state = state.opt_state.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        # Compute explained variance
        y_pred = aux.get('value_pred_mean', 0.0)
        explained_var = 1 - jnp.var(batch.returns - batch.old_values) / (jnp.var(batch.returns) + 1e-8)

        # Create new state
        new_state = TrainState(
            params=new_params,
            opt_state=new_opt_state,
            step=state.step + 1,
            policy_version=state.policy_version,
            dual_params=state.dual_params,
        )

        # Create metrics
        metrics = Metrics(
            loss=float(loss),
            policy_loss=float(aux['policy_loss']),
            value_loss=float(aux['value_loss']),
            entropy=float(aux['entropy']),
            kl_divergence=float(aux['kl_divergence']),
            clip_fraction=float(aux['clip_fraction']),
            explained_variance=float(explained_var),
            grad_norm=float(grad_norm),
            extra={
                'ratio_mean': float(aux['ratio_mean']),
                'ratio_std': float(aux['ratio_std']),
            },
        )

        return new_state, metrics

    def train_epoch(
        self,
        state: TrainState,
        batch: Batch,
        network: Any,
        minibatch_size: int,
        key: jax.Array,
    ) -> Tuple[TrainState, Metrics]:
        """Train for one epoch over the batch.

        Shuffles data and iterates over minibatches.

        Args:
            state: Current training state
            batch: Full batch of experience
            network: Actor-critic network
            minibatch_size: Size of each minibatch
            key: PRNG key for shuffling

        Returns:
            Tuple of (final state, average metrics)
        """
        batch_size = batch.batch_size
        num_minibatches = batch_size // minibatch_size

        # Shuffle indices
        indices = jax.random.permutation(key, batch_size)

        # Accumulate metrics
        total_metrics = None

        for i in range(num_minibatches):
            start = i * minibatch_size
            end = start + minibatch_size
            mb_indices = indices[start:end]

            # Create minibatch
            minibatch = Batch(
                observations=batch.observations[mb_indices],
                actions=batch.actions[mb_indices],
                rewards=batch.rewards[mb_indices],
                dones=batch.dones[mb_indices],
                old_log_probs=batch.old_log_probs[mb_indices],
                old_values=batch.old_values[mb_indices],
                advantages=batch.advantages[mb_indices],
                returns=batch.returns[mb_indices],
                policy_versions=batch.policy_versions[mb_indices],
                weights=batch.weights[mb_indices] if batch.weights is not None else None,
                indices=batch.indices[mb_indices] if batch.indices is not None else None,
            )

            # Update
            state, metrics = self.update_step(state, minibatch, network)

            # Accumulate metrics
            if total_metrics is None:
                total_metrics = metrics
            else:
                # Average metrics
                total_metrics = Metrics(
                    loss=(total_metrics.loss + metrics.loss) / 2,
                    policy_loss=(total_metrics.policy_loss + metrics.policy_loss) / 2,
                    value_loss=(total_metrics.value_loss + metrics.value_loss) / 2,
                    entropy=(total_metrics.entropy + metrics.entropy) / 2,
                    kl_divergence=(total_metrics.kl_divergence + metrics.kl_divergence) / 2,
                    clip_fraction=(total_metrics.clip_fraction + metrics.clip_fraction) / 2,
                    explained_variance=(total_metrics.explained_variance + metrics.explained_variance) / 2,
                    grad_norm=(total_metrics.grad_norm + metrics.grad_norm) / 2,
                )

        return state, total_metrics
