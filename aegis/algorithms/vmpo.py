"""V-MPO (On-Policy Maximum a Posteriori Policy Optimization) algorithm.

Implements V-MPO from Song et al. (2020) with:
- Top-k advantage selection (E-step)
- KL-constrained policy fitting (M-step)
- Dual gradient descent for temperature (η) and KL multiplier (α)
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax

from aegis.algorithms.base import BaseAlgorithm
from aegis.core.types import Batch, Metrics, TrainState


class VMPOAlgorithm(BaseAlgorithm):
    """V-MPO: On-policy Maximum a Posteriori Policy Optimization.

    V-MPO frames policy optimization as expectation maximization:
    - E-step: Construct non-parametric target using top-k% advantages
    - M-step: Fit parametric policy to match target

    Key features:
    - No clipping (unlike PPO)
    - Learned temperature for weighting
    - KL constraint via Lagrangian relaxation
    - More stable in high-dimensional continuous control

    Attributes:
        top_k_fraction: Fraction of samples to use (e.g., 0.5 for top 50%)
        eta: Temperature parameter (controls advantage weighting)
        alpha: KL constraint multiplier
        eps_eta: Temperature constraint threshold
        eps_alpha: KL constraint threshold
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize V-MPO algorithm.

        Args:
            config: Configuration with V-MPO hyperparameters:
                - vmpo.top_k_fraction: Top-k% samples to use
                - vmpo.eta_init: Initial temperature
                - vmpo.alpha_init: Initial KL multiplier
                - vmpo.eps_eta: Temperature constraint
                - vmpo.eps_alpha: KL constraint
                - vmpo.dual_lr: Learning rate for dual variables
        """
        super().__init__(config)

        # V-MPO specific hyperparameters
        vmpo_config = config.get("vmpo", {})
        self.top_k_fraction = vmpo_config.get("top_k_fraction", 0.5)
        self.eta_init = vmpo_config.get("eta_init", 1.0)
        self.alpha_init = vmpo_config.get("alpha_init", 5.0)
        self.eps_eta = vmpo_config.get("eps_eta", 0.01)
        self.eps_alpha = vmpo_config.get("eps_alpha", 0.1)
        self.dual_lr = vmpo_config.get("dual_lr", 1e-2)

    def init_dual_params(self) -> dict[str, float]:
        """Initialize dual variables for V-MPO.

        Returns:
            Dictionary with 'eta' and 'alpha' parameters
        """
        return {
            "log_eta": jnp.log(self.eta_init),
            "log_alpha": jnp.log(self.alpha_init),
        }

    def loss_fn(
        self,
        params: dict[str, Any],
        batch: Batch,
        network: Any,
        dual_params: dict[str, float] = None,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute V-MPO loss.

        The loss combines:
        1. Weighted policy loss (top-k advantages)
        2. Value function loss
        3. Entropy bonus
        4. Dual losses for η and α

        Args:
            params: Network parameters
            batch: Batch of experience with advantages
            network: Actor-critic network
            dual_params: Dual variables (eta, alpha)

        Returns:
            Tuple of (total loss, auxiliary metrics)
        """
        if dual_params is None:
            dual_params = self.init_dual_params()

        # Convert log params to actual values
        eta = jnp.exp(dual_params["log_eta"])
        alpha = jnp.exp(dual_params["log_alpha"])

        # Get current policy outputs
        log_probs, entropy, values = network.apply(
            params,
            batch.observations,
            method=network.evaluate_actions,
            actions=batch.actions,
        )

        # Importance sampling weights
        if batch.weights is not None:
            weights = batch.weights
        else:
            weights = jnp.ones(batch.batch_size)

        advantages = batch.advantages

        # =====================
        # Top-k Sample Selection
        # =====================

        # Select top-k% samples by advantage
        k = int(self.top_k_fraction * batch.batch_size)
        k = max(k, 1)  # At least 1 sample

        # Get indices of top-k advantages
        top_k_indices = jnp.argsort(advantages)[-k:]
        top_k_advantages = advantages[top_k_indices]
        top_k_log_probs = log_probs[top_k_indices]
        top_k_weights = weights[top_k_indices]

        # =====================
        # Policy Loss (Weighted)
        # =====================

        # Compute softmax weights based on advantages / temperature
        advantage_weights = jax.nn.softmax(top_k_advantages / eta)

        # Weighted negative log likelihood
        policy_loss = -jnp.sum(advantage_weights * top_k_log_probs * top_k_weights)

        # =====================
        # Temperature Loss (η)
        # =====================

        # Temperature constraint: E[exp(A/η)] ≤ ε_η + 1
        # Dual loss: η * (ε_η + log(E[exp(A/η)]))
        normalized_advantages = top_k_advantages - jnp.max(
            top_k_advantages
        )  # For numerical stability
        log_exp_advantages = jax.scipy.special.logsumexp(normalized_advantages / eta) - jnp.log(k)

        eta_loss = eta * (self.eps_eta + log_exp_advantages)

        # =====================
        # KL Divergence & α Loss
        # =====================

        # KL(π_old || π_new) approximation
        log_ratio = log_probs - batch.old_log_probs
        kl_div = jnp.mean(jnp.exp(log_ratio) * log_ratio - log_ratio)

        # KL constraint: KL ≤ ε_α
        # Dual loss: α * (ε_α - KL)
        alpha_loss = alpha * (self.eps_alpha - kl_div)

        # =====================
        # Value Loss
        # =====================

        value_loss = 0.5 * jnp.mean(weights * jnp.square(values - batch.returns))

        # =====================
        # Entropy Bonus
        # =====================

        entropy_mean = jnp.mean(entropy)
        entropy_loss = -entropy_mean

        # =====================
        # Total Loss
        # =====================

        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
            + eta_loss
            - alpha_loss  # Minus because we maximize w.r.t. α
        )

        # Auxiliary metrics
        aux = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy_mean,
            "kl_divergence": kl_div,
            "eta": eta,
            "alpha": alpha,
            "eta_loss": eta_loss,
            "alpha_loss": alpha_loss,
            "top_k_advantage_mean": jnp.mean(top_k_advantages),
            "log_exp_advantages": log_exp_advantages,
        }

        return total_loss, aux

    def dual_loss_fn(
        self,
        dual_params: dict[str, float],
        advantages: jax.Array,
        kl_div: float,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute dual variable update loss.

        Args:
            dual_params: Current dual variables
            advantages: Top-k advantages from last forward pass
            kl_div: KL divergence from last forward pass

        Returns:
            Tuple of (dual loss, metrics)
        """
        eta = jnp.exp(dual_params["log_eta"])
        alpha = jnp.exp(dual_params["log_alpha"])

        k = len(advantages)

        # Temperature loss
        normalized_advantages = advantages - jnp.max(advantages)
        log_exp_advantages = jax.scipy.special.logsumexp(normalized_advantages / eta) - jnp.log(k)
        eta_loss = eta * (self.eps_eta + log_exp_advantages)

        # KL loss
        alpha_loss = alpha * (self.eps_alpha - kl_div)

        total = eta_loss - alpha_loss

        return total, {"eta_loss": eta_loss, "alpha_loss": alpha_loss}

    @partial(jax.jit, static_argnums=(0, 3))
    def update_step(
        self,
        state: TrainState,
        batch: Batch,
        network: Any,
    ) -> tuple[TrainState, Metrics]:
        """Perform one V-MPO optimization step.

        Updates both network parameters and dual variables.

        Args:
            state: Current training state (includes dual_params)
            batch: Batch of experience
            network: Actor-critic network

        Returns:
            Tuple of (updated state, training metrics)
        """
        dual_params = state.dual_params
        if dual_params is None:
            dual_params = self.init_dual_params()

        # Compute gradients for network parameters
        def network_loss(params):
            return self.loss_fn(params, batch, network, dual_params)

        grad_fn = jax.value_and_grad(network_loss, has_aux=True)
        (loss, aux), grads = grad_fn(state.params)

        # Compute gradient norm
        grad_norm = optax.global_norm(grads)

        # Update network parameters
        updates, new_opt_state = state.opt_state.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        # Update dual variables (gradient ascent for α, descent for η)
        # We use the values from aux to compute dual gradients
        k = int(self.top_k_fraction * batch.batch_size)
        top_k_indices = jnp.argsort(batch.advantages)[-k:]
        top_k_advantages = batch.advantages[top_k_indices]

        dual_grad_fn = jax.grad(
            lambda dp: self.dual_loss_fn(dp, top_k_advantages, aux["kl_divergence"])[0]
        )
        dual_grads = dual_grad_fn(dual_params)

        # Simple SGD update for dual variables
        new_dual_params = {
            "log_eta": dual_params["log_eta"] - self.dual_lr * dual_grads["log_eta"],
            "log_alpha": dual_params["log_alpha"]
            + self.dual_lr * dual_grads["log_alpha"],  # Ascent
        }

        # Clamp dual params to reasonable range
        new_dual_params["log_eta"] = jnp.clip(new_dual_params["log_eta"], -5.0, 5.0)
        new_dual_params["log_alpha"] = jnp.clip(new_dual_params["log_alpha"], -5.0, 5.0)

        # Create new state
        new_state = TrainState(
            params=new_params,
            opt_state=new_opt_state,
            step=state.step + 1,
            policy_version=state.policy_version,
            dual_params=new_dual_params,
        )

        # Create metrics
        metrics = Metrics(
            loss=float(loss),
            policy_loss=float(aux["policy_loss"]),
            value_loss=float(aux["value_loss"]),
            entropy=float(aux["entropy"]),
            kl_divergence=float(aux["kl_divergence"]),
            clip_fraction=0.0,  # V-MPO doesn't use clipping
            explained_variance=0.0,
            grad_norm=float(grad_norm),
            extra={
                "eta": float(aux["eta"]),
                "alpha": float(aux["alpha"]),
                "eta_loss": float(aux["eta_loss"]),
                "alpha_loss": float(aux["alpha_loss"]),
                "top_k_advantage_mean": float(aux["top_k_advantage_mean"]),
            },
        )

        return new_state, metrics
