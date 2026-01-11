"""GPU Learner for distributed policy optimization.

The learner is responsible for:
- Sampling batches from the replay buffer
- Applying V-trace correction for stale data
- Computing gradients and updating the policy
- Publishing updated weights to the parameter server
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from aegis.algorithms.base import BaseAlgorithm
from aegis.core.types import Batch, LearnerInfo, Metrics, TrainState
from aegis.learners.vtrace import compute_log_rhos, compute_vtrace


class LearnerLocal:
    """Local learner for policy optimization.

    Runs on GPU and performs gradient-based optimization
    using data from the replay buffer.

    Attributes:
        learner_id: Unique identifier
        config: Configuration dictionary
        network: Policy network
        algorithm: RL algorithm (PPO or V-MPO)
        state: Training state (params, optimizer)
    """

    def __init__(
        self,
        learner_id: int,
        config: Dict[str, Any],
        network,
        algorithm: BaseAlgorithm,
        initial_params: Dict[str, Any],
    ):
        """Initialize the learner.

        Args:
            learner_id: Unique identifier
            config: Configuration dictionary
            network: Policy network module
            algorithm: RL algorithm instance
            initial_params: Initial policy parameters
        """
        self.learner_id = learner_id
        self.config = config
        self.network = network
        self.algorithm = algorithm

        # Training settings
        self.batch_size = config.get("batch_size", 2048)
        self.num_epochs = config.get("num_epochs", 4)
        self.minibatch_size = config.get("minibatch_size", 256)
        self.use_vtrace = config.get("use_vtrace", True)
        self.vtrace_clip_rho = config.get("vtrace_clip_rho", 1.0)
        self.vtrace_clip_c = config.get("vtrace_clip_c", 1.0)

        # Initialize optimizer
        optimizer = algorithm.create_optimizer()
        opt_state = optimizer.init(initial_params)

        # Training state
        self.state = TrainState(
            params=initial_params,
            opt_state=opt_state,
            step=0,
            policy_version=0,
            dual_params=getattr(algorithm, "init_dual_params", lambda: None)(),
        )

        # PRNG key
        self._rng = jax.random.PRNGKey(learner_id + 2000)

        # Statistics
        self.total_updates = 0
        self.total_samples = 0
        self._last_time = time.time()
        self._updates_since_last = 0

    def train_on_batch(self, batch: Batch) -> Tuple[TrainState, Metrics]:
        """Train on a single batch.

        Args:
            batch: Batch of experience data

        Returns:
            Tuple of (updated state, metrics)
        """
        # Normalize advantages
        batch = batch.normalize_advantages()

        # Run multiple epochs
        all_metrics = []

        for epoch in range(self.num_epochs):
            self._rng, shuffle_key = jax.random.split(self._rng)

            # Shuffle and iterate minibatches
            indices = jax.random.permutation(shuffle_key, batch.batch_size)
            num_minibatches = batch.batch_size // self.minibatch_size

            for mb_idx in range(num_minibatches):
                start = mb_idx * self.minibatch_size
                end = start + self.minibatch_size
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
                self.state, metrics = self.algorithm.update_step(
                    self.state, minibatch, self.network
                )
                all_metrics.append(metrics)

        # Average metrics
        avg_metrics = self._average_metrics(all_metrics)

        # Update stats
        self.total_updates += 1
        self.total_samples += batch.batch_size * self.num_epochs
        self._updates_since_last += 1

        # Increment policy version
        self.state = self.state.increment_version()

        return self.state, avg_metrics

    def compute_vtrace_advantages(
        self,
        batch: Batch,
        current_log_probs: jax.Array,
    ) -> Batch:
        """Apply V-trace correction to batch.

        Args:
            batch: Batch with old log probs and values
            current_log_probs: Log probs under current policy

        Returns:
            Batch with V-trace corrected advantages and returns
        """
        log_rhos = compute_log_rhos(current_log_probs, batch.old_log_probs)

        # Compute V-trace targets
        vtrace_targets, advantages, _ = compute_vtrace(
            rewards=batch.rewards,
            values=batch.old_values,
            bootstrap_value=batch.old_values[-1],  # Approximate
            log_rhos=log_rhos,
            dones=batch.dones,
            gamma=self.algorithm.gamma,
            clip_rho=self.vtrace_clip_rho,
            clip_c=self.vtrace_clip_c,
        )

        return Batch(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            dones=batch.dones,
            old_log_probs=batch.old_log_probs,
            old_values=batch.old_values,
            advantages=advantages,
            returns=vtrace_targets,
            policy_versions=batch.policy_versions,
            weights=batch.weights,
            indices=batch.indices,
        )

    def get_weights(self) -> Tuple[Dict[str, Any], int]:
        """Get current policy weights.

        Returns:
            Tuple of (params, version)
        """
        return self.state.params, self.state.policy_version

    def get_info(self) -> LearnerInfo:
        """Get learner statistics.

        Returns:
            LearnerInfo with current stats
        """
        now = time.time()
        elapsed = now - self._last_time
        updates_per_sec = self._updates_since_last / max(elapsed, 1e-6)
        self._last_time = now
        self._updates_since_last = 0

        return LearnerInfo(
            learner_id=self.learner_id,
            policy_version=self.state.policy_version,
            updates_completed=self.total_updates,
            samples_processed=self.total_samples,
            updates_per_second=updates_per_sec,
        )

    def _average_metrics(self, metrics_list: List[Metrics]) -> Metrics:
        """Average a list of metrics."""
        if not metrics_list:
            return Metrics(0, 0, 0, 0)

        n = len(metrics_list)
        return Metrics(
            loss=sum(m.loss for m in metrics_list) / n,
            policy_loss=sum(m.policy_loss for m in metrics_list) / n,
            value_loss=sum(m.value_loss for m in metrics_list) / n,
            entropy=sum(m.entropy for m in metrics_list) / n,
            kl_divergence=sum(m.kl_divergence for m in metrics_list) / n,
            clip_fraction=sum(m.clip_fraction for m in metrics_list) / n,
            explained_variance=sum(m.explained_variance for m in metrics_list) / n,
            grad_norm=sum(m.grad_norm for m in metrics_list) / n,
        )


if RAY_AVAILABLE:

    @ray.remote(num_gpus=1)
    class Learner:
        """Ray actor for distributed learning on GPU.

        Samples from distributed replay buffer and updates policy.
        """

        def __init__(
            self,
            learner_id: int,
            config: Dict[str, Any],
            network_config: Dict[str, Any],
            obs_shape: Tuple[int, ...],
            action_dim: int,
            continuous: bool = False,
            algorithm_name: str = "ppo",
        ):
            """Initialize the distributed learner.

            Args:
                learner_id: Unique identifier
                config: Configuration dictionary
                network_config: Network architecture config
                obs_shape: Observation shape
                action_dim: Action dimension
                continuous: Whether actions are continuous
                algorithm_name: Algorithm to use ('ppo' or 'vmpo')
            """
            import jax

            from aegis.algorithms import create_algorithm
            from aegis.networks import create_network, init_network

            self.learner_id = learner_id
            self.config = config

            # Create network
            self.network = create_network(
                observation_shape=obs_shape,
                action_dim=action_dim,
                continuous=continuous,
                config=network_config,
            )

            # Initialize parameters
            rng = jax.random.PRNGKey(learner_id + 3000)
            initial_params = init_network(self.network, obs_shape, rng)

            # Create algorithm
            algorithm = create_algorithm(algorithm_name, config)

            # Create local learner
            self._learner = LearnerLocal(
                learner_id=learner_id,
                config=config,
                network=self.network,
                algorithm=algorithm,
                initial_params=initial_params,
            )

        def train_on_batch(
            self, batch_dict: Dict[str, Any]
        ) -> Tuple[Dict[str, Any], Dict[str, float]]:
            """Train on a batch.

            Args:
                batch_dict: Batch data as dictionary (for Ray serialization)

            Returns:
                Tuple of (metrics dict, updated weights)
            """
            # Convert dict to Batch
            batch = Batch(**batch_dict)

            state, metrics = self._learner.train_on_batch(batch)

            return metrics.to_dict(), state.params

        def get_weights(self) -> Tuple[Dict[str, Any], int]:
            """Get current weights."""
            return self._learner.get_weights()

        def get_info(self) -> LearnerInfo:
            """Get learner info."""
            return self._learner.get_info()
