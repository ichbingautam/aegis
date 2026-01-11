"""Main training script for Aegis.

Entry point for distributed RL training with Ray orchestration.
Use Hydra to configure all aspects of training.

Example:
    # Local training
    python -m aegis.train scaling.num_actors=8

    # Distributed training
    python -m aegis.train ray.address=auto scaling.num_actors=256
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_local(cfg: DictConfig) -> Dict[str, Any]:
    """Run training on a single machine without Ray.

    Useful for development and debugging.

    Args:
        cfg: Hydra configuration

    Returns:
        Dictionary of final results
    """
    import gymnasium as gym
    import jax

    from aegis.actors import RolloutWorkerLocal
    from aegis.algorithms import create_algorithm
    from aegis.core import set_seed
    from aegis.core.utils import compute_gae_numpy
    from aegis.learners import LearnerLocal
    from aegis.networks import create_network, init_network
    from aegis.replay import PrioritizedReplayBuffer

    logger.info("Starting local training...")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed
    set_seed(cfg.experiment.seed)

    # Create environment to get specs
    env = gym.make(cfg.env.name)
    obs_shape = env.observation_space.shape
    if hasattr(env.action_space, "n"):
        action_dim = env.action_space.n
        continuous = False
    else:
        action_dim = env.action_space.shape[0]
        continuous = True
    env.close()

    logger.info(f"Environment: {cfg.env.name}")
    logger.info(f"Observation shape: {obs_shape}, Action dim: {action_dim}")

    # Create network
    network = create_network(
        observation_shape=obs_shape,
        action_dim=action_dim,
        continuous=continuous,
        config=OmegaConf.to_container(cfg.network),
    )

    # Initialize parameters
    rng = jax.random.PRNGKey(cfg.experiment.seed)
    rng, init_key = jax.random.split(rng)
    initial_params = init_network(network, obs_shape, init_key)

    # Create algorithm
    algorithm = create_algorithm(
        cfg.algorithm.name,
        OmegaConf.to_container(cfg),
    )

    # Create replay buffer
    buffer = PrioritizedReplayBuffer(
        capacity=cfg.replay.buffer_size,
        obs_shape=obs_shape,
        action_shape=(action_dim,) if continuous else (),
        alpha=cfg.replay.priority_alpha,
        beta_start=cfg.replay.priority_beta_start,
    )

    # Create actors
    num_actors = min(cfg.scaling.num_actors, 8)  # Limit for local training
    actors = []
    for i in range(num_actors):
        actor = RolloutWorkerLocal(
            actor_id=i,
            config=OmegaConf.to_container(cfg),
            env_fn=lambda: gym.make(cfg.env.name),
            network=network,
            initial_params=initial_params,
        )
        actors.append(actor)

    # Create learner
    learner = LearnerLocal(
        learner_id=0,
        config=OmegaConf.to_container(cfg),
        network=network,
        algorithm=algorithm,
        initial_params=initial_params,
    )

    # Training loop
    total_steps = cfg.training.total_steps
    log_interval = cfg.logging.log_interval
    save_interval = cfg.logging.save_interval

    global_step = 0
    start_time = time.time()

    logger.info(f"Starting training for {total_steps} steps...")

    try:
        while global_step < total_steps:
            # Collect trajectories from all actors
            for actor in actors:
                rng, collect_key = jax.random.split(rng)
                trajectory = actor.collect_trajectory(collect_key)

                # Compute advantages
                advantages, returns = compute_gae_numpy(
                    rewards=np.asarray(trajectory.rewards),
                    values=np.asarray(trajectory.values),
                    dones=np.asarray(trajectory.dones),
                    gamma=cfg.algorithm.gamma,
                    gae_lambda=cfg.algorithm.gae_lambda,
                )

                # Compute priorities (|advantage| + epsilon)
                priorities = np.abs(advantages) + cfg.replay.priority_epsilon

                # Add to buffer
                buffer.add_trajectory(trajectory, priorities)

                global_step += trajectory.length

            # Train if buffer has enough data
            if len(buffer) >= cfg.training.batch_size:
                batch, indices = buffer.sample(cfg.training.batch_size)

                # Compute GAE for sampled batch
                batch.advantages = (batch.advantages - np.mean(batch.advantages)) / (
                    np.std(batch.advantages) + 1e-8
                )
                batch.returns = batch.old_values + batch.advantages

                # Convert to JAX
                import jax.numpy as jnp

                batch = batch._replace(
                    **{
                        k: jnp.asarray(v)
                        for k, v in batch.__dict__.items()
                        if isinstance(v, np.ndarray)
                    }
                )

                # Train
                state, metrics = learner.train_on_batch(batch)

                # Update actor policies
                params, version = learner.get_weights()
                for actor in actors:
                    actor.update_policy(params, version)

            # Logging
            if global_step % log_interval < num_actors * cfg.rollout.trajectory_length:
                elapsed = time.time() - start_time
                sps = global_step / elapsed

                # Get actor stats
                actor_info = actors[0].get_info()

                logger.info(
                    f"Step {global_step:,} | "
                    f"SPS: {sps:.0f} | "
                    f"Return: {actor_info.mean_episode_return:.1f} | "
                    f"Length: {actor_info.mean_episode_length:.0f}"
                )

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    finally:
        # Cleanup
        for actor in actors:
            actor.close()

    elapsed = time.time() - start_time
    logger.info(f"Training complete! Total time: {elapsed:.1f}s, Steps: {global_step:,}")

    return {
        "total_steps": global_step,
        "elapsed_time": elapsed,
        "steps_per_second": global_step / elapsed,
        "final_return": actors[0].get_info().mean_episode_return,
    }


def train_distributed(cfg: DictConfig) -> Dict[str, Any]:
    """Run distributed training with Ray.

    Args:
        cfg: Hydra configuration

    Returns:
        Dictionary of final results
    """
    import ray

    logger.info("Starting distributed training with Ray...")

    # Initialize Ray
    ray_address = cfg.ray.get("address", "auto")
    if ray_address == "auto":
        ray.init()
    else:
        ray.init(address=ray_address)

    logger.info(f"Ray cluster: {ray.cluster_resources()}")

    # TODO: Implement full distributed training loop
    # This would create remote actors, learners, and parameter servers

    ray.shutdown()

    return {"status": "distributed training not yet implemented"}


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for training.

    Args:
        cfg: Hydra configuration
    """
    # Check if we should use Ray
    use_ray = cfg.ray.get("address") is not None

    if use_ray:
        results = train_distributed(cfg)
    else:
        results = train_local(cfg)

    logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()
