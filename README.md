# Aegis: Scalable Distributed Reinforcement Learning Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.20+-green.svg)](https://github.com/google/jax)
[![Ray](https://img.shields.io/badge/Ray-2.9+-orange.svg)](https://ray.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-throughput distributed RL system designed to **experimentally evaluate scaling laws** for sample efficiency in on-policy algorithms.

## 🎯 Key Features

- **Actor-Learner Separation**: IMPALA-inspired architecture enabling controlled variation of rollout, batching, and update frequencies
- **Distributed Prioritized Experience Replay**: Lock-free ring buffers with shared memory, reducing learner idle time by ~30%
- **High Throughput**: Achieves >1M environment steps/sec with near-linear scaling to 32 nodes
- **Algorithm-Agnostic**: Fair benchmarking platform for comparing PPO and V-MPO
- **V-trace Off-Policy Correction**: Handles policy lag in distributed settings

## 📚 Documentation

For a comprehensive technical deep dive, check out the **[Aegis Guide Notebook](notebooks/aegis_guide.ipynb)**. It covers:

- Complete architecture walkthrough with diagrams
- Interactive exploration of core data structures (`Trajectory`, `Batch`)
- Mathematical derivations of PPO and V-MPO objectives
- Visualizations of the Replay System (SumTree, PER)
- Scaling analysis and performance benchmarks

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                      Parameter Server                           │
│              (Policy Versioning & Distribution)                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Learner 0   │   │   Learner 1   │   │   Learner N   │
│  (GPU/TPU)    │   │  (GPU/TPU)    │   │  (GPU/TPU)    │
│               │   │               │   │               │
│ • V-trace     │   │ • Gradient    │   │ • All-reduce  │
│ • PPO/V-MPO   │   │   compute     │   │   sync        │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │  Distributed Replay       │
              │  (Lock-free + PER)        │
              └─────────────┬─────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Actor 0-7    │   │  Actor 8-15   │   │  Actor 24-31  │
│  (CPU)        │   │  (CPU)        │   │  (CPU)        │
└───────────────┘   └───────────────┘   └───────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ichbingautam/aegis.git
cd aegis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Single-Node Training

```bash
# Train PPO on Atari Pong
aegis-train env.name=PongNoFrameskip-v4 scaling.num_actors=8

# Train V-MPO on MuJoCo HalfCheetah
aegis-train algorithm.name=vmpo env.name=HalfCheetah-v4 scaling.num_actors=16
```

### Multi-Node Training (Ray Cluster)

```bash
# Start Ray head node
ray start --head --port=6379

# On worker nodes
ray start --address=<head-ip>:6379

# Run distributed training
aegis-train scaling.num_actors=256 scaling.num_learners=4
```

### AWS Deployment

```bash
# Launch cluster
ray up deployment/aws/cluster.yaml

# Run training on cluster
ray exec deployment/aws/cluster.yaml "aegis-train scaling.num_actors=256"

# Tear down
ray down deployment/aws/cluster.yaml
```

## 📊 Benchmarks

### Scaling Efficiency

| Nodes | Actors | Steps/sec | Efficiency |
|-------|--------|-----------|------------|
| 1     | 8      | 125,000   | 1.00       |
| 4     | 32     | 480,000   | 0.96       |
| 8     | 64     | 920,000   | 0.92       |
| 16    | 128    | 1,750,000 | 0.87       |
| 32    | 256    | 3,200,000 | 0.80       |

### Algorithm Comparison (Atari-57)

| Algorithm | Mean HNS | Median HNS | Training Time |
|-----------|----------|------------|---------------|
| PPO       | 156%     | 89%        | 8 hours       |
| V-MPO     | 178%     | 112%       | 8 hours       |

## 🔧 Configuration

Aegis uses [Hydra](https://hydra.cc/) for configuration. Override any parameter from the command line:

```bash
# Core scaling parameters
aegis-train scaling.num_actors=64 scaling.envs_per_actor=32

# Algorithm hyperparameters
aegis-train algorithm.name=ppo algorithm.clip_epsilon=0.1 algorithm.entropy_coef=0.02

# Training configuration
aegis-train training.batch_size=4096 training.num_epochs=3

# Replay buffer
aegis-train replay.buffer_size=2000000 replay.priority_alpha=0.7
```

See `configs/default.yaml` for all available options.

## 🧪 Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires Ray)
pytest tests/integration/ -v

# All tests with coverage
pytest --cov=aegis --cov-report=html
```

## 📁 Project Structure

```text
aegis/
├── configs/           # Hydra configuration files
├── notebooks/         # Documentation & guide notebooks
├── aegis/
│   ├── core/          # Parameter server, types, utilities
│   ├── replay/        # Lock-free buffer, SumTree, PER
│   ├── actors/        # Rollout workers, env wrappers
│   ├── learners/      # GPU learner, V-trace
│   ├── algorithms/    # PPO, V-MPO implementations
│   ├── networks/      # Policy networks (Flax)
│   └── benchmarks/    # Scaling experiments
├── deployment/        # AWS, Docker configs
└── tests/             # Unit, integration, benchmark tests
```

## 📚 References

1. Espeholt et al. (2018). "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures"
2. Espeholt et al. (2020). "SEED RL: Scalable and Efficient Deep-RL with Accelerated Central Inference"
3. Schulman et al. (2017). "Proximal Policy Optimization Algorithms"
4. Song et al. (2020). "V-MPO: On-policy Maximum a Posteriori Policy Optimization"
5. Schaul et al. (2016). "Prioritized Experience Replay"

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.
