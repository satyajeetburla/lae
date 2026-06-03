# Latent Activation Editing (LAE)

Official repository for the paper:

**Latent Activation Editing: Inference-Time Refinement of Learned Policies for Safer Multirobot Navigation**

[Project Page](https://lae-robotics.github.io) | [arXiv](https://arxiv.org/abs/2509.20623)


## Simulator Setup

Before running the GRU-LAE experiments, install and verify the QuadSwarm
simulator by following the instructions in:

https://github.com/satyajeetburla/quadswarm-latent/tree/main/quad-swarm-rl

Make sure the simulator is working correctly before proceeding.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate quadswarm-latent
```

## Required Local Files

This repository tracks only code and configuration files. The policy
checkpoints, GRU-LAE checkpoints, and evaluation datasets should be added
locally.

Expected files:

```text
outputs/train_dir/<experiment>/config.json
outputs/train_dir/<experiment>/checkpoint_p0/*.pth

path/to/classifier_checkpoint.pth
path/to/lae_gru_checkpoint.pt

path/to/initial_conditions.npy
```

Set the paths used by the examples:

```bash
export TRAIN_DIR=outputs/train_dir
export EXPERIMENT=02_sim2real_obst_density_see_1111_q.o.den_0.2
export DEVICE=gpu
export DATASET=path/to/initial_conditions.npy
```



## Evaluation

The evaluation scripts support two initialization modes:

* `dataset_seq` — deterministic evaluation using a fixed dataset of initial conditions.
* `random` — evaluation using randomly generated initial conditions.

### Base RL Policy (Dataset Initialization)

```bash
PYTHONPATH=$PWD/quad-swarm-rl python -m swarm_rl.enjoy \
  --env=quadrotor_multi \
  --train_dir="$TRAIN_DIR" \
  --experiment="$EXPERIMENT" \
  --device="$DEVICE" \
  --rnn_size=10 \
  --sae_encoder_mode=0 \
  --quads_init_mode=dataset_seq \
  --quads_init_dataset_path "$DATASET" \
  --eval_deterministic=True \
  --replay_buffer_sample_prob=0.0 \
  --no_render \
  --max_num_episodes=1
```

### GRU-LAE (Dataset Initialization)

```bash
scripts/run_paper_gru_eval.sh \
  --quads_init_mode=dataset_seq \
  --quads_init_dataset_path "$DATASET" \
  --eval_deterministic=True \
  --replay_buffer_sample_prob=0.0 \
  --no_render \
  --max_num_episodes=1
```

### GRU-LAE (Random Initialization)

```bash
scripts/run_paper_gru_eval.sh \
  --quads_init_mode=random \
  --eval_deterministic=True \
  --replay_buffer_sample_prob=0.0 \
  --no_render \
  --max_num_episodes=1
```

## Release Status

This repository is an initial research release of the LAE experiment code.
It includes the core code and configuration for the paper setup. Additional
artifacts, documentation, and examples are planned for
future updates.

