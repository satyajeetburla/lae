#!/usr/bin/env python3
"""Reset and step the quadrotor environment with no LAE artifacts."""

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "quad-swarm-rl"
sys.path.insert(0, str(PACKAGE_ROOT))

from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti


def main():
    env = QuadrotorEnvMulti(
        num_agents=2,
        ep_time=0.2,
        rew_coeff=None,
        obs_repr="xyz_vxyz_R_omega",
        neighbor_visible_num=1,
        neighbor_obs_type="pos_vel",
        collision_hitbox_radius=2.0,
        collision_falloff_radius=-1.0,
        use_obstacles=False,
        obst_density=0.2,
        obst_size=1.0,
        obst_spawn_area=[6.0, 6.0],
        use_downwash=False,
        use_numba=False,
        quads_mode="static_same_goal",
        room_dims=[10.0, 10.0, 10.0],
        use_replay_buffer=False,
        quads_view_mode=["topdown"],
        quads_render=False,
        dynamics_params="Crazyflie",
        raw_control=True,
        raw_control_zero_middle=True,
        dynamics_randomize_every=None,
        dynamics_change=dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0)),
        dyn_sampler_1=None,
        sense_noise="default",
        init_random_state=False,
        init_mode="random",
        seed=1,
    )

    obs = env.reset()
    actions = np.stack([env.action_space.sample() for _ in range(env.num_agents)])
    next_obs, rewards, dones, infos = env.step(actions)

    print(f"reset obs shape={tuple(obs.shape)}")
    print(f"step obs shape={tuple(next_obs.shape)}")
    print(f"reward shape={tuple(np.asarray(rewards).shape)}")
    print(f"done shape={tuple(np.asarray(dones).shape)}")
    print(f"infos={len(infos)}")


if __name__ == "__main__":
    main()
