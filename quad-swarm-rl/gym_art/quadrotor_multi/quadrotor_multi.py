import copy
import time
from collections import deque
from copy import deepcopy

import gym
import numpy as np
import random
from gym_art.quadrotor_multi.aerodynamics.downwash import perform_downwash
from gym_art.quadrotor_multi.collisions.obstacles import perform_collision_with_obstacle
from gym_art.quadrotor_multi.collisions.quadrotors import calculate_collision_matrix, \
    calculate_drone_proximity_penalties, perform_collision_between_drones
from gym_art.quadrotor_multi.collisions.room import perform_collision_with_wall, perform_collision_with_ceiling
from gym_art.quadrotor_multi.obstacles.utils import get_cell_centers
from gym_art.quadrotor_multi.quad_utils import QUADS_OBS_REPR, QUADS_NEIGHBOR_OBS_TYPE

from gym_art.quadrotor_multi.obstacles.obstacles import MultiObstacles
from gym_art.quadrotor_multi.quadrotor_multi_visualization import Quadrotor3DSceneMulti
from gym_art.quadrotor_multi.quadrotor_single import QuadrotorSingle
from gym_art.quadrotor_multi.scenarios.mix import create_scenario

# Data Logging
from swarm_rl.env_wrappers.logger_manager import get_logger
import os
import imageio.v3 as iio
import imageio


from datetime import datetime
import imageio_ffmpeg

class QuadrotorEnvMulti(gym.Env):
    def __init__(self, num_agents, ep_time, rew_coeff, obs_repr,
                 # Neighbor
                 neighbor_visible_num, neighbor_obs_type, collision_hitbox_radius, collision_falloff_radius,

                 # Obstacle
                 use_obstacles, obst_density, obst_size, obst_spawn_area,

                 # Aerodynamics, Numba Speed Up, Scenarios, Room, Replay Buffer, Rendering
                 use_downwash, use_numba, quads_mode, room_dims, use_replay_buffer, quads_view_mode,
                 quads_render,

                 # Quadrotor Specific (Do Not Change)
                 dynamics_params, raw_control, raw_control_zero_middle,
                 dynamics_randomize_every, dynamics_change, dyn_sampler_1,
                 sense_noise, init_random_state,
                 init_mode="random", init_dataset_path=None, custom_init_indices=None,
                 seed=None, print_dataset_id=False):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.fixed = True
        # Predefined Parameters
        self.num_agents = num_agents
        self.obs_save = None
        self.init_pos_save = None
        self.deterministic = True
        obs_self_size = QUADS_OBS_REPR[obs_repr]
        if neighbor_visible_num == -1:
            self.num_use_neighbor_obs = self.num_agents - 1
        else:
            self.num_use_neighbor_obs = neighbor_visible_num
        self.obst_index = None
        self.obst_map = None
        self.obst_pos_arr = []
        # Set to True means that sample_factory will treat it as a multi-agent vectorized environment even with
        # num_agents=1. More info, please look at sample-factory: envs/quadrotors/wrappers/reward_shaping.py
        self.is_multiagent = True
        self.room_dims = room_dims
        self.quads_view_mode = quads_view_mode

        # Generate All Quadrotors
        self.envs = []
        for i in range(self.num_agents):
            e = QuadrotorSingle(
                # Quad Parameters
                dynamics_params=dynamics_params, dynamics_change=dynamics_change,
                dynamics_randomize_every=dynamics_randomize_every, dyn_sampler_1=dyn_sampler_1,
                raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle, sense_noise=sense_noise,
                init_random_state=init_random_state, obs_repr=obs_repr, ep_time=ep_time, room_dims=room_dims,
                use_numba=use_numba,
                # Neighbor
                num_agents=num_agents,
                neighbor_obs_type=neighbor_obs_type, num_use_neighbor_obs=self.num_use_neighbor_obs,
                # Obstacle
                use_obstacles=use_obstacles,
            )
            if seed is not None:
                e._seed(seed + i)
            self.envs.append(e)

        # Set Obs & Act
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        # Aux variables
        self.quad_arm = self.envs[0].dynamics.arm
        self.control_freq = self.envs[0].control_freq
        self.control_dt = 1.0 / self.control_freq
        self.pos = np.zeros([self.num_agents, 3])
        self.vel = np.zeros([self.num_agents, 3])
        self.acc = np.zeros([self.num_agents, 3])
        self.omega = np.zeros([self.num_agents, 3])
        self.rel_pos = np.zeros((self.num_agents, self.num_agents, 3))
        self.rel_vel = np.zeros((self.num_agents, self.num_agents, 3))

        # Reward
        self.rew_coeff = dict(
            pos=1., effort=0.05, action_change=0., crash=1., orient=1., yaw=0., rot=0., attitude=0., spin=0.1, vel=0.,
            quadcol_bin=5., quadcol_bin_smooth_max=4., quadcol_bin_obst=5.
        )
        rew_coeff_orig = copy.deepcopy(self.rew_coeff)

        if rew_coeff is not None:
            assert isinstance(rew_coeff, dict)
            assert set(rew_coeff.keys()).issubset(set(self.rew_coeff.keys()))
            self.rew_coeff.update(rew_coeff)
        for key in self.rew_coeff.keys():
            self.rew_coeff[key] = float(self.rew_coeff[key])

        orig_keys = list(rew_coeff_orig.keys())
        # Checking to make sure we didn't provide some false rew_coeffs (for example by misspelling one of the params)
        assert np.all([key in orig_keys for key in self.rew_coeff.keys()])

        # Neighbors
        neighbor_obs_size = QUADS_NEIGHBOR_OBS_TYPE[neighbor_obs_type]

        self.clip_neighbor_space_length = self.num_use_neighbor_obs * neighbor_obs_size
        self.clip_neighbor_space_min_box = self.observation_space.low[
                                           obs_self_size:obs_self_size + self.clip_neighbor_space_length]
        self.clip_neighbor_space_max_box = self.observation_space.high[
                                           obs_self_size:obs_self_size + self.clip_neighbor_space_length]

        # Obstacles
        self.use_obstacles = use_obstacles
        self.obstacles = None
        self.num_obstacles = 0
        if self.use_obstacles:
            self.prev_obst_quad_collisions = []
            self.obst_quad_collisions_per_episode = 0
            self.obst_quad_collisions_after_settle = 0
            self.curr_quad_col = []
            self.obst_density = obst_density
            self.obst_spawn_area = obst_spawn_area
            self.num_obstacles = int(obst_density * obst_spawn_area[0] * obst_spawn_area[1])
            self.obst_map = None
            self.obst_size = obst_size

            # Log more info
            self.distance_to_goal_3_5 = 0
            self.distance_to_goal_5 = 0
            self.obst_pos_arr = None



        # Scenarios
        self.quads_mode = quads_mode
        self.scenario = create_scenario(quads_mode=quads_mode, envs=self.envs, num_agents=num_agents,
                                        room_dims=room_dims)

        # Collisions
        # # Collisions: Neighbors
        self.collisions_per_episode = 0
        # # # Ignore collisions because of spawn
        self.collisions_after_settle = 0
        self.collisions_grace_period_steps = 1.5 * self.control_freq
        self.collisions_grace_period_seconds = 1.5
        self.prev_drone_collisions = []

        self.collisions_final_grace_period_steps = 5.0 * self.control_freq
        self.collisions_final_5s = 0

        # # # Dense reward info
        self.collision_threshold = collision_hitbox_radius * self.quad_arm
        self.collision_falloff_threshold = collision_falloff_radius * self.quad_arm

        # # Collisions: Room
        self.collisions_room_per_episode = 0
        self.collisions_floor_per_episode = 0
        self.collisions_wall_per_episode = 0
        self.collisions_ceiling_per_episode = 0

        self.prev_crashed_walls = []
        self.prev_crashed_ceiling = []
        self.prev_crashed_room = []

        # Replay
        self.use_replay_buffer = use_replay_buffer
        # # only start using the buffer after the drones learn how to fly
        self.activate_replay_buffer = False
        # # since the same collisions happen during replay, we don't want to keep resaving the same event
        self.saved_in_replay_buffer = False
        self.last_step_unique_collisions = False
        self.crashes_in_recent_episodes = deque([], maxlen=100)
        self.crashes_last_episode = 0

        # Numba
        self.use_numba = use_numba

        # Aerodynamics
        self.use_downwash = use_downwash

        # Rendering
        # # set to true whenever we need to reset the OpenGL scene in render()
        self.quads_render = quads_render
        self.scenes = []
        if self.quads_render:
            self.reset_scene = False
            self.simulation_start_time = 0
            self.frames_since_last_render = self.render_skip_frames = 0
            self.render_every_nth_frame = 1
            # # Use this to control rendering speed
            self.render_speed = 1.0
            self.quads_formation_size = 2.0
            self.all_collisions = {}

        # Log
        self.distance_to_goal = [[] for _ in range(len(self.envs))]
        self.reached_goal = [False for _ in range(len(self.envs))]

        # Log metric
        self.agent_col_agent = np.ones(self.num_agents)
        self.agent_col_obst = np.ones(self.num_agents)

        # Others
        self.apply_collision_force = True

        # Aux
        self._record_enabled = False  # call enable_recording(...) to turn on
        self._writers = None  # list of imageio writers, one per scene
        self._record_fps = 30
        self._record_prefix = "scene"  # filename prefix
        self._record_dir = None
        self._record_traj_tag = None  # e.g., "traj_0123" based on init_pos_index
        self._record_refresh_needed = False  # set True when episode/trajectory changes

        self.init_mode = init_mode
        self.print_dataset_id = bool(print_dataset_id)
        self.init_pos_dataset = []
        self.init_pos_index = 0
        self.max_init_pos = 0
        self.det_init_pos = False  # default, will be updated in reset()

        if self.init_mode != "random":
            if init_dataset_path is None:
                raise ValueError("--quads_init_dataset_path is required when --quads_init_mode is not random")
            init_dataset_path = os.path.abspath(os.path.expanduser(init_dataset_path))
            if not os.path.exists(init_dataset_path):
                raise FileNotFoundError(f"Initial-condition dataset not found: {init_dataset_path}")
            self.init_pos_dataset = list(np.load(init_dataset_path, allow_pickle=True))
            self.max_init_pos = len(self.init_pos_dataset)
            if self.max_init_pos == 0:
                raise ValueError(f"Initial-condition dataset is empty: {init_dataset_path}")

        self.custom_init_indices = list(custom_init_indices or [0])
        if self.init_mode == "dataset_custom":
            bad_indices = [idx for idx in self.custom_init_indices if idx < 0 or idx >= self.max_init_pos]
            if bad_indices:
                raise ValueError(f"Custom init index out of range for dataset: {bad_indices}")

        try:
            self.logger = get_logger()
        except RuntimeError:
            self.logger = None
        self._custom_ptr = 0  # internal cursor for dataset_custom
    def all_dynamics(self):
        return tuple(e.dynamics for e in self.envs)

    def get_rel_pos_vel_item(self, env_id, indices=None):
        i = env_id

        if indices is None:
            # if not specified explicitly, consider all neighbors
            indices = [j for j in range(self.num_agents) if j != i]

        cur_pos = self.pos[i]
        cur_vel = self.vel[i]

        pos_neighbor = np.stack([self.pos[j] for j in indices])
        vel_neighbor = np.stack([self.vel[j] for j in indices])
        pos_rel = pos_neighbor - cur_pos
        vel_rel = vel_neighbor - cur_vel
        # if self.vis_vis == 0:
        #     print("cur_vel",cur_vel)
        #     print("rel_vel", vel_rel)
        return pos_rel, vel_rel

    def get_obs_neighbor_rel(self, env_id, closest_drones):
        i = env_id
        pos_neighbors_rel, vel_neighbors_rel = self.get_rel_pos_vel_item(env_id=i, indices=closest_drones[i])
        obs_neighbor_rel = np.concatenate((pos_neighbors_rel, vel_neighbors_rel), axis=1)
        return obs_neighbor_rel

    def extend_obs_space(self, obs, closest_drones):
        obs_neighbors = []
        for i in range(len(self.envs)):
            obs_neighbor_rel = self.get_obs_neighbor_rel(env_id=i, closest_drones=closest_drones)
            obs_neighbors.append(obs_neighbor_rel.reshape(-1))
        obs_neighbors = np.stack(obs_neighbors)

        # clip observation space of neighborhoods
        obs_neighbors = np.clip(
            obs_neighbors, a_min=self.clip_neighbor_space_min_box, a_max=self.clip_neighbor_space_max_box,
        )
        obs_ext = np.concatenate((obs, obs_neighbors), axis=1)
        return obs_ext

    def neighborhood_indices(self):
        """Return a list of closest drones for each drone in the swarm."""
        # indices of all the other drones except us
        indices = [[j for j in range(self.num_agents) if i != j] for i in range(self.num_agents)]
        indices = np.array(indices)


        if self.num_use_neighbor_obs == self.num_agents - 1:
            return indices
        elif 1 <= self.num_use_neighbor_obs < self.num_agents - 1:
            close_neighbor_indices = []

            for i in range(self.num_agents):
                rel_pos, rel_vel = self.get_rel_pos_vel_item(env_id=i, indices=indices[i])

                rel_dist = np.linalg.norm(rel_pos, axis=1)
                rel_dist = np.maximum(rel_dist, 0.01)
                rel_pos_unit = rel_pos / rel_dist[:, None]

                # new relative distance is a new metric that combines relative position and relative velocity
                # the smaller the new_rel_dist, the closer the drones
                new_rel_dist = rel_dist + np.sum(rel_pos_unit * rel_vel, axis=1)

                rel_pos_index = new_rel_dist.argsort()
                rel_pos_index = rel_pos_index[:self.num_use_neighbor_obs]
                close_neighbor_indices.append(indices[i][rel_pos_index])

            return close_neighbor_indices
        else:
            raise RuntimeError("Incorrect number of neigbors")

    def add_neighborhood_obs(self, obs):
        indices = self.neighborhood_indices()
        obs_ext = self.extend_obs_space(obs, closest_drones=indices)
        return obs_ext

    def can_drones_fly(self):
        """
        Here we count the average number of collisions with the walls and ground in the last N episodes
        Returns: True if drones are considered proficient at flying
        """
        res = abs(np.mean(self.crashes_in_recent_episodes)) < 1 and len(self.crashes_in_recent_episodes) >= 10
        return res

    def calculate_room_collision(self):
        floor_collisions = np.array([env.dynamics.crashed_floor for env in self.envs])
        wall_collisions = np.array([env.dynamics.crashed_wall for env in self.envs])
        ceiling_collisions = np.array([env.dynamics.crashed_ceiling for env in self.envs])

        floor_crash_list = np.where(floor_collisions >= 1)[0]

        cur_wall_crash_list = np.where(wall_collisions >= 1)[0]
        wall_crash_list = np.setdiff1d(cur_wall_crash_list, self.prev_crashed_walls)

        cur_ceiling_crash_list = np.where(ceiling_collisions >= 1)[0]
        ceiling_crash_list = np.setdiff1d(cur_ceiling_crash_list, self.prev_crashed_ceiling)

        return floor_crash_list, wall_crash_list, ceiling_crash_list

    def obst_generation_given_density(self, grid_size=1.0, obst_index=None):
        obst_area_length, obst_area_width = int(self.obst_spawn_area[0]), int(self.obst_spawn_area[1])
        num_room_grids = obst_area_length * obst_area_width

        cell_centers = get_cell_centers(obst_area_length=obst_area_length, obst_area_width=obst_area_width,
                                        grid_size=grid_size)
        room_map = [i for i in range(0, num_room_grids)]

        if obst_index is None:
            # Mode-1 ("random") → fresh entropy every reset
            if getattr(self, "init_mode", "random") == "random":
                obst_index = np.random.default_rng().choice(
                    room_map, size=int(num_room_grids * self.obst_density), replace=False)
            # All other modes → reproducible with global RNG
            else:
                obst_index = np.random.choice(
                    room_map, size=int(num_room_grids * self.obst_density), replace=False)

        obst_pos_arr = []
        pos_arr =[]
        # 0: No Obst, 1: Obst

        obst_map = np.zeros([obst_area_length, obst_area_width])
        for obst_id in obst_index:
            rid, cid = obst_id // obst_area_width, obst_id - (obst_id // obst_area_width) * obst_area_width
            obst_map[rid, cid] = 1
            obst_item = list(cell_centers[rid + int(obst_area_length / grid_size) * cid])
            obst_item.append(self.room_dims[2] / 2.)
            obst_pos_arr.append(obst_item)
        pos_arr = obst_pos_arr
        return obst_map, obst_pos_arr, cell_centers, obst_index

    def init_scene_multi(self):
        models = tuple(e.dynamics.model for e in self.envs)
        for i in range(len(self.quads_view_mode)):
            # self.scenes.append(Quadrotor3DSceneMulti(
            #     models=models,
            #     w=600, h=480, resizable=True, viewpoint=self.quads_view_mode[i],
            #     room_dims=self.room_dims, num_agents=self.num_agents,
            #     render_speed=self.render_speed, formation_size=self.quads_formation_size, obstacles=self.obstacles,
            #     vis_vel_arrows=False, vis_acc_arrows=True, viz_traces=25, viz_trace_nth_step=1,
            #     num_obstacles=self.num_obstacles, scene_index=i
            # ))
            sc = Quadrotor3DSceneMulti(
                models=models,
                w=1280*2, h=1280*2, resizable=True, viewpoint=self.quads_view_mode[i],
                room_dims=self.room_dims, num_agents=self.num_agents,
                render_speed=self.render_speed, formation_size=self.quads_formation_size, obstacles=self.obstacles,
                vis_vel_arrows=False, vis_acc_arrows=True, viz_traces=25, viz_trace_nth_step=1,
                num_obstacles=self.num_obstacles, scene_index=i
            )
            sc.camera_drone_index =5 # <<< set your desired agent for visualization (0-based)
            self.scenes.append(sc)

    # ──────────────────────────────────────────────────────────────
    #  choose (obst_index, goals, spawns) for each mode
    # ──────────────────────────────────────────────────────────────
    def _spec_random(self):
        """
        We don't need to sample goals/spawns here—`self.scenario.reset()`
        will create them.  So just return a placeholder.
        """
        return None, None, None  # obst_index, goals, spawns

    def _spec_dataset_seq(self):
        entry = self.init_pos_dataset[self.init_pos_index]
        self.current_init_episode = entry['episode']
        self.current_init_collisions = entry['total_collisions']
        self.perdrone_exp = np.asarray(entry['per_drone_collisions'])
        if self.print_dataset_id:
            print(f"\n[Reset] Using dataset-seq entry idx={self.init_pos_index}")

        self.init_pos_index = (self.init_pos_index + 1) % self.max_init_pos
        return (np.asarray(entry['obst_index']),
                np.asarray(entry['goal']),
                np.asarray(entry['init_pos']),
                # np.asarray(entry['Initial_pos_drones'])
                )



    def _spec_dataset_custom(self):
        idx = self.custom_init_indices[self._custom_ptr]
        self._custom_ptr = (self._custom_ptr + 1) % len(self.custom_init_indices)
        entry = self.init_pos_dataset[idx]
        self.current_init_episode = entry['episode']
        self.current_init_collisions = entry['total_collisions']
        self.perdrone_exp = np.asarray(entry['per_drone_collisions'])
        if self.print_dataset_id:
            print(f"\n[Reset] Using dataset-custom entry idx={idx}")

        return (np.asarray(entry['obst_index']),
                np.asarray(entry['goal']),
                np.asarray(entry['init_pos']),
                # np.asarray(entry['Initial_pos_drones'])
                           )

    # ──────────────────────────────────────────────────────────────
    # zero all per-episode counters / flags in one place
    # ──────────────────────────────────────────────────────────────
    def _episode_counters_zero(self):
        # ── collision counters ─────────────────────────────────────
        self.collisions_per_episode = self.collisions_after_settle = self.collisions_final_5s = 0
        self.prev_drone_collisions = []
        self.prev_obst_quad_collisions = []
        self.obst_quad_collisions_per_episode = self.obst_quad_collisions_after_settle = 0

        # ── room-collision “previous” lists (needed in step) ───────
        self.prev_crashed_walls = []  # ← already in original class
        self.prev_crashed_ceiling = []
        self.prev_crashed_room = []
        self.prev_floor_crash = []

        # ── distance / success bookkeeping ────────────────────────
        self.distance_to_goal = [[] for _ in range(self.num_agents)]
        self.agent_col_agent = np.ones(self.num_agents)
        self.agent_col_obst = np.ones(self.num_agents)
        self.reached_goal = [False] * self.num_agents

        # ── misc per-episode counters  ─────────────────────────────
        self.obstacle_coll_without_grace = self.obstacle_coll_ct = \
            self.drone_coll_ct = self.floor_coll_ct = 0
        self.episode_collisions_drone_drone = np.zeros(self.num_agents, dtype=int)
        self.episode_collisions_drone_obst = np.zeros(self.num_agents, dtype=int)
        self.episode_collisions_floor = np.zeros(self.num_agents, dtype=int)
        self.episode_collisions_drone_obst_without_grace = np.zeros(self.num_agents, dtype=int)

    # ---------------------------------------------------------------------------
    #  Compact reset for all three modes  ("random", "dataset_seq", "dataset_custom")
    # ---------------------------------------------------------------------------
    def reset(self, obst_density=None, obst_size=None) -> np.ndarray:
        # — close any stray pyglet windows —
        if len(self.scenes) > 1:
            import pyglet
            for w in list(pyglet.app.windows):
                w.close()
        self.scenes.clear()
        if not hasattr(self, "current_init_episode"):
            self.current_init_episode = -1
            self.current_init_collisions = 0
            self.perdrone_exp = np.zeros(self.num_agents, dtype=int)
        obs = []

        # — 1. choose episode spec —
        if self.init_mode == "random":
            obst_index, goals, spawns = self._spec_random()
        elif self.init_mode == "dataset_seq":
            obst_index, goals, spawns = self._spec_dataset_seq()
        elif self.init_mode == "dataset_custom":
            obst_index, goals, spawns = self._spec_dataset_custom()
        else:
            raise ValueError(f"Unknown init_mode '{self.init_mode}'")
        self.det_init_pos = (self.init_mode != "random")  # ← ADD THIS LINE

        # — 2. build obstacles & reset scenario —
        if self.use_obstacles:
            if obst_density is not None:
                self.obst_density = obst_density
                self.num_obstacles = int(self.obst_density * self.obst_spawn_area[0] * self.obst_spawn_area[1])
            if obst_size is not None:
                self.obst_size = obst_size
        if self.use_obstacles:
            self.obst_map, self.obst_pos_arr, cell_centers, self.obst_index = \
                self.obst_generation_given_density(obst_index=obst_index)
            self.obstacles = MultiObstacles(obstacle_size=self.obst_size,
                                            quad_radius=self.quad_arm)

            self.scenario.reset(obst_info=dict(
                obst_map=self.obst_map,
                cell_centers=cell_centers,
                obst_pos_arr=self.obst_pos_arr,
                obst_size=self.obst_size,
                obst_density=self.obst_density,
                spawns=spawns,
                goals=goals
            ))
        else:
            self.obst_map = None
            self.obst_pos_arr = []
            self.obst_index = None
            self.scenario.reset()

        # —— if we came from _spec_random(), fill in goals/spawns now ——
        if goals is None:
            goals = self.scenario.goals
            spawns = getattr(self.scenario, "spawn_points", None)
        if spawns is None:
            spawns = [None] * self.num_agents

        # — 3. spawn quadrotors & collect base observations —
        for i, env in enumerate(self.envs):
            env.goal, env.spawn_point, env.rew_coeff = goals[i], spawns[i], self.rew_coeff
            obs_i = env.reset()
            obs.append(obs_i)
            self.pos[i] = env.dynamics.pos
            self.vel[i] = env.dynamics.vel
            self.acc[i] = env.dynamics.acc

        # — 4. neighbour & obstacle observations —
        if self.num_use_neighbor_obs > 0:
            obs = self.add_neighborhood_obs(obs)
        if self.use_obstacles:
            obs = self.obstacles.reset(obs=obs, quads_pos=self.pos,
                                       pos_arr=self.obst_pos_arr)

        # — 5. zero per-episode counters —
        self._episode_counters_zero()
        self.total_col= 0

        # print("pos", self.pos)
        # print("obstacle index", self.obst_index)
        # — 6. render / logging bookkeeping —
        self.obs_save = obs
        self.save_goal = goals
        self.save_obst_index = self.obst_index
        self.init_pos_save = spawns
        if self.quads_render:
            self.reset_scene = True
            self.quads_formation_size = self.scenario.formation_size
            self.all_collisions = {k: [0.0] * self.num_agents
                                   for k in ('drone', 'ground', 'obstacle')}
            ## aadded for recording
            self._record_refresh_needed = True

        return np.asarray(obs)

    def step(self, actions):
        if not hasattr(self, "episode_collisions_total"):
            self.episode_collisions_total = 0
            self.episode_collisions_per_drone = np.zeros(self.num_agents, dtype=int)
            # ensure it exists on first ever step
            self.episode_collisions_drone_obst_without_grace = np.zeros(self.num_agents, dtype=int)

        obs, rewards, dones, infos = [], [], [], []
        # self.vis_vis =1

        # print("action", actions)
        if not hasattr(self, "episode_collisions_total"):
            self.episode_collisions_total = 0
            self.episode_collisions_per_drone = np.zeros(self.num_agents, dtype=int)

        for i, a in enumerate(actions):
            self.envs[i].rew_coeff = self.rew_coeff
            observation, reward, done, info = self.envs[i].step(a)
            obs.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

            self.pos[i, :] = self.envs[i].dynamics.pos
            # added
            self.vel[i, :] = self.envs[i].dynamics.vel
            self.acc[i,:] = self.envs[i].dynamics.accelerometer
        # print("pos",self.pos)
        # print("obs", obs)
        # 1. Calculate collisions: 1) between drones 2) with obstacles 3) with room
        # 1) Collisions between drones
        # print("obs", len(obs), obs[0].shape)
        drone_col_matrix, curr_drone_collisions, distance_matrix = \
            calculate_collision_matrix(positions=self.pos, collision_threshold=self.collision_threshold)
        # print("goals", self.scenario.goals)
        # if self.vis_vis < 10:


        #     self.vis_vis += 1
        #     print("self.neighborhood_indices()",self.neighborhood_indices())

        # # Filter curr_drone_collisions
        curr_drone_collisions = curr_drone_collisions.astype(int)
        curr_drone_collisions = np.delete(curr_drone_collisions, np.unique(
            np.where(curr_drone_collisions == [-1000, -1000])[0]), axis=0)

        old_quad_collision = set(map(tuple, self.prev_drone_collisions))
        new_quad_collision = np.array([x for x in curr_drone_collisions if tuple(x) not in old_quad_collision])
        self.last_step_unique_collisions = np.setdiff1d(curr_drone_collisions, self.prev_drone_collisions)

        # # Filter distance_matrix; Only contains quadrotor pairs with distance <= self.collision_threshold
        near_quad_ids = np.where(distance_matrix[:, 2] <= self.collision_falloff_threshold)
        distance_matrix = distance_matrix[near_quad_ids]

        # Collision between 2 drones counts as a single collision
        # # Calculate collisions (i) All collisions (ii) collisions after grace period
        collisions_curr_tick = len(self.last_step_unique_collisions) // 2
        self.collisions_per_episode += collisions_curr_tick
        self.collisions_curr_tick = collisions_curr_tick
        if collisions_curr_tick > 0 and self.envs[0].tick >= self.collisions_grace_period_steps:
            self.collisions_after_settle += collisions_curr_tick
            for agent_id in self.last_step_unique_collisions:
                self.agent_col_agent[agent_id] = 0
        if collisions_curr_tick > 0 and self.envs[0].time_remain <= self.collisions_final_grace_period_steps:
            self.collisions_final_5s += collisions_curr_tick


        # # Aux: Neighbor Collisions
        self.prev_drone_collisions = curr_drone_collisions

        # 2) Collisions with obstacles
        obst_col_data_fin = np.zeros(self.num_agents)
        obst_quad_col_matrix = np.array([], dtype=int)
        quad_obst_pair = {}
        self.collisions_obst_curr_tick = 0
        if self.use_obstacles:
            rew_obst_quad_collisions_raw = np.zeros(self.num_agents)
            obst_quad_col_matrix, quad_obst_pair = self.obstacles.collision_detection(pos_quads=self.pos)
            # We assume drone can only collide with one obstacle at the same time.
            # Given this setting, in theory, the gap between obstacles should >= 0.1 (drone diameter: 0.46*2 = 0.92)
            self.curr_quad_col = np.setdiff1d(obst_quad_col_matrix, self.prev_obst_quad_collisions)
            collisions_obst_curr_tick = len(self.curr_quad_col)
            self.collisions_obst_curr_tick = collisions_obst_curr_tick
            # if collisions_obst_curr_tick> 0:
            #     print("collisions_obst_curr_tick------------",collisions_obst_curr_tick)
            #     print(" self.curr_quad_col", self.curr_quad_col)
            self.obst_quad_collisions_per_episode += collisions_obst_curr_tick

            if collisions_obst_curr_tick > 0 and self.envs[0].tick >= self.collisions_grace_period_steps:
                self.obst_quad_collisions_after_settle += collisions_obst_curr_tick
                for qid in self.curr_quad_col:
                    q_rel_dist = np.linalg.norm(obs[qid][0:3])
                    if q_rel_dist > 3.5:
                        self.distance_to_goal_3_5 += 1
                    if q_rel_dist > 5.0:
                        self.distance_to_goal_5 += 1
                    # Used for log agent_success
                    self.agent_col_obst[qid] = 0

            # # Aux: Obstacle Collisions
            self.prev_obst_quad_collisions = obst_quad_col_matrix
            # Added
            obst_col_data_fin = rew_obst_quad_collisions_raw

            if len(obst_quad_col_matrix) > 0:
                # We assign penalties to the drones which collide with the obstacles
                # And obst_quad_last_step_unique_collisions only include drones' id
                rew_obst_quad_collisions_raw[self.curr_quad_col] = -1.0
                #aDDED
                obst_col_data_fin[rew_obst_quad_collisions_raw == -1.0] = 1.0

        # 3) Collisions with room
        floor_crash_list, wall_crash_list, ceiling_crash_list = self.calculate_room_collision()
        room_crash_list = np.unique(np.concatenate([floor_crash_list, wall_crash_list, ceiling_crash_list]))
        # Added
        room_crash_list1 = room_crash_list

        room_crash_list = np.setdiff1d(room_crash_list, self.prev_crashed_room)

        floor_crash_list = np.setdiff1d(floor_crash_list, self.prev_floor_crash)

        # print("room_crash_list", room_crash_list)
        # # Aux: Room Collisions
        self.prev_crashed_walls = wall_crash_list
        self.prev_crashed_ceiling = ceiling_crash_list
        self.prev_crashed_room = room_crash_list
        self.prev_floor_crash =  floor_crash_list

        # Added
        room_crash_fin = np.zeros(self.num_agents)
        room_crash_fin[room_crash_list1] = 1.0
        floor_crash_list_fin = np.zeros(self.num_agents)
        wall_crash_list_fin= np.zeros(self.num_agents)
        ceiling_crash_list_fin= np.zeros(self.num_agents)
        floor_crash_list_fin[floor_crash_list] = 1.0
        wall_crash_list_fin[wall_crash_list] = 1.0
        ceiling_crash_list_fin[ceiling_crash_list] = 1.0
        # print("floor_crash_list_fin",floor_crash_list_fin)
        drone_safety = np.where((drone_col_matrix == 1.0) | (obst_col_data_fin == 1.0) | (room_crash_fin == 1.0), 1.0,
                                0)
        drone_col_matrix_final = np.where(drone_col_matrix == 1.0, 1.0, 0.0)

        drone_safety_final = np.where((drone_col_matrix == 1.0) | (obst_col_data_fin == 1.0), 1.0, 0)
        if any(drone_safety):
            if self.envs[0].tick < self.collisions_grace_period_steps:
                if self.collisions_obst_curr_tick == 1:
                    self.obstacle_coll_without_grace += 1
                    self.episode_collisions_drone_obst_without_grace += obst_col_data_fin.astype(int)

                    # print("Obstacle Collision before settling", obst_col_data_fin)
            if self.envs[0].tick >= self.collisions_grace_period_steps:
                if self.collisions_curr_tick == 1:

                    self.drone_coll_ct += 1

                    self.episode_collisions_drone_drone += drone_col_matrix_final.astype(int)


                    # print("Collision - drone to drone", drone_col_matrix_final)
                if self.collisions_obst_curr_tick == 1:
                    self.obstacle_coll_ct +=1
                    self.episode_collisions_drone_obst += obst_col_data_fin.astype(int)

                self.episode_collisions_floor += floor_crash_list_fin.astype(int)
                self.floor_coll_ct = sum(self.episode_collisions_floor)
                # print("self.floor_coll_ct ", self.floor_coll_ct , self.episode_collisions_floor)


            # if self.envs[0].tick > self.collisions_grace_period_steps:
            #     print("Collision - drone to drone", drone_col_matrix)
            #
            # if collisions_curr_tick> 0 and self.envs[0].tick < self.collisions_grace_period_steps:
            #     # print("collisions_curr_tick", collisions_curr_tick)
            #
            #     print("drone_safety_final before settling", drone_col_matrix )
            #
            #
            # if self.collisions_after_settle > 0 and self.envs[0].tick >= self.collisions_grace_period_steps:
            #     print("collisions_curr_tick", collisions_curr_tick)
            #     # print("collisions_after_settle",  self.collisions_after_settle)
            #
            #
            #     print("drone_safety_final after settiling - drone to drone", drone_col_matrix )
            #
            # if collisions_obst_curr_tick > 0 and self.envs[0].tick < self.collisions_grace_period_steps:
            #     print("drone_safety_final - obstacle before settling", obst_col_data_fin)
            #
            # if collisions_obst_curr_tick > 0  and self.envs[0].tick >= self.collisions_grace_period_steps:
            #     print("self.obst_quad_collisions_after_settle ",self.obst_quad_collisions_after_settle )
            #
            #     print("drone_safety_final - obstacle", obst_col_data_fin )


            # drone_safety_final_np = np.array(drone_safety_final, dtype=int)
            # self.episode_collisions_total += int(drone_safety_final_np.sum())
            # self.episode_collisions_per_drone += drone_safety_final_np

        # assert hasattr(self, 'iter_step')
        # print('iter_step', self.iter_step)

        drone_col_met_dict = {
            'drone_col_matrix': drone_col_matrix,
            'curr_drone_collisions': curr_drone_collisions,
            'distance_matrix': distance_matrix,
            'obst_quad_col_matrix': obst_quad_col_matrix,
            'quad_obst_pair': quad_obst_pair,  # quad_obst_pair is a dictionary itself
            'floor_crash_list': floor_crash_list,
            'wall_crash_list': wall_crash_list,
            'ceiling_crash_list': ceiling_crash_list,
            'obst_col_data_fin': obst_col_data_fin,
            'room_crash_fin': room_crash_fin,
            'drone_safety': drone_safety
        }

        infos[0]["drone_safety"] = drone_safety
        infos[0]["drone_safety_final"] = drone_safety_final
        infos[0]["drone_col"] = drone_col_matrix_final
        infos[0]["obst_col"] = obst_col_data_fin
        infos[0]["drone_col_met_dict"] = drone_col_met_dict
        infos[0]['floor_crash_list'] = floor_crash_list_fin
        infos[0]['wall_crash_list'] = wall_crash_list_fin
        infos[0]['ceiling_crash_list'] = ceiling_crash_list_fin
        infos[0]["Initial_pos_drones"] = self.init_pos_save
        infos[0]["Initial_obs_drones"] = self.obs_save
        infos[0]["Obstacle_Index"] = self.save_obst_index
        infos[0]["goal"] = self.save_goal
        infos[0]["collisions_curr_tick"] = self.collisions_curr_tick
        infos[0]["collisions_obst_curr_tick"] = self.collisions_obst_curr_tick
        infos[0]["tick"] = self.envs[0].tick
        if self.init_mode == "random":
            infos[0]["org_coll"] =self.drone_coll_ct + self.obstacle_coll_ct + self.floor_coll_ct
            infos[0]["org_coll_det"] =  self.episode_collisions_drone_drone + self.episode_collisions_drone_obst + self.episode_collisions_floor
        if self.det_init_pos:
            infos[0]["org_coll"] = self.current_init_collisions
            infos[0]["org_coll_det"] = self.perdrone_exp
        infos[0]["New_coll"] = self.drone_coll_ct + self.obstacle_coll_ct + self.floor_coll_ct
        # print("coll----------", self.episode_collisions_drone_drone + self.episode_collisions_drone_obst + self.episode_collisions_floor)
        infos[0]["New_coll_det"] = self.episode_collisions_drone_drone + self.episode_collisions_drone_obst + self.episode_collisions_floor
        infos[0]["floor_coll"] = self.floor_coll_ct
        infos[0]["drone-drone_coll"] = self.episode_collisions_drone_drone
        infos[0]["drone-obst_coll"] = self.episode_collisions_drone_obst
        infos[0]["drone-floor_coll"] = self.episode_collisions_floor
        infos[0]["drone-drone_coll_count"] = self.drone_coll_ct
        infos[0]["drone-obst_coll_count"] = self.obstacle_coll_ct
        infos[0]["drone-floor_coll_count"] = self.floor_coll_ct
        infos[0]["pos"]= self.pos
        infos[0]["vel"] = self.vel
        infos[0]["acc"] = self.acc
        infos[0]["obst_map"] = self.obst_map
        infos[0]["goal"] =self.scenario.goals

        if (self.drone_coll_ct + self.obstacle_coll_ct + self.floor_coll_ct) > self.total_col:
            self.total_col = self.drone_coll_ct + self.obstacle_coll_ct + self.floor_coll_ct

            # print("Total_Coll", self.total_col, "details", (self.episode_collisions_drone_drone + self.episode_collisions_drone_obst + self.episode_collisions_floor))

        # 2. Calculate rewards and infos for collision
        # 1) Between drones
        rew_collisions_raw = np.zeros(self.num_agents)
        if self.last_step_unique_collisions.any():
            rew_collisions_raw[self.last_step_unique_collisions] = -1.0
        rew_collisions = self.rew_coeff["quadcol_bin"] * rew_collisions_raw

        # penalties for being too close to other drones
        if len(distance_matrix) > 0:
            rew_proximity = -1.0 * calculate_drone_proximity_penalties(
                distance_matrix=distance_matrix, collision_falloff_threshold=self.collision_falloff_threshold,
                dt=self.control_dt, max_penalty=self.rew_coeff["quadcol_bin_smooth_max"], num_agents=self.num_agents,
            )
        else:
            rew_proximity = np.zeros(self.num_agents)

        # 2) With obstacles
        rew_collisions_obst_quad = np.zeros(self.num_agents)
        if self.use_obstacles:
            rew_collisions_obst_quad = self.rew_coeff["quadcol_bin_obst"] * rew_obst_quad_collisions_raw

        # 3) With room
        # # TODO: reward penalty
        if self.envs[0].tick >= self.collisions_grace_period_steps:
            self.collisions_room_per_episode += len(room_crash_list)
            self.collisions_floor_per_episode += len(floor_crash_list)
            self.collisions_wall_per_episode += len(wall_crash_list)
            self.collisions_ceiling_per_episode += len(ceiling_crash_list)
        # Reward & Info
        for i in range(self.num_agents):
            rewards[i] += rew_collisions[i]
            rewards[i] += rew_proximity[i]

            infos[i]["rewards"]["rew_quadcol"] = rew_collisions[i]
            infos[i]["rewards"]["rew_proximity"] = rew_proximity[i]
            infos[i]["rewards"]["rewraw_quadcol"] = rew_collisions_raw[i]

            if self.use_obstacles:
                rewards[i] += rew_collisions_obst_quad[i]
                infos[i]["rewards"]["rew_quadcol_obstacle"] = rew_collisions_obst_quad[i]
                infos[i]["rewards"]["rewraw_quadcol_obstacle"] = rew_obst_quad_collisions_raw[i]

            self.distance_to_goal[i].append(-infos[i]["rewards"]["rewraw_pos"])
            if len(self.distance_to_goal[i]) >= 5 and \
                    np.mean(self.distance_to_goal[i][-5:]) / self.envs[0].dt < self.scenario.approch_goal_metric \
                    and not self.reached_goal[i]:
                self.reached_goal[i] = True

        # 3. Applying random forces: 1) aerodynamics 2) between drones 3) obstacles 4) room
        self_state_update_flag = False

        # # 1) aerodynamics
        if self.use_downwash:
            envs_dynamics = [env.dynamics for env in self.envs]
            applied_downwash_list = perform_downwash(drones_dyn=envs_dynamics, dt=self.control_dt)
            downwash_agents_list = np.where(applied_downwash_list == 1)[0]
            if len(downwash_agents_list) > 0:
                self_state_update_flag = True

        # # 2) Drones
        if self.apply_collision_force:
            if len(new_quad_collision) > 0:
                self_state_update_flag = True
                for val in new_quad_collision:
                    dyn1, dyn2 = self.envs[val[0]].dynamics, self.envs[val[1]].dynamics
                    dyn1.vel, dyn1.omega, dyn2.vel, dyn2.omega = perform_collision_between_drones(
                        pos1=dyn1.pos, vel1=dyn1.vel, omega1=dyn1.omega, pos2=dyn2.pos, vel2=dyn2.vel, omega2=dyn2.omega)
            # # 3) Obstacles
            if self.use_obstacles:
                if len(self.curr_quad_col) > 0:
                    self_state_update_flag = True
                    for val in self.curr_quad_col:
                        obstacle_id = quad_obst_pair[int(val)]
                        obstacle_pos = self.obstacles.pos_arr[int(obstacle_id)]
                        perform_collision_with_obstacle(drone_dyn=self.envs[int(val)].dynamics,
                                                        obstacle_pos=obstacle_pos,
                                                        obstacle_size=self.obst_size)

            # # 4) Room
            if len(wall_crash_list) > 0 or len(ceiling_crash_list) > 0:
                self_state_update_flag = True

                for val in wall_crash_list:
                    perform_collision_with_wall(drone_dyn=self.envs[val].dynamics, room_box=self.envs[0].room_box)

                for val in ceiling_crash_list:
                    perform_collision_with_ceiling(drone_dyn=self.envs[val].dynamics)

        # 4. Run the scenario passed to self.quads_mode
        self.scenario.step()
        # 5. Collect final observations
        # Collect positions after physical interaction

        for i in range(self.num_agents):
            self.pos[i, :] = self.envs[i].dynamics.pos
            self.vel[i, :] = self.envs[i].dynamics.vel
        # print("self.pos",self.pos)
        # print("self.vel",self.vel )
        if self_state_update_flag:
            obs = [e.state_vector(e) for e in self.envs]

        # Concatenate observations of neighbor drones
        if self.num_use_neighbor_obs > 0:
            obs = self.add_neighborhood_obs(obs)
        # Concatenate obstacle observations
        if self.use_obstacles:
            obs = self.obstacles.step(obs=obs, quads_pos=self.pos)

        # print("obs",obs)

        # 6. Update info for replay buffer
        # Once agent learns how to take off, activate the replay buffer
        if self.use_replay_buffer and not self.activate_replay_buffer:
            self.crashes_last_episode += infos[0]["rewards"]["rew_crash"]

        # Rendering
        if self.quads_render:
            # Collisions with room
            ground_collisions = [1.0 if env.dynamics.on_floor else 0.0 for env in self.envs]
            if self.use_obstacles:
                obst_coll = [1.0 if i < 0 else 0.0 for i in rew_obst_quad_collisions_raw]
            else:
                obst_coll = [0.0 for _ in range(self.num_agents)]
            self.all_collisions = {'drone': drone_col_matrix, 'ground': ground_collisions,

                                   'obstacle': obst_coll}

        #for saving
        self.reached_goal = np.array(self.reached_goal)
        scenario_name = self.scenario.name()[9:]
        agent_col_flag_list = np.logical_and(self.agent_col_agent, self.agent_col_obst)
        agent_success_flag_list = np.logical_and(agent_col_flag_list, self.reached_goal)
        agent_success_ratio = 1.0 * np.sum(agent_success_flag_list) / self.num_agents

        # agent_deadlock_rate
        # Doesn't approach to the goal while no collisions with other objects
        agent_deadlock_list = np.logical_and(agent_col_flag_list, 1 - self.reached_goal)
        agent_deadlock_ratio = 1.0 * np.sum(agent_deadlock_list) / self.num_agents

        # agent_col_rate
        # Collide with other drones and obstacles
        agent_col_ratio = 1.0 - np.sum(agent_col_flag_list) / self.num_agents

        # agent_neighbor_col_rate
        agent_neighbor_col_ratio = 1.0 - np.sum(self.agent_col_agent) / self.num_agents
        # agent_obst_col_rate
        agent_obst_col_ratio = 1.0 - np.sum(self.agent_col_obst) / self.num_agents

        infos[0]['metric/agent_success_rate'] = agent_success_ratio
        infos[0]['scenario_name/agent_success_rate'] = agent_success_ratio
        # agent_deadlock_rate
        infos[0]['metric/agent_deadlock_rate'] = agent_deadlock_ratio
        infos[0]['scenario_name/agent_deadlock_rate'] = agent_deadlock_ratio
        # agent_col_rate
        infos[0]['metric/agent_col_rate'] = agent_col_ratio
        infos[0]['scenario_name/agent_col_rate'] = agent_col_ratio
        # agent_neighbor_col_rate
        infos[0]['metric/agent_neighbor_col_rate'] = agent_neighbor_col_ratio
        infos[0]['scenario_name/agent_neighbor_col_rate'] = agent_neighbor_col_ratio
        # agent_obst_col_rate
        infos[0]['metric/agent_obst_col_rate'] = agent_obst_col_ratio
        infos[0]['scenario_name/agent_obst_col_rate'] = agent_obst_col_ratio
        self.distance_to_goal1 = np.array(self.distance_to_goal)
        infos[0]['distance_to_goal_1s'] = (1.0 / self.envs[0].dt) * np.mean(self.distance_to_goal1[i, int(-1 * self.control_freq):])
        infos[0]['distance_to_goal_3s'] = (1.0 / self.envs[0].dt) * np.mean(self.distance_to_goal1[i, int(-3 * self.control_freq):])
        infos[0]['distance_to_goal_5s'] = (1.0 / self.envs[0].dt) * np.mean(self.distance_to_goal1[i, int(-5 * self.control_freq):])

        infos[0]['scenario_name'] = scenario_name

        collision_information = {
            "drone_safety": drone_safety,
            "drone_safety_final": drone_safety_final,
            "drone_col_matrix_final": drone_col_matrix_final,
            "obst_col_data_final": obst_col_data_fin,
            # "drone_col_metrics": drone_col_met_dict,
            "floor_crash_list_final": floor_crash_list_fin,
            "wall_crash_list_final": wall_crash_list_fin,
            "ceiling_crash_list_final": ceiling_crash_list_fin,
            "initial_positions": self.init_pos_save,
            "initial_observations": self.obs_save,
            "obstacle_indices": self.save_obst_index,
            "goal_position": self.save_goal,
            "collisions_this_tick": self.collisions_curr_tick,
            "obstacle_collisions_this_tick": self.collisions_obst_curr_tick,
            "tick": self.envs[0].tick,
            "original_collisions": (
                self.drone_coll_ct + self.obstacle_coll_ct + self.floor_coll_ct
                if self.init_mode == "random"
                else self.current_init_collisions
            ),
            "original_collision_details": (
                self.episode_collisions_drone_drone + self.episode_collisions_drone_obst + self.episode_collisions_floor
                if self.init_mode == "random"
                else self.perdrone_exp
            ),
            "new_collisions": self.drone_coll_ct + self.obstacle_coll_ct + self.floor_coll_ct,
            "new_collision_details": self.episode_collisions_drone_drone + self.episode_collisions_drone_obst + self.episode_collisions_floor,
            "drone_drone_collisions": self.episode_collisions_drone_drone,
            "drone_obstacle_collisions": self.episode_collisions_drone_obst,
            "drone_floor_collisions": self.episode_collisions_floor,
            "drone_drone_collision_count": self.drone_coll_ct,
            "drone_obstacle_collision_count": self.obstacle_coll_ct,
            "drone_floor_collision_count": self.floor_coll_ct,
            # We likely do not need the below values
            # "positions": self.pos,
            # "velocities": self.vel,
            # "accelerations": self.acc,
            # "obstacle_map": self.obst_map,
            # "scenario_goals": self.scenario.goals,

            "agent_success_ratio": agent_success_ratio,
            "agent_deadlock_ratio": agent_deadlock_ratio,
            "agent_col_ratio": agent_col_ratio,
            "agent_neighbor_col_ratio": agent_neighbor_col_ratio,
            "agent_obst_col_ratio": agent_obst_col_ratio,
            "distance_to_goal_1s": (1.0 / self.envs[0].dt) * np.mean(self.distance_to_goal1[i, int(-1 * self.control_freq):]),
            "distance_to_goal_3s": (1.0 / self.envs[0].dt) * np.mean(self.distance_to_goal1[i, int(-3 * self.control_freq):]),
            "distance_to_goal_5s": (1.0 / self.envs[0].dt) * np.mean(self.distance_to_goal1[i, int(-5 * self.control_freq):]),
        }
        if self.logger is not None:
            self.logger.update_collision_information(collision_information)


        # 7. DONES
        if any(dones):
            self._close_writers()

            print('obst_quad_collisions_per_episode: ', self.obst_quad_collisions_per_episode)
            scenario_name = self.scenario.name()[9:]
            for i in range(len(infos)):
                if self.saved_in_replay_buffer:
                    infos[i]['episode_extra_stats'] = {
                        'num_collisions_replay': self.collisions_per_episode,
                        'num_collisions_obst_replay': self.obst_quad_collisions_per_episode,
                    }
                else:
                    self.distance_to_goal = np.array(self.distance_to_goal)
                    self.reached_goal = np.array(self.reached_goal)
                    infos[i]['episode_extra_stats'] = {
                        'num_collisions': self.collisions_per_episode,
                        'num_collisions_with_room': self.collisions_room_per_episode,
                        'num_collisions_with_floor': self.collisions_floor_per_episode,
                        'num_collisions_with_wall': self.collisions_wall_per_episode,
                        'num_collisions_with_ceiling': self.collisions_ceiling_per_episode,
                        'num_collisions_after_settle': self.collisions_after_settle,
                        f'{scenario_name}/num_collisions': self.collisions_after_settle,

                        'num_collisions_final_5_s': self.collisions_final_5s,
                        f'{scenario_name}/num_collisions_final_5_s': self.collisions_final_5s,

                        'distance_to_goal_1s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-1 * self.control_freq):]),
                        'distance_to_goal_3s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-3 * self.control_freq):]),
                        'distance_to_goal_5s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-5 * self.control_freq):]),

                        f'{scenario_name}/distance_to_goal_1s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-1 * self.control_freq):]),
                        f'{scenario_name}/distance_to_goal_3s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-3 * self.control_freq):]),
                        f'{scenario_name}/distance_to_goal_5s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-5 * self.control_freq):]),
                    }

                    if self.use_obstacles:
                        infos[i]['episode_extra_stats']['num_collisions_obst_quad'] = \
                            self.obst_quad_collisions_per_episode
                        infos[i]['episode_extra_stats']['num_collisions_obst_quad_after_settle'] = \
                            self.obst_quad_collisions_after_settle
                        infos[i]['episode_extra_stats'][f'{scenario_name}/num_collisions_obst'] = \
                            self.obst_quad_collisions_per_episode

                        infos[i]['episode_extra_stats']['num_collisions_obst_quad_3_5'] = \
                            self.distance_to_goal_3_5
                        infos[i]['episode_extra_stats'][f'{scenario_name}/num_collisions_obst_quad_3_5'] = \
                            self.distance_to_goal_3_5

                        infos[i]['episode_extra_stats']['num_collisions_obst_quad_5'] = \
                            self.distance_to_goal_5
                        infos[i]['episode_extra_stats'][f'{scenario_name}/num_collisions_obst_quad_5'] = \
                            self.distance_to_goal_5

            if not self.saved_in_replay_buffer:
                # agent_success_rate: base_success_rate, based on per agent
                # 0: collision; 1: no collision
                agent_col_flag_list = np.logical_and(self.agent_col_agent, self.agent_col_obst)
                agent_success_flag_list = np.logical_and(agent_col_flag_list, self.reached_goal)
                agent_success_ratio = 1.0 * np.sum(agent_success_flag_list) / self.num_agents

                # agent_deadlock_rate
                # Doesn't approach to the goal while no collisions with other objects
                agent_deadlock_list = np.logical_and(agent_col_flag_list, 1 - self.reached_goal)
                agent_deadlock_ratio = 1.0 * np.sum(agent_deadlock_list) / self.num_agents

                # agent_col_rate
                # Collide with other drones and obstacles
                agent_col_ratio = 1.0 - np.sum(agent_col_flag_list) / self.num_agents

                # agent_neighbor_col_rate
                agent_neighbor_col_ratio = 1.0 - np.sum(self.agent_col_agent) / self.num_agents
                # agent_obst_col_rate
                agent_obst_col_ratio = 1.0 - np.sum(self.agent_col_obst) / self.num_agents

                for i in range(len(infos)):
                    # agent_success_rate
                    infos[i]['episode_extra_stats']['metric/agent_success_rate'] = agent_success_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_success_rate'] = agent_success_ratio
                    # agent_deadlock_rate
                    infos[i]['episode_extra_stats']['metric/agent_deadlock_rate'] = agent_deadlock_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_deadlock_rate'] = agent_deadlock_ratio
                    # agent_col_rate
                    infos[i]['episode_extra_stats']['metric/agent_col_rate'] = agent_col_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_col_rate'] = agent_col_ratio
                    # agent_neighbor_col_rate
                    infos[i]['episode_extra_stats']['metric/agent_neighbor_col_rate'] = agent_neighbor_col_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_neighbor_col_rate'] = agent_neighbor_col_ratio
                    # agent_obst_col_rate
                    infos[i]['episode_extra_stats']['metric/agent_obst_col_rate'] = agent_obst_col_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_obst_col_rate'] = agent_obst_col_ratio

                    infos[0]['metric/agent_success_rate'] = agent_success_ratio
                    infos[0]['scenario_name/agent_success_rate'] = agent_success_ratio
                    # agent_deadlock_rate
                    infos[0]['metric/agent_deadlock_rate'] = agent_deadlock_ratio
                    infos[0]['scenario_name/agent_deadlock_rate'] = agent_deadlock_ratio
                    # agent_col_rate
                    infos[0]['metric/agent_col_rate'] = agent_col_ratio
                    infos[0]['scenario_name/agent_col_rate'] = agent_col_ratio
                    # agent_neighbor_col_rate
                    infos[0]['metric/agent_neighbor_col_rate'] = agent_neighbor_col_ratio
                    infos[0]['scenario_name/agent_neighbor_col_rate'] = agent_neighbor_col_ratio
                    # agent_obst_col_rate
                    infos[0]['metric/agent_obst_col_rate'] = agent_obst_col_ratio
                    infos[0]['scenario_name/agent_obst_col_rate'] = agent_obst_col_ratio
                    infos[0]['distance_to_goal_1s'] =  (1.0 / self.envs[0].dt) * np.mean(self.distance_to_goal[i, int(-1 * self.control_freq):])
                    infos[0]['distance_to_goal_3s'] =  (1.0 / self.envs[0].dt) * np.mean(self.distance_to_goal[i, int(-3 * self.control_freq):])
                    infos[0]['distance_to_goal_5s'] =  (1.0 / self.envs[0].dt) * np.mean(self.distance_to_goal[i, int(-5 * self.control_freq):])

                    infos[0]['scenario_name']= scenario_name

           #############################################################
            # # ✅ Print collision summary for the episode
            # print(f"\n🧠 Episode Collision Summary")
            # print(f"→ Expected Total Collisions: {self.current_init_collisions}")
            # print(f"→ Actual Collisions This Episode: {self.episode_collisions_total}")
            # print(f"→ Expected Per-Drone Collision Vector: {self.perdrone_exp}")
            # print(f"→ Actual Per-Drone Collision Count: {self.episode_collisions_per_drone.tolist()}")

            # ✅ Print collision summary for the episode
            # infos[0]["New_coll"] = self.drone_coll_ct + self.obstacle_coll_ct + self.floor_coll_ct
            # infos[0]["org_coll_det"] = self.perdrone_exp
            # infos[0]["New_coll_det"] = self.episode_collisions_drone_drone + self.episode_collisions_drone_obst + self.episode_collisions_floor
            # print(f"\n🧠 Episode Collision Summary")
            # print(f"→ Expected Total Collisions: {self.current_init_collisions}")

            print(f"→ Actual Collisions This Episode: {self.drone_coll_ct  + self.obstacle_coll_ct + self.floor_coll_ct}")
            # print(f"→ Actual Collisions Drone-Drone: {self.drone_coll_ct } and Drone-Obstacle :{ self.obstacle_coll_ct} and Drone-Obst Without Grace Period: {self.obstacle_coll_without_grace}")
            print(f"→ Actual Collisions Drone-Drone: {self.drone_coll_ct } and Drone-Obstacle :{ self.obstacle_coll_ct} and Drone-floor: {self.floor_coll_ct}")

            #added

            if self.det_init_pos:
                print(f"→ Expected Per-Drone Collision Vector: {self.perdrone_exp}")
            print(f"→ Actual Per-Drone Collision Count: {self.episode_collisions_drone_drone +  self.episode_collisions_drone_obst + self.episode_collisions_floor}")
            #
            # print(f"→ Actual Collisions This Episode: {self.drone_coll_ct  + self.obstacle_coll_ct }")
            # print(f"→ Actual Per-Drone Collision Count: {self.episode_collisions_drone_drone + self.episode_collisions_drone_obst}")
            # 🔄 Reset counters for next episode

            self.episode_collisions_total = 0
            self.episode_collisions_per_drone[:] = 0
            #################################################################
            obs = self.reset()
            # terminate the episode for all "sub-envs"
            dones = [True] * len(dones)


        # print("obs total", obs)

        return obs, rewards, dones, infos

    # ---------- Recording helpers ----------
    def enable_recording(self, out_dir: str, fps: int = 30, prefix: str = "scene"):
        """Enable recording. Writers will be opened lazily at first render of each episode/trajectory."""
        self._record_enabled = True
        self._record_dir = out_dir
        self._record_fps = fps
        self._record_prefix = prefix
        os.makedirs(out_dir, exist_ok=True)
        # force (re)open on next render with the current trajectory tag
        self._record_refresh_needed = True
        print(f"[record] enabled: dir={out_dir}, fps={fps}, prefix={prefix}")

    def disable_recording(self):
        self._record_enabled = False
        self._close_writers()

    def _close_writers(self):
        if self._writers:
            for w in self._writers:
                try:
                    w.close()
                except Exception:
                    pass
        self._writers = None

    def close(self):
        self._close_writers()
        try:
            return super().close()
        except Exception:
            return

    def _current_traj_tag(self):
        """
        Build a stable tag for filenames using your trajectory index.
        For dataset modes you increment init_pos_index after picking the entry,
        so the one used in *this* episode is (init_pos_index - 1) % max_init_pos.
        """
        try:
            used_idx = (self.init_pos_index - 1) % self.max_init_pos
            return f"traj_{used_idx:04d}"
        except Exception:
            # fallback if not available (e.g., random mode)
            return "traj_random"

    def _open_writers_for_current_episode(self):
        if not self._record_enabled or len(self.scenes) == 0:
            return

        cur_tag = self._current_traj_tag()
        if (self._writers is not None) and (self._record_traj_tag == cur_tag) and not self._record_refresh_needed:
            return

        self._close_writers()
        self._record_traj_tag = cur_tag
        self._record_refresh_needed = False

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._writers = []
        self._writer_paths = []
        self._writer_modes = []  # 'ffmpeg' or 'gif'

        for i, _ in enumerate(self.scenes):
            mp4_path = os.path.join(self._record_dir, f"{self._record_prefix}{i}_{cur_tag}_{ts}.mp4")
            try:
                # Prefer ImageIO v2 + FFMPEG backend (works reliably if imageio-ffmpeg is installed)
                # w = imageio.get_writer(
                #     mp4_path, format="FFMPEG",
                #     fps=self._record_fps, codec="libx264", pixelformat="yuv420p"
                # )
                w = imageio.get_writer(
                    mp4_path,
                    format="FFMPEG",
                    fps=self._record_fps,
                    codec="libx264",
                    pixelformat="yuv420p",  # keep for compatibility; yuv444p = sharper text but less compatible
                    ffmpeg_params=[
                        "-crf", "2",  # 18~20 is visually lossless-ish; lower = higher quality
                        "-preset", "slow",  # slow|medium|fast (slow = better compression quality)
                        "-tune", "animation",  # or "film" / "ssim" (optional, try removing if colors look off)
                        "-movflags", "+faststart"
                        # Optional: force bitrate instead of CRF (pick one approach, not both):
                        # "-b:v", "6M",          # ~6 Mbps
                        # "-maxrate", "8M", "-bufsize", "16M",
                    ],
                    # Avoid macroblock-alignment resizing:
                    macro_block_size=None,
                )

                self._writers.append(w)
                self._writer_paths.append(mp4_path)
                self._writer_modes.append("ffmpeg")
            except Exception:
                # Fallback: animated GIF (no ffmpeg needed)
                gif_path = mp4_path[:-4] + ".gif"
                w = imageio.get_writer(gif_path, format="GIF", duration=1.0 / max(1, self._record_fps), loop=0)
                self._writers.append(w)
                self._writer_paths.append(gif_path)
                self._writer_modes.append("gif")

        print(f"[record] writers opened for {len(self._writers)} scenes:", *self._writer_paths, sep="\n  - ")

    def render(self, mode='human', verbose=False):
        models = tuple(e.dynamics.model for e in self.envs)
        # print("scene length", len(self.scenes))
        if len(self.scenes) == 0:
            self.init_scene_multi()

        if self.reset_scene:
            for i in range(len(self.scenes)):
                self.scenes[i].update_models(models)
                self.scenes[i].formation_size = self.quads_formation_size
                self.scenes[i].update_env(self.room_dims)

                self.scenes[i].reset(tuple(e.goal for e in self.envs), self.all_dynamics(), self.obstacles,
                                     self.all_collisions)

            self.reset_scene = False

        if self.quads_mode == "mix":
            for i in range(len(self.scenes)):
                self.scenes[i].formation_size = self.scenario.scenario.formation_size
        else:
            for i in range(len(self.scenes)):
                self.scenes[i].formation_size = self.scenario.formation_size
        self.frames_since_last_render += 1

        if self.render_skip_frames > 0:
            self.render_skip_frames -= 1
            return None

        # this is to handle the 1st step of the simulation that will typically be very slow
        if self.simulation_start_time > 0:
            simulation_time = time.time() - self.simulation_start_time
        else:
            simulation_time = 0

        realtime_control_period = 1 / self.control_freq

        render_start = time.time()
        goals = tuple(e.goal for e in self.envs)
        frames = []
        first_spawn = None
        for i in range(len(self.scenes)):
            frame, first_spawn = self.scenes[i].render_chase(all_dynamics=self.all_dynamics(), goals=goals,
                                                             collisions=self.all_collisions,
                                                             mode=mode, obstacles=self.obstacles,
                                                             first_spawn=first_spawn)
            frames.append(frame)
        ## added for recording
        # after you fill frames[] from the human pass
        if self._record_enabled:
            self._open_writers_for_current_episode()

            # If the window path returned frames[i] (some visualizers do), prefer those:
            # Ensure frame is uint8 RGB without alpha
            rgb_frames = []
            for f in frames:
                if f is None:
                    rgb_frames.append(None)
                else:
                    f = np.ascontiguousarray(f[..., :3]).astype(np.uint8)
                    rgb_frames.append(f)

            # If frames[] are None (your window path returns None), fall back to the offscreen pass:
            if all(fr is None for fr in rgb_frames):
                rgb_frames = []
                for i in range(len(self.scenes)):
                    f_off, _ = self.scenes[i].render_chase(
                        all_dynamics=self.all_dynamics(), goals=goals, collisions=self.all_collisions,
                        mode="rgb_array", obstacles=self.obstacles, first_spawn=None
                    )
                    f_off = np.ascontiguousarray(f_off[..., :3]).astype(np.uint8) if f_off is not None else None
                    rgb_frames.append(f_off)

            for i, f in enumerate(rgb_frames):
                if f is None:
                    continue
                w = self._writers[i]
                if hasattr(w, "append_data"):
                    w.append_data(f)
                else:
                    w.write(f)

                self._record_frame_count = getattr(self, "_record_frame_count", 0) + 1
                if self._record_frame_count % 60 == 0:
                    print(f"[record] wrote frame #{self._record_frame_count}")


        # Update the formation size of the scenario
        if self.quads_mode == "mix":
            for i in range(len(self.scenes)):
                self.scenario.scenario.update_formation_size(self.scenes[i].formation_size)
        else:
            for i in range(len(self.scenes)):
                self.scenario.update_formation_size(self.scenes[i].formation_size)

        render_time = time.time() - render_start

        desired_time_between_frames = realtime_control_period * self.frames_since_last_render / self.render_speed
        time_to_sleep = desired_time_between_frames - simulation_time - render_time

        # wait so we don't simulate/render faster than realtime
        if mode == "human" and time_to_sleep > 0:
            time.sleep(time_to_sleep)

        if simulation_time + render_time > desired_time_between_frames:
            self.render_every_nth_frame += 1
            if verbose:
                print(f"Last render + simulation time {render_time + simulation_time:.3f}")
                print(f"Rendering does not keep up, rendering every {self.render_every_nth_frame} frames")
        elif simulation_time + render_time < realtime_control_period * (
                self.frames_since_last_render - 1) / self.render_speed:
            self.render_every_nth_frame -= 1
            if verbose:
                print(f"We can increase rendering framerate, rendering every {self.render_every_nth_frame} frames")

        if self.render_every_nth_frame > 5:
            self.render_every_nth_frame = 5
            if self.envs[0].tick % 20 == 0:
                print(f"Rendering cannot keep up! Rendering every {self.render_every_nth_frame} frames")

        self.render_skip_frames = self.render_every_nth_frame - 1
        self.frames_since_last_render = 0

        self.simulation_start_time = time.time()

        if mode == "rgb_array":
            return frame

    def __deepcopy__(self, memo):
        """OpenGL scene can't be copied naively."""

        cls = self.__class__
        copied_env = cls.__new__(cls)
        memo[id(self)] = copied_env

        # this will actually break the reward shaping functionality in PBT, but we need to fix it in SampleFactory, not here
        skip_copying = {"scene", "reward_shaping_interface"}

        for k, v in self.__dict__.items():
            if k not in skip_copying:
                setattr(copied_env, k, deepcopy(v, memo))

        # warning! deep-copied env has its scene uninitialized! We need to reuse one from the existing env
        # to avoid creating tons of windows
        copied_env.scene = None

        return copied_env
