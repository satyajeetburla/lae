import pickle
import os
import numpy as np
import sys

import h5py
import json

class DataLogger():
    def __init__(self, num_agents, ep_duration, save_path, num_traj):
        self.save_directory = save_path
        self.num_agents = num_agents
        self.num_traj = num_traj

        # [x,y,z] location of cylinder obstacle center
        self.obst_map = None
        # Radius of cylinder obstacle
        self.obst_r = None
        # Obstacle density of the current run
        self.obst_density = None

        self.file_id = 0
        self.ep_duration = ep_duration

        # The planned trajectory during simulation
        self.desired_traj = np.zeros([self.ep_duration + 1, self.num_agents, 3])

        # The visited states [x,y,z,vx,vy,vz]
        self.states = np.zeros([self.ep_duration + 1, self.num_agents, 6])

        # Action array for each agent
        self.quadrotor_actions = np.zeros([self.ep_duration + 1, self.num_agents, 4])

        # used to train the behavior classifier
        self.attention_latent = np.zeros([self.ep_duration + 1, self.num_agents, 30])
        
        # List of dictionaries that will keep the collision information
        self.collision_information = []
        self.last_collision = None                # new: track only the final dict

        
        # Current tick of simulation for logging purposes
        self.tick = 0

        self.inference_mode = True


    def save_obst_collision_point(self, state):
        self.collision_obst_loc.append(state[:3])

    def save_quad_collision_point(self, state):
        self.collision_quad_loc.append(state[:3])

    def save_step(self, state, goal, action):
        self.states[self.tick] = state
        self.desired_traj[self.tick] = goal
        self.quadrotor_actions[self.tick] = action
        self.tick += 1
    
    def save_latent(self, latent):
        latent = latent.cpu().detach().numpy() # [num_agents, 30] 
        
        self.attention_latent[self.tick] = latent
        

    def clear_data(self):
        self.desired_traj = np.zeros([self.ep_duration + 1, self.num_agents, 3])
        self.states = np.zeros([self.ep_duration + 1, self.num_agents, 6])
        self.attention_latent = np.zeros([self.ep_duration + 1, self.num_agents, 30])
        self.obst_map = None
        self.cell_centers = None
        self.tick = 0
        self.collision_information = []

    def log_obstacle_data(self, obst_map, obst_size, obst_density):
        self.obst_map = obst_map
        self.obst_r = obst_size * 0.5
        self.obst_density = obst_density

    # def save_class(self):
    #     filename = self.save_directory + "/Run_" + str(self.file_id) + '.pkl'
    #     os.makedirs(os.path.dirname(filename), exist_ok=True)
    #
    #     with open(filename, 'wb') as outp:
    #         pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
    #
    #     self.file_id += 1
    #
    #     if self.file_id == self.num_traj:
    #         sys.exit(0)
    def save_class(self):
        """
        - If inference_mode=False: pickle the full logger (unchanged).
        - If inference_mode=True: write only the last-step arrays and final collision dict.
        """
        if self.save_directory is None:
            self.file_id += 1
            if self.num_traj is not None and self.file_id == self.num_traj:
                sys.exit(0)
            return

        base_filename = f"Run_{self.file_id}"
        extension = "h5" if self.inference_mode else "pkl"
        fn = os.path.join(self.save_directory, f"{base_filename}.{extension}")

        # ── 🔒 Stop-guard: prevent overwriting ──
        if os.path.exists(fn):
            print(f"\033[1;91m[⚠ GUARD] File '{fn}' already exists. Skipping save! Use a NEW folder or modify file_id.\033[0m")
            return

        # ── Ensure directory exists ──
        os.makedirs(self.save_directory, exist_ok=True)

        if not getattr(self, 'inference_mode', False):
            # Original pickle behavior
            # fn = os.path.join(self.save_directory, f"Run_{self.file_id}.pkl")
            # os.makedirs(os.path.dirname(fn), exist_ok=True)
            with open(fn, 'wb') as outp:
                pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        else:
            # HDF5 last-step exporter with collision_information

            # fn = os.path.join(self.save_directory, f"Run_{self.file_id}.h5")
            # os.makedirs(os.path.dirname(fn), exist_ok=True)
            last_idx = max(0, self.tick - 1)

            with h5py.File(fn, 'w') as f:
                # Write final-step arrays
                f.create_dataset('state',
                                 data=self.states[last_idx],
                                 compression='gzip')
                f.create_dataset('goal',
                                 data=self.desired_traj[last_idx],
                                 compression='gzip')
                f.create_dataset('action',
                                 data=self.quadrotor_actions[last_idx],
                                 compression='gzip')
                f.create_dataset('latent',
                                 data=self.attention_latent[last_idx],
                                 compression='gzip')

                # Obstacle metadata
                # if self.obst_map is not None:
                #     f.create_dataset('obst_map', data=self.obst_map)
                # if self.obst_r is not None:
                #     f.attrs['obst_r'] = self.obst_r
                # if self.obst_density is not None:
                #     f.attrs['obst_density'] = self.obst_density

                # Add the final collision_information dict as JSON
                if self.collision_information:
                    last = self.collision_information[-1]
                    f.attrs['collision_information'] = json.dumps(
                        last,
                        default=lambda o: o.tolist() if hasattr(o, 'tolist') else o
                    )

        self.file_id += 1
        if self.num_traj is not None and self.file_id == self.num_traj:
            sys.exit(0)

    def update_collision_information(self, collision_dict):
        self.collision_information.append(collision_dict)
    
    def is_empty(self):
        if not np.any(self.states):
            return True
