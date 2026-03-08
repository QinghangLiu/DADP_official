"""Implementation of the Walker2d environment supporting
domain randomization optimization.

Randomizations:
    - 7 masses
    - 4 link lengths (torso, then the 3 links for each leg symmetrically)
    - 2 friction coefficient (sliding), separately for each foot

    All details: https://www.gymlibrary.ml/environments/mujoco/walker2d/
"""

import numpy as np
import gymnasium as gym
from gymnasium import utils
from dr_envs.mujoco_locomotion.jinja_mujoco_env import MujocoEnv
from copy import deepcopy
import pdb
from itertools import product

class RandomWalker2dEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 20,
    }
    def __init__(self, noisy=False):
        self.original_lengths = np.array([.4, .45, 0.6, .2])
        self.model_args = {"size": list(self.original_lengths)}

        self.noisy = noisy
        # Rewards:
        #   noiseless:  2426
        #   1e-5:  
        #   1e-4: 2429
        #   1e-3: 2085 +- 500
        self.noise_level = 1e-3

        MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

        self.original_masses = self.sim.model.body_mass[1:]
        self.original_friction = np.array([0.9, 1.9])
        self.nominal_values = np.concatenate([self.original_masses, self.original_lengths, self.original_friction])
        self.task_dim = self.nominal_values.shape[0]
        self.current_lengths = np.array(self.original_lengths)

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.dyn_ind_to_name = {0: 'torso', 1: 'thigh', 2: 'leg', 3: 'foot', 4: 'thigh_left', 5: 'leg_left', 6: 'foot_left', 7: 'torsosize', 8: 'thighsize', 9: 'legsize', 10: 'footsize', 11: 'friction_right', 12: 'friction_left'}
        self.default_task = self.get_task()
        self.preferred_lr = 0.0005
        self.reward_threshold = 2200
        self.max_train_step = 2000000
        self.least_train_step = 200000
        self.dynamics_group = [[0,],(1,2,3),(4,5,6),(7,8,9,10),(11,12)]
        values = [i for i in range(1,4)]
        self.task_seq = list(product(values, repeat=len(self.dynamics_group)))[:27]
        self.task_num = 0
    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized"""
        search_bounds_mean = {
               'torso': (2, 7.0),

               'thigh': (2, 10.0),
               'leg': (2, 10.0),
               'foot': (2, 10.0),

               'thigh_left': (2, 10.0),
               'leg_left': (2, 10.0),
               'foot_left': (2, 10.0),

               'torsosize': (0.2, 1.0),
               'thighsize': (0.2, 0.7),
               'legsize': (0.2, 0.7),
               'footsize': (0.2, 1.0),

               'friction_right': (1, 3.0),
               'friction_left': (1, 3.0)
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
                    'torso': 0.1,
                    'thigh': 0.1,
                    'leg': 0.1,
                    'foot': 0.1,
                    'thigh_left': 0.1,
                    'leg_left': 0.1,
                    'foot_left': 0.1,

                    'torsosize': 0.1,
                    'thighsize': 0.1,
                    'legsize': 0.1,
                    'footsize': 0.1,

                    'friction_right': 0.05,
                    'friction_left': 0.05
        }

        return lowest_value[self.dyn_ind_to_name[index]]

    def get_all_task_upper_bound(self):
        """Returns highest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        highest_value = {
                    'torso': 7.0,
                    'thigh': 10.0,
                    'leg': 10.0,
                    'foot': 10.0,
                    'thigh_left': 10.0,
                    'leg_left': 10.0,
                    'foot_left': 10.0,

                    'torsosize': 1.0,
                    'thighsize': 0.7,
                    'legsize': 0.7,
                    'footsize': 1.0,

                    'friction_right': 3.0,
                    'friction_left': 3.0
        }

        upper_bound = np.zeros(self.task_dim)
        for i in range(self.task_dim):
            highest_val = highest_value[self.dyn_ind_to_name[i]]
            upper_bound[i] = highest_val

        return upper_bound
    def get_task(self):
        masses = self.sim.model.body_mass[1:]
        friction = np.array( self.sim.model.pair_friction[0:2,0] )
        return np.concatenate((masses, self.current_lengths, friction))

    def set_task(self, *task):
        # self.current_lengths = np.array(task[-len(self.original_lengths):])
        self.current_lengths = np.array(task[len(self.original_masses):len(self.original_masses)+len(self.original_lengths)])
        self.model_args = {"size": list(self.current_lengths)}
        self.build_model()
        self.sim.model.body_mass[1:] = task[:len(self.original_masses)]
        self.sim.model.pair_friction[0,0:2] = task[-2]
        self.sim.model.pair_friction[1,0:2] = task[-1]


    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)

        ob = self._get_obs()

        return ob, reward, done, False,{}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        obs = np.concatenate([qpos[1:], qvel[:]]).ravel()

        if self.noisy:
            obs += np.sqrt(self.noise_level)*np.random.randn(obs.shape[0])

        return obs

    def reset_model(self):
        if self.dr_training:
            self.set_random_task() # Sample new dynamics

        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )        

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def set_sim_state(self, state):
        return self.sim.set_state(state)

    def get_sim_state(self):
        return self.sim.get_state()
    
    def get_random_task(self):
        bound = np.zeros((self.dyn_ind_to_name.__len__(),2))
        for i in range(self.dyn_ind_to_name.__len__()):
            bound[i,:] = np.array(self.get_search_bounds_mean(i))
        # series_task = np.linspace(bound[:,0],bound[:,1], 5)
        # Use geometric (log-space) progression for equal proportional increase
        series_task = np.exp(np.linspace(np.log(bound[:,0]), np.log(bound[:,1]), 5))
        # task_index = np.random.choice(np.arange(1,series_task.shape[0]-1), size=len(self.default_task), replace=True)
        task_index = np.zeros(len(self.default_task), dtype=int)
        for task_group in range(len(self.dynamics_group)):
            indices = list(self.dynamics_group[task_group])
            task_index[indices] = int(self.task_seq[self.task_num][task_group])
        task = series_task[task_index,np.arange(len(self.default_task))]
        self.task_num = (self.task_num + 1) % len(self.task_seq)
        return task

gym.register(
        id="RandomWalker2d-v0",
        entry_point="%s:RandomWalker2dEnv" % __name__,
        max_episode_steps=1000
)


gym.register(
        id="RandomWalker2dNoisy-v0",
        entry_point="%s:RandomWalker2dEnv" % __name__,
        max_episode_steps=1000,
        kwargs={"noisy": True}
)


