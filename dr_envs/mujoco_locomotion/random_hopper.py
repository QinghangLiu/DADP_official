"""Implementation of the Hopper environment supporting
domain randomization optimization.

For all details: https://www.gymlibrary.ml/environments/mujoco/hopper/
"""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gymnasium as gym
from gymnasium import utils
from dr_envs.mujoco_locomotion.jinja_mujoco_env import MujocoEnv
from scipy.stats import truncnorm
from itertools import product
class RandomHopperEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, noisy=False):
        self.original_lengths = np.array([.4, .45, 0.5, .39])
        self.model_args = {"size": list(self.original_lengths)}

        MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

        # Define randomized dynamics
        self.dyn_ind_to_name = {0: 'torsomass', 1: 'thighmass', 2: 'legmass', 3: 'footmass',
                                4: 'damping0', 5: 'damping1', 6: 'damping2', 7: 'friction'}

        self.default_task = self.get_task()
        self.set_task(*self.default_task)
        self.original_masses = np.copy(self.get_task())
        self.nominal_values = np.concatenate([self.original_masses])
        self.task_dim = self.nominal_values.shape[0]

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.preferred_lr = 0.0003
        self.reward_threshold = 1600
        self.max_train_step = 6000000
        self.least_train_step = 4000000
        self.dynamics_group = [(4,),(5,),(6,),(7,)]
        values = [i for i in range(1,4)]
        self.task_seq = list(product(values, repeat=len(self.dynamics_group)))
        self.task_num = 0

    def get_default_task(self):
        mean_of_search_bounds = np.array([(self.get_search_bounds_mean(i)[0] + self.get_search_bounds_mean(i)[1])/2 for i in range(len(self.dyn_ind_to_name.keys()))])
        return mean_of_search_bounds

    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized"""
        search_bounds_mean = {
               'torsomass': (2.35, 5.3),
               'thighmass': (2.6, 5.9),
               'legmass': (1.8, 4.5),
               'footmass': (3.4, 7.63),
               'damping0': (0.6, 1.5),
               'damping1': (0.6, 1.5),
               'damping2': (0.6, 1.5),
               'friction': (0.6, 1.5)
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
                    'torsomass': 0.001,
                    'thighmass': 0.001,
                    'legmass': 0.001,
                    'footmass': 0.001,
                    'damping0': 0.05,
                    'damping1': 0.05,
                    'damping2': 0.05,
                    'friction': 0.01
        }

        return lowest_value[self.dyn_ind_to_name[index]]

    def get_all_task_upper_bound(self):
        """Returns highest feasible value for each dynamics parameter.

        This uses the upper bound returned by `get_search_bounds_mean(index)`
        for each index and returns a numpy array shaped (task_dim,).
        """
        upper_bound = np.zeros(self.task_dim)
        for i in range(self.task_dim):
            # get_search_bounds_mean returns (low, high)
            bounds = self.get_search_bounds_mean(i)
            upper_bound[i] = bounds[1]
        return upper_bound


    def get_task(self):
        masses = np.array( self.sim.model.body_mass[1:] )
        damping = np.array( self.sim.model.dof_damping[3:] )
        friction = np.array( [self.sim.model.pair_friction[0, 0]] )
        return np.concatenate([masses, damping, friction])

    def set_task(self, *task):
        self.sim.model.body_mass[1:] = task[:4]
        self.sim.model.dof_damping[3:] = task[4:7]  # damping on the three actuated joints
        self.sim.model.pair_friction[0, :2] = np.repeat(task[7], 2)


    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, False,{}

    def _get_obs(self):
        obs = np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
            # np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

        return obs

    def reset_model(self):
        if self.dr_training:
            self.set_random_task() # Sample new dynamics

        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
            
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def set_sim_state(self, mjstate):
        return self.sim.set_state(mjstate)

    def get_sim_state(self):
        return self.sim.get_state()

    def get_random_task(self,task_num=None):
        if task_num is not None:
            self.task_num = task_num
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
        task = np.concatenate([self.default_task[:4],
            series_task[task_index,np.arange(len(self.default_task))][4:]])
        self.task_num = (self.task_num + 1) % len(self.task_seq)
        return task
    
gym.register(
        id="RandomHopper-v0",
        entry_point="%s:RandomHopperEnv" % __name__,
        max_episode_steps=1000,
)