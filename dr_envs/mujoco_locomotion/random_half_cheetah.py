"""Implementation of the HalfCheetah environment supporting
domain randomization optimization.

Randomizations:
    - 7 mass links
    - 1 friction coefficient (sliding)

For all details: https://www.gymlibrary.ml/environments/mujoco/half_cheetah/
"""
import numpy as np
import gymnasium as gym
from gymnasium import utils
from dr_envs.mujoco_locomotion.jinja_mujoco_env import MujocoEnv
from copy import deepcopy
import pdb
from itertools import product
class RandomHalfCheetah(MujocoEnv, utils.EzPickle):
    def __init__(self, noisy=False):
        self.original_lengths = np.array([1., .15, .145, .15, .094, .133, .106, .07])
        self.model_args = {"size": list(self.original_lengths)}


        self.noisy = noisy
        # Rewards:
        #   noiseless: 5348
        #   1e-5: 5273 +- 75
        #   1e-4: 4793 +- 804
        #   1e-3: 2492 +- 472
        #   1e-2: 
        self.noise_level = 1e-4


        MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])
        self.original_friction = np.array([0.4])
        self.nominal_values = np.concatenate([self.original_masses, self.original_lengths, self.original_friction])
        self.task_dim = self.nominal_values.shape[0]
        self.current_lengths = np.array(self.original_lengths)
        self.default_task = self.get_task()
        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)
        # self.observation_space = gym.spaces.Box(-10e5, 10e5, shape=(17,), dtype=np.float32)
        mass_names = ['torso', 'bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
        size_names = [f'size{i}' for i in range(len(self.original_lengths))]
        friction_name = ['friction']
        names = mass_names + size_names + friction_name
        self.dyn_ind_to_name = {i: name for i, name in enumerate(names)}

        self.preferred_lr = 0.0005 # --algo Sac -t 5M
        self.reward_threshold = 4500
        self.max_train_step = 2000000
        self.least_train_step = 200000
        self.dynamics_group = [(1,2,3), (7,),(8,),(12,13,14)]
        values = [i for i in range(1,4)]
        # Use the full Cartesian product (3^len(dynamics_group)=81) so task indices can be partitioned across GPUs
        self.task_seq = list(product(values, repeat=len(self.dynamics_group)))
        self.task_num = 0
    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized"""
        search_bounds_mean = {
               'torso': (4, 10),
               'bthigh': (2, 10.0),
               'bshin': (2, 10.0),
               'bfoot': (2, 10.0),
               'fthigh': (2, 10.0),
               'fshin': (2, 10.0),
               'ffoot': (2, 10.0),
             'size0': (0.2, 1.6),
             'size1': (0.05, 0.4),
             'size2': (0.05, 0.3),
             'size3': (0.05, 0.3),
             'size4': (0.05, 0.3),
             'size5': (0.05, 0.4),
             'size6': (0.05, 0.8),
             'size7': (0.05, 0.25),
             'friction': (1, 2.0),
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
                    'torso': 0.1,
                    'bthigh': 0.1,
                    'bshin': 0.1,
                    'bfoot': 0.1,
                    'fthigh': 0.1,
                    'fshin': 0.1,
                    'ffoot': 0.1,
                    'size0': 0.05,
                    'size1': 0.01,
                    'size2': 0.01,
                    'size3': 0.01,
                    'size4': 0.01,
                    'size5': 0.01,
                    'size6': 0.01,
                    'size7': 0.01,
                    'friction': 0.02,
        }

        return lowest_value[self.dyn_ind_to_name[index]]
    def get_all_task_upper_bound(self):
        """Returns highest feasible value for each dynamics parameter.

        This uses the upper bound returned by `get_search_bounds_mean(index)`
        for each index and returns a numpy array shaped (task_dim,).
        """
        upper_bound = np.zeros(self.task_dim)
        for i in range(self.task_dim):
            bounds = self.get_search_bounds_mean(i)
            upper_bound[i] = bounds[1]
        return upper_bound
    def get_task(self):
        masses = np.array(self.sim.model.body_mass[1:])
        friction = np.array(self.sim.model.pair_friction[0,0])
        return np.concatenate([masses, self.current_lengths, friction.reshape(1)])

    def set_task(self, *task):
        task = np.array(task, dtype=float)
        n_mass = len(self.original_masses)
        n_size = len(self.original_lengths)
        mass_vals = task[:n_mass]
        size_vals = task[n_mass:n_mass+n_size]
        friction_val = task[-1]

        # Update lengths and rebuild to apply sizes
        self.current_lengths = np.array(size_vals)
        self.model_args = {"size": list(self.current_lengths)}
        self.build_model()

        self.sim.model.body_mass[1:] = mass_vals
        self.sim.model.pair_friction[0:2,0:2] = friction_val


    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, False, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        obs = np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

        if self.noisy:
            obs += np.sqrt(self.noise_level)*np.random.randn(obs.shape[0])

        return obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        if self.dr_training:
            self.set_random_task() # Sample new dynamics

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_sim_state(self, mjstate):
        mjstate = self.sim.set_state(mjstate)

    def get_sim_state(self):
        return self.sim.get_state()
    
    def get_random_task(self):
        bound = np.zeros((self.dyn_ind_to_name.__len__(),2))
        for i in range(self.dyn_ind_to_name.__len__()):
            bound[i,:] = np.array(self.get_search_bounds_mean(i))
        # series_task = np.linspace(bound[:,0],bound[:,1], 5)
        # Use geometric (log-space) progression for equal proportional increase
        series_task = np.exp(np.linspace(np.log(bound[:,0]), np.log(bound[:,1]), 5))
        # Start from default so non-randomized parameters stay unchanged
        task = np.array(self.default_task, dtype=float)

        for group_idx, indices in enumerate(self.dynamics_group):
            value_idx = int(self.task_seq[self.task_num][group_idx])
            for param_idx in indices:
                task[param_idx] = series_task[value_idx, param_idx]

        self.task_num = (self.task_num + 1) % len(self.task_seq)
        return task

gym.register(
        id="RandomHalfCheetah-v0",
        entry_point="%s:RandomHalfCheetah" % __name__,
        max_episode_steps=1000
)

gym.register(
        id="RandomHalfCheetahNoisy-v0",
        entry_point="%s:RandomHalfCheetah" % __name__,
        max_episode_steps=1000,
        kwargs={"noisy": True}
)
