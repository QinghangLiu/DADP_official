"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import pdb
import math
import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.utils import seeding
import numpy as np
from itertools import product
from dr_envs.random_env import RandomEnv

try:
    import pygame
except ImportError:  # pragma: no cover - optional dependency
    pygame = None


class RandomCartPoleEnv(RandomEnv):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

        Augmented with Domain Randomization capabilities (see `Randomized
        Dynamics` below).

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        vanilla:
            Type: Box(4)
            Num     Observation               Min                     Max
            0       Cart Position             -4.8                    4.8
            1       Cart Velocity             -Inf                    Inf
            2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
            3       Pole Angular Velocity     -Inf                    Inf
        inverted:
            Type: Box(5)
            Num     Observation               Min                     Max
            0       Cart Position             -4.8                    4.8
            1       Cart Velocity             -Inf                    Inf
            2       Pole Cos(Angle)           -1                      1
            3       Pole Sin(Angle)           -1                      1
            4       Pole Angular Velocity     -Inf                    Inf

    Actions:
        continuous_action=False
            Type: Discrete(2)
            Num   Action
            0     Push cart to the left
            1     Push cart to the right
        continuous_action=True
            Type: Box(1)
            Num   Action                      Min                     Max
            0     Normalized torque           -1                      1

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        vanilla: Reward is 1 for every step taken, including the termination step
        inverted: Reward is -(theta^2 + 0.1*theta_dot^2 + 0.001*theta_acc^2)

    Starting State:
        vanilla: All observations are assigned a uniform random value in [-0.05..0.05]
        inverted: obs in [-0.05, 0.05], except for theta [-pi-0.05, -pi+0.05]
                  (pole hanging at the bottom)

    Episode Termination:
        vanilla:
            Pole Angle is more than 12 degrees.
            Cart Position is more than 2.4 (center of the cart reaches the edge of
            the display).
            Episode length is greater than 200.
        inverted:
            Cart Position is more than 2.4 (center of the cart reaches the edge of
            the display).
            Episode length is greater than 200.
    
    Solved Requirements: 
        vanilla: Considered solved when the average return is greater than or equal to
                 195.0 over 100 consecutive trials.

    Randomized dynamics:
        Gravity, Cart mass, Pole mass, Pole length
        4 parameters.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 inverted=False,
                 continuous_action=False,
                 version='hard'):
        """
            inverted: bool
                      If set, the task is an inverted cartpole pendulum
                      which starts in vertical position towards the bottom.

            continuous_action: bool
                               actions in range [-1, 1] are expected, which
                               get translated to forces exerted on the cart.

            version: str
                     how many dynamics parameters to randomize.
                     easy: 2, hard: 4
        """
        RandomEnv.__init__(self)

        self.inverted = inverted
        self.version = version

        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.total_mass = (self.pole_mass + self.cart_mass)
        self.pole_length = 0.5  # actually half the pole's length
        self.polemass_length = (self.pole_mass * self.pole_length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Cart pos at which to fail the episode
        self.x_threshold = 2.4


        ### Observation space
        if not self.inverted:
            # Angle at which to fail the episode
            self.theta_threshold_radians = 12 * 2 * math.pi / 360
            
            # Angle limit set to 2 * theta_threshold_radians so failing observation
            # is still within bounds.
            high = np.array([self.x_threshold * 2,  # x
                             np.finfo(np.float32).max,  # x_dot
                             self.theta_threshold_radians * 2,  # theta
                             np.finfo(np.float32).max],  # theta_dot
                            dtype=np.float32)

            self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        else:
            high = np.array([self.x_threshold * 2,  # x
                             np.finfo(np.float32).max,  # x_dot
                             1,  # cos(theta)
                             1,  # sin(theta)
                             np.finfo(np.float32).max],  # theta_dot
                            dtype=np.float32)

            self.observation_space = spaces.Box(-high, high, dtype=np.float32)


        ### Action space
        self.continuous_action = continuous_action
        if self.continuous_action:
            self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(2)

        self.seed()
        self.viewer = None
        self.window = None
        self.clock = None
        self.surface = None
        self.state = None
        self.steps_beyond_done = None
        self.isopen = True


        ### Domain randomization
        if version == 'hard':
            self.dyn_ind_to_name = {0: 'gravity',
                                    1: 'cart_mass',
                                    2: 'pole_mass',
                                    3: 'pole_length'}
            self.original_task = np.array([
                                    self.gravity,
                                    self.cart_mass,
                                    self.pole_mass,
                                    self.pole_length
                                ])

        elif version == 'easy':
            self.dyn_ind_to_name = {0: 'gravity',
                                    1: 'pole_length'}
            self.original_task = np.array([
                                    self.gravity,
                                    self.pole_length
                                ])

        self.nominal_values = np.copy(self.original_task)

        self.task_dim = self.original_task.shape[0]
        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)
        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        if self.inverted:
            self.reward_threshold = 4000
        else:
            self.reward_threshold = 500
        self.default_task = np.copy(self.original_task)
        self.preferred_lr = 0.0005
        self.least_train_step = 500000
        self.max_train_step = 1000000
        self.dynamics_group = [(0,),(1,),(2,),(3,)]
        values = [i for i in range(1,4)]
        self.task_seq = list(product(values, repeat=len(self.dynamics_group)))
        self.task_num = 0
    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized"""
        search_bounds_mean = {
               'gravity': (2., 15.0),
               'cart_mass': (0.5, 3.0),
               'pole_mass': (0.05, 0.3),
               'pole_length': (0.1, 1.),
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
                    'gravity': 0.1,
                    'cart_mass': 0.1,
                    'pole_mass': 0.1,
                    'pole_length': 0.1
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
        """Returns current dynamics parameters"""
        full_task = {'gravity': self.gravity,
                     'cart_mass': self.cart_mass,
                     'pole_mass': self.pole_mass,
                     'pole_length': self.pole_length}

        # Return current task based on how many parameters are randomized
        curr_task = []
        for index, task_name in self.dyn_ind_to_name.items():
            curr_task.append(full_task[task_name])

        return np.array(curr_task)

    def set_task(self, *task):
        """Set dynamics parameters

            task : arr of [gravity, cart_mass, pole_mass, pole_length]
        """
        if self.version == 'hard':
            self.gravity = task[0]
            self.cart_mass = task[1]
            self.pole_mass = task[2]
            self.pole_length = task[3]
            self.total_mass = (self.pole_mass + self.cart_mass)
        elif self.version == 'easy':
            self.gravity = task[0]
            self.pole_length = task[1]
        else:
            raise ValueError(f'Given version is not compatible: {version}')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        info = {}
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        assert action <= 1 and action >= -1
        if self.continuous_action:
            # Scale action to force boundaries
            force = action[0] * self.force_mag
        else:
            force = self.force_mag if action == 1 else -self.force_mag

        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.pole_length * (4.0 / 3.0 - self.pole_mass * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        ### Done
        if self.inverted:
            done = bool(
                    x < -self.x_threshold or
                    x > self.x_threshold
            )
        else:
            done = bool(
                x < -self.x_threshold
                or x > self.x_threshold
                or theta < -self.theta_threshold_radians
                or theta > self.theta_threshold_radians
            )
        

        ### Reward
        if self.inverted:
            reward = - (self.angle_normalize(theta)**2  +  0.1*theta_dot**2  +  0.001*(thetaacc**2))

            if not done:
                reward += 10  # positive reward encourages the agent to atleast stay alive
            elif self.steps_beyond_done is None:  # done just turned True
                self.steps_beyond_done = 0
                reward += 0
            else:
                self.steps_beyond_done += 1
                reward = 0

            info['norm_theta'] = self.angle_normalize(theta)
        else:
            if not done:
                reward = 1.0
            elif self.steps_beyond_done is None:
                # Pole just fell!
                self.steps_beyond_done = 0
                reward = 1.0
            else:
                if self.steps_beyond_done == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned done = True. You "
                        "should always call 'reset()' once you receive 'done = "
                        "True' -- any further steps are undefined behavior."
                    )
                self.steps_beyond_done += 1
                reward = 0.0

        info['theta'] = theta
        info['theta_dot'] = theta_dot
        info['theta_acc'] = thetaacc

        return self._get_obs(), reward, done, False, info

    def angle_normalize(self, th):
        return ((th + np.pi) % (2 * np.pi)) - np.pi

    def _get_obs(self):
        if self.inverted:
            x, x_dot, theta, theta_dot = self.state
            return np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)
        else:
            return np.array(self.state)

    def reset(self,seed = None, options = None):
        if self.dr_training:
            self.set_random_task() # Sample new dynamics

        if self.inverted:
            # x, x_dot, theta, theta_dot
            all_but_theta = tuple(self.np_random.uniform(low=-0.05, high=0.05, size=(3,)))  # x, x_dot, theta_dot
            theta = self.np_random.uniform(low=-np.pi-0.05, high=-np.pi+0.05)  # theta
            self.state = all_but_theta[0], all_but_theta[1], theta, all_but_theta[2]
        else:
            self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None

        return self._get_obs()

    def render(self, mode='human'):
        if pygame is None:
            raise ImportError(
                "pygame is required for rendering. Install it via `pip install pygame`."
            )

        if mode not in ("human", "rgb_array"):
            raise ValueError(f"Unsupported render mode '{mode}'.")

        screen_width = 600
        screen_height = 400
        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        cart_height_center = screen_height * 0.75
        polewidth = 10
        polelen = scale * (2 * self.pole_length)
        cartwidth = 50
        cartheight = 30

        if self.surface is None:
            pygame.init()
            self.surface = pygame.Surface((screen_width, screen_height))

        if mode == "human" and self.window is None:
            pygame.display.init()
            self.window = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("RandomCartPole")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state is None:
            return None

        surface = self.surface
        surface.fill((255, 255, 255))

        track_y = cart_height_center + cartheight / 2
        pygame.draw.line(surface, (0, 0, 0), (0, int(track_y)), (screen_width, int(track_y)), 1)

        x, x_dot, theta, theta_dot = self.state
        cartx = x * scale + screen_width / 2.0
        cart_rect = pygame.Rect(0, 0, cartwidth, cartheight)
        cart_rect.center = (cartx, cart_height_center)
        pygame.draw.rect(surface, (0, 0, 0), cart_rect)

        pivot = (cartx, cart_height_center - cartheight / 2)
        pole_tip = (
            pivot[0] + polelen * math.sin(theta),
            pivot[1] - polelen * math.cos(theta),
        )
        pygame.draw.line(surface, (204, 153, 102), pivot, pole_tip, max(1, int(polewidth)))
        pygame.draw.circle(surface, (128, 128, 204), (int(pivot[0]), int(pivot[1])), max(3, polewidth // 2))

        self.isopen = True

        if mode == "human":
            pygame.event.pump()
            assert self.window is not None
            self.window.blit(surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata['video.frames_per_second'])
            return None

        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))

    def close(self):
        if pygame is not None:
            if self.window is not None:
                pygame.display.quit()
                self.window = None
            if self.surface is not None:
                self.surface = None
            if self.clock is not None:
                self.clock = None
            pygame.quit()
        self.viewer = None
        self.isopen = False

    def set_verbosity(self, verbose):
        self.verbose = verbose

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

gym.envs.register(
    id="RandomCartPoleHard-v0",
    entry_point="%s:RandomCartPoleEnv" % __name__,
    max_episode_steps=1000,
    kwargs={}
)

gym.envs.register(
    id="RandomInvertedCartPoleHard-v0",
    entry_point="%s:RandomCartPoleEnv" % __name__,
    max_episode_steps=1000,
    kwargs={"inverted": True}
)
gym.envs.register(
    id="RandomContinuousCartPoleHard-v0",
    entry_point="%s:RandomCartPoleEnv" % __name__,
    max_episode_steps=1000,
    kwargs={"inverted": False, "continuous_action": True}
)
gym.envs.register(
    id="RandomContinuousInvertedCartPoleHard-v0",
    entry_point="%s:RandomCartPoleEnv" % __name__,
    max_episode_steps=1000,
    kwargs={"inverted": True, "continuous_action": True}
)


gym.envs.register(
    id="RandomContinuousInvertedCartPoleEasy-v0",
    entry_point="%s:RandomCartPoleEnv" % __name__,
    max_episode_steps=1000,
    kwargs={"inverted": True, "continuous_action": True, "version": "easy"}
)

gym.envs.register(
    id="RandomContinuousCartPoleEasy-v0",
    entry_point="%s:RandomCartPoleEnv" % __name__,
    max_episode_steps=1000,
    kwargs={"inverted": False, "continuous_action": True, "version": "easy"}
)

