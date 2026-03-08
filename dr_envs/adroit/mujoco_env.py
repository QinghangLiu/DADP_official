import os
import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.utils import seeding
import numpy as np
from os import path
import six
import time as timer

try:
    import mujoco_py
    from mujoco_py import load_model_from_path, load_model_from_xml, MjSim, MjViewer, MjRenderContextOffscreen
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

def get_sim(model_path, model_xml):
    if model_xml is None:
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        model = load_model_from_path(fullpath)
    else:
        model = load_model_from_xml(model_xml)
    return MjSim(model)

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip=1, model_xml=None, sim=None, render_mode=None):

        if sim is None:
            self.sim = get_sim(model_path=model_path, model_xml=model_xml)
        else:
            self.sim = sim
        self.data = self.sim.data
        self.model = self.sim.model

        self.frame_skip = frame_skip
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.mujoco_render_frames = False
        self.render_mode = render_mode

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        try:
            step_result = self.step(np.zeros(self.model.nu))
            if len(step_result) == 4:
                observation, _reward, done, _info = step_result
            elif len(step_result) == 5:
                observation, _reward, terminated, truncated, _info = step_result
                done = terminated or truncated
            else:
                raise ValueError(f"step() returned {len(step_result)} values, expected 4 or 5")
        except NotImplementedError:
            step_result = self._step(np.zeros(self.model.nu))
            if len(step_result) == 4:
                observation, _reward, done, _info = step_result
            elif len(step_result) == 5:
                observation, _reward, terminated, truncated, _info = step_result
                done = terminated or truncated
            else:
                raise ValueError(f"_step() returned {len(step_result)} values, expected 4 or 5")
        
        assert not done
        self.obs_dim = np.sum([o.size for o in observation]) if type(observation) is tuple else observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high, dtype=np.float64)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def mj_viewer_setup(self):
        """
        Due to specifics of new mujoco rendering, the standard viewer cannot be used
        with this set-up. Instead we use this mujoco specific function.
        """
        pass

    def viewer_setup(self):
        """
        Does not work. Use mj_viewer_setup() instead
        """
        pass

    def evaluate_success(self, paths, logger=None):
        """
        Log various success metrics calculated based on input paths into the logger
        """
        pass

    # -----------------------------

    def reset(self,seed=None, options=None):
        self.sim.reset()
        self.sim.forward()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        for i in range(self.model.nu):
            self.sim.data.ctrl[i] = ctrl[i]
        for _ in range(n_frames):
            self.sim.step()
            if self.mujoco_render_frames is True:
                self.mj_render()

    def mj_render(self):
        if self.render_mode == 'rgb_array':
            if not hasattr(self, 'viewer') or self.viewer is None:
                self.mj_viewer_setup()
            
            self.viewer.render(width=480, height=480)
            return self.viewer.read_pixels(width=480, height=480, depth=False)[::-1, :, :]
        
        try:
            self.viewer.render()
        except:
            self.mj_viewer_setup()
            self.viewer._run_speed = 0.5
            self.viewer._run_speed /= self.frame_skip
            self.viewer.render()

    def render(self, *args, **kwargs):
        return self.mj_render()

