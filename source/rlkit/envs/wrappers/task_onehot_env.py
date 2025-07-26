import numpy as np
from gym.spaces import Box

from rlkit.envs.proxy_env import ProxyEnv

class TaskOnehotWrapper(ProxyEnv):
    """Append a one-hot task representation to an environment.
    Args:
        env (Environment): The environment to wrap.
        task_index (int): The index of this task among the tasks.
        n_total_tasks (int): The number of total tasks.

    """

    def __init__(self, wrapped_env, task_index, n_total_tasks):
        assert 0 <= task_index < n_total_tasks
        self._task_index = task_index
        self._n_total_tasks = n_total_tasks
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        env_lb = self._wrapped_env.observation_space.low
        env_ub = self._wrapped_env.observation_space.high
        one_hot_ub = np.ones(self._n_total_tasks)
        one_hot_lb = np.zeros(self._n_total_tasks)
        self.observation_space = Box(
            np.concatenate([env_lb, one_hot_lb]), np.concatenate([env_ub, one_hot_ub])
        )

    def reset(self):
        """Sample new task and call reset on new task env.
        Returns:
            numpy.ndarray: The first observation conforming to`observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)

        """
        first_obs = self._wrapped_env.reset()
        first_obs = self._obs_with_one_hot(first_obs)

        return [first_obs]

    def step(self, action):
        """
        Environment step for the active task env.
        Args: action (np.ndarray): Action performed by the agent in the environment.
        Returns: EnvStep: The environment step resulting from the action.
        """
        next_state, reward, done, info = self._wrapped_env.step(action)
        oh_next_state = self._obs_with_one_hot(next_state)
        info["task_id"] = self._task_index
        # modify ...
        return oh_next_state, reward, done, info

    def _obs_with_one_hot(self, obs):
        """
        Concatenate observation and task one-hot.
        Args: obs (numpy.ndarray): observation
        Returns: numpy.ndarray: observation + task one-hot.
        """
        one_hot = np.zeros(self._n_total_tasks)
        one_hot[self._task_index] = 1.0
        return np.concatenate([obs, one_hot])
