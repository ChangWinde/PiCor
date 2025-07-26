#!/usr/bin/env python3
import os
import pickle as pkl
import random
import time
from datetime import datetime

import numpy as np
import torch
import wandb
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer

torch.set_float32_matmul_precision("high")

from agent.picor import PiCor
from tools import (
    create_lr_schedulers,
    eval_mode,
    load_config,
    log,
    make_mt8_envs,
    make_mt10_envs,
    update_config,
)


class PiCorRunner:
    """
    PiCor training runner that manages the training loop and evaluation.

    This class handles the complete training pipeline including environment setup,
    agent initialization, training loop, evaluation, and logging.
    """

    RUNNER_NAME = "PiCor"

    def __init__(self, cfg):
        """
        Initialize the PiCor runner.

        Args:
            cfg: Configuration object containing training parameters
        """
        self.cfg = cfg

        self._setup_basic()
        self._setup_seed()
        self._setup_logger()
        self._setup_training()

        self.run = self.eval if self.cfg.exec_type == "eval" else self.train

    def _setup_basic(self):
        """Setup basic configuration and output directory."""
        self.output_dir = os.path.join(
            os.path.dirname(os.getcwd()),
            "results",
            f"{self.RUNNER_NAME}_{self.cfg.env_name}",
            time.strftime("%Y%m%d-%H%M%S"),
        )
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"workspace: {self.output_dir}")

        self.step = 0
        self.algo_args = load_config(self.RUNNER_NAME, self.cfg.config_path)
        self.device = torch.device(f"cuda:{self.cfg.device}")
        self.num_task = self.cfg.num_task

        meta_file = os.path.join(self.output_dir, "metadata.pkl")
        pkl.dump({"cfg": self.cfg}, open(meta_file, "wb"))

    def _setup_seed(self):
        """Setup random seeds for reproducibility."""
        seed = self.cfg.seed
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def _setup_logger(self):
        """Setup logging and wandb configuration."""
        # setup logger
        self.log_file = open(os.path.join(self.output_dir, "log.txt"), "w")

        # setup wandb
        monitor_config = update_config(self.cfg, self.algo_args)

        if self.cfg.wandb:
            wandb.login(key=os.environ.get("WANDB_API_KEY"))

            group = self.cfg.experiment
            date = datetime.strftime(datetime.now(), "%m%d")
            name = f"{self.cfg.experiment}_{str(self.cfg.seed)}_{date}"
            wandb.init(
                project=os.environ.get("WANDB_PROJECT"),
                entity=os.environ.get("WANDB_ENTITY"),
                config=monitor_config,
                tags=[self.cfg.env_name, self.cfg.experiment.lower()],
                save_code=True,
                group=group,
                name=name,
            )

    def _setup_training(self):
        """Setup training environment, agent, and buffers."""
        # setup env
        if "metaworld" in self.cfg.env_name:
            self.envs = make_mt10_envs(self.cfg)[: self.num_task]
        else:
            self.envs = make_mt8_envs(self.cfg)

        # setup variables
        self.max_ep_len = self.envs[0]._max_episode_steps
        self.num_updates = self.cfg.num_train_steps * self.num_task
        self.act_dim = self.envs[0].action_space.shape[0]
        self.obs_dim = self.envs[0].observation_space.shape[0]
        self.act_range = [
            float(self.envs[0].action_space.low.min()),
            float(self.envs[0].action_space.high.max()),
        ]
        self.env_cfg = {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "act_range": self.act_range,
        }
        self.cfg.max_ep_len = self.max_ep_len

        # setup agent
        self.agent = PiCor(self.cfg, self.algo_args, self.env_cfg, self.device)

        self.buffers = [
            ReplayBuffer(storage=LazyTensorStorage(self.cfg.buffer_size, device=self.device))
            for _ in range(self.num_task)
        ]

        # setup lr schedulers
        self.lr_schedulers = create_lr_schedulers(
            {
                "actor": self.agent.actor_optimizer,
                "critic": self.agent.critic_optimizer,
                "alpha": self.agent.log_alpha_optimizer,
            },
            self.num_updates,
            type="linear",
        )

    @torch.no_grad()
    def eval(self):
        """
        Evaluate the agent on all tasks.

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        total_episodes = self.cfg.num_eval_episodes * self.num_task
        mean_episode_reward, mean_success_rate = 0, 0

        for task_i in range(self.num_task):
            for _ in range(self.cfg.num_eval_episodes):
                obs = self.envs[task_i].reset()[0]
                done = False
                episode_reward, episode_success = 0, 0

                while not done:
                    with eval_mode(self.agent):
                        action = self.agent.act(obs, task_i, sample=False)
                    obs, reward, done, extra = self.envs[task_i].step(action)

                    episode_reward += reward
                    episode_success = max(episode_success, extra.get("success", 0))

                mean_episode_reward += episode_reward
                mean_success_rate += episode_success

        mean_episode_reward /= total_episodes
        mean_success_rate = (mean_success_rate / total_episodes) * 100.0

        log_dict = {
            "eval/episode_reward": mean_episode_reward,
            "eval/episode_success": mean_success_rate,
        }
        return log_dict

    def train(self):
        """
        Main training loop for PiCor agent.

        This method implements the complete training pipeline including data collection,
        policy updates, and periodic evaluation.
        """
        episode, done = 0, True

        while self.step < self.cfg.num_train_steps * self.num_task:
            avg_episode_reward, avg_episode_success = 0, 0
            train_duration, sample_duration = 0, 0

            start_time = time.time()
            eval_log_dict, train_log_dict = {}, {}
            for task_i in range(self.num_task):
                if done:
                    obs = self.envs[task_i].reset()[0]
                    done = False
                    episode_step = 0
                    episode += 1

                episode_reward, episode_success = 0, 0
                while not done:

                    # sample action
                    sample_start_time = time.time()
                    if self.step < self.cfg.num_random_steps * self.num_task:
                        action = self.envs[task_i].action_space.sample()
                    else:
                        with torch.no_grad():
                            action = self.agent.act(obs, task_i, sample=True)
                    sample_duration += time.time() - sample_start_time

                    # collect data
                    next_obs, reward, done, extra = self.envs[task_i].step(action)
                    done_no_max = 0 if episode_step + 1 == self.max_ep_len else float(done)
                    episode_reward += reward
                    episode_success = max(episode_success, extra.get("success", 0))

                    transition = make_transition(
                        obs, action, reward, next_obs, done_no_max, task_i, self.device
                    )
                    obs = next_obs
                    self.buffers[task_i].extend(transition)  # type: ignore
                    data = self.buffers[task_i].sample(self.algo_args["sac"]["batch_size"])

                    # training
                    train_start_time = time.time()
                    if self.step > self.cfg.num_random_steps * self.num_task:
                        # Policy updating
                        train_log_dict = self.agent.update(data, self.step)
                        # Policy correction
                        if (
                            self.cfg.enable_corr
                            and self.step % (self.max_ep_len * self.num_task) == 0
                        ):
                            self.agent.projection(self.buffers)

                        for scheduler in self.lr_schedulers.values():
                            scheduler.step()
                    train_duration += time.time() - train_start_time

                    episode_step += 1
                    self.step += 1
                avg_episode_reward += episode_reward
                avg_episode_success += episode_success * 100.0

            # evaluate agent periodically
            if self.step % (self.cfg.eval_frequency * self.num_task) == 0:
                eval_log_dict = self.eval()

            # logging metrics
            log_dict = {
                "charts/update": self.step,
                "charts/sample_duration": sample_duration,
                "charts/train_duration": train_duration,
                "charts/duration": time.time() - start_time,
                "train/episode_reward": avg_episode_reward / self.num_task,
                "train/episode_success": avg_episode_success / self.num_task,
            }
            log_dict.update(eval_log_dict)
            log(log_dict, self.RUNNER_NAME, self.log_file)

            if self.cfg.wandb:
                log_dict.update(train_log_dict)
                wandb.log(log_dict, step=self.step)

        # save final model
        self.agent.save(self.output_dir)
        self.log_file.close()


def make_transition(obs, action, reward, next_obs, done_no_max, task_i, device) -> TensorDict:
    """
    Create a transition tensor dictionary for replay buffer.

    Args:
        obs: Current observation
        action: Action taken
        reward: Reward received
        next_obs: Next observation
        done_no_max: Done flag (without max episode length)
        task_i: Task index
        device: Device to place tensors on

    Returns:
        TensorDict: Transition data
    """
    obs = torch.as_tensor(np.array([obs]), device=device, dtype=torch.float32)
    next_obs = torch.as_tensor(np.array([next_obs]), device=device, dtype=torch.float32)
    action = torch.as_tensor(np.array([action]), device=device, dtype=torch.float32)
    reward = torch.as_tensor(np.array([reward]), device=device, dtype=torch.float32)
    done_no_max = torch.as_tensor(np.array([done_no_max]), device=device, dtype=torch.bool)
    task_i = torch.as_tensor(np.array([task_i]), device=device, dtype=torch.long)

    return TensorDict(
        observations=obs,  # type: ignore
        next_observations=next_obs,  # type: ignore
        actions=action,  # type: ignore
        rewards=reward,  # type: ignore
        dones=done_no_max,  # type: ignore
        task_i=task_i,  # type: ignore
        batch_size=obs.shape[0],
        device=device,
    )
