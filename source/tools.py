import json
import os
from typing import Any, Dict, Optional, TextIO

import gym
import yaml
import metaworld
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CyclicLR, LinearLR
from garage.experiment import MetaWorldTaskSampler
from gym.wrappers.time_limit import TimeLimit

from rlkit.envs.wrappers import NormalizedBoxEnv, NormalizedBoxEnv_v2, TaskOnehotWrapper

class eval_mode(object):
    """
    Context manager for setting models to evaluation mode.

    This context manager temporarily sets models to evaluation mode
    and restores their original training state when exiting.
    """

    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.set_training_mode(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.set_training_mode(state)
        return False


def make_mt10_envs(cfg):
    """
    Create MetaWorld MT10 environments.

    Args:
        cfg: Configuration object containing seed and num_task

    Returns:
        list: List of MetaWorld environments
    """
    seed = cfg.seed
    mt10 = metaworld.MT10(seed=cfg.seed)

    def env_wrapper(env, _):
        return TimeLimit(NormalizedBoxEnv(env, seed), env.max_path_length)

    train_task_sampler = MetaWorldTaskSampler(mt10, "train", env_wrapper, add_env_onehot=True)
    envs = [env_up() for env_up in train_task_sampler.sample(10)]
    for i in range(cfg.num_task):
        envs[i].seed(cfg.seed)
    return envs


def make_mt8_envs(cfg):
    """
    Create MT8 HalfCheetah environments.

    Args:
        cfg: Configuration object containing seed

    Returns:
        list: List of HalfCheetah environments
    """
    from gym_extensions.continuous import mujoco

    envs_list = [
        "HalfCheetahSmallFoot-v0",
        "HalfCheetahBigFoot-v0",
        "HalfCheetahSmallLeg-v0",
        "HalfCheetahBigLeg-v0",
        "HalfCheetahSmallThigh-v0",
        "HalfCheetahBigThigh-v0",
        "HalfCheetahSmallTorso-v0",
        "HalfCheetahBigTorso-v0",
    ]
    envs = []
    num_task = len(envs_list)
    for id, env_name in enumerate(envs_list):
        env = NormalizedBoxEnv_v2(
            TaskOnehotWrapper(gym.make(env_name), task_index=id, n_total_tasks=num_task)
        )
        env.seed(cfg.seed)
        envs.append(env)
    return envs


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None, layer_norm=False):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        if layer_norm:
            mods = [
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            ]
        else:
            mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            if layer_norm:
                mods += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            else:
                mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        if t.dtype == torch.bfloat16:
            return t.cpu().detach().float().numpy()
        else:
            return t.cpu().detach().numpy()


def flatten_dict(data, parent_key="", sep="_"):
    """
    Flatten a nested dictionary.

    :param d: The dictionary to flatten.
    :param parent_key: The base key for the current level of recursion.
    :param sep: The separator between parent and child keys.
    :return: A flattened dictionary.
    """
    items = []
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def update_config(cfg, algo_args):
    algo_args_flat = flatten_dict(algo_args)
    cfg_dict = vars(cfg).copy()
    cfg_dict.update(algo_args_flat)
    return cfg_dict


def load_config(config_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_path, f"{config_name}/{config_name}.yaml")
    with open(config_path, "r", encoding="utf-8") as file:
        algo_args = yaml.load(file, Loader=yaml.FullLoader)
    return algo_args  # type: ignore


def create_lr_scheduler(
    optimizer: optim.Optimizer, total_num_epochs: int, type: str = "linear", **kwargs
):
    """Creates a learning rate scheduler

    Args:
        optimizer: The optimizer whose learning rate to schedule
        total_num_epochs: Total number of epochs
        type: Type of scheduler ("linear", "cyclic" or "multistep")
        **kwargs: Additional arguments for specific schedulers:
            - For cyclic: base_lr, max_lr, step_size_up, mode
            - For multistep: milestones, gamma

    Returns:
        A PyTorch learning rate scheduler
    """
    if type == "linear":
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_num_epochs)
    elif type == "cyclic":
        base_lr = kwargs.get("base_lr", optimizer.param_groups[0]["lr"] * 0.1)
        max_lr = kwargs.get("max_lr", optimizer.param_groups[0]["lr"])
        step_size_up = kwargs.get("step_size_up", total_num_epochs // 4)
        mode = kwargs.get("mode", "triangular")

        return CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            mode=mode,
            cycle_momentum=False,
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {type}")


def create_lr_schedulers(
    optimizers: Dict[str, optim.Optimizer], total_num_epochs: int, type: str = "linear", **kwargs
):
    """Creates learning rate schedulers for multiple optimizers

    Args:
        optimizers: Dictionary mapping optimizer names to optimizers
        total_num_epochs: Total number of epochs
        type: Type of scheduler ("linear", "cyclic" or "multistep")
        **kwargs: Additional arguments for specific schedulers

    Returns:
        Dictionary mapping optimizer names to their schedulers
    """
    schedulers = {}
    for name, optimizer in optimizers.items():
        schedulers[name] = create_lr_scheduler(optimizer, total_num_epochs, type, **kwargs)
    return schedulers


def log(log_dict: Dict[str, Any], log_name: str, log_file: TextIO):
    print(f"******************{log_name.upper()} Training**********************")
    print(f"Update {log_dict['charts/update']} | Duration: {log_dict['charts/duration']:.2f} s")
    print(f"sample duration: {log_dict['charts/sample_duration']:.2f} s")
    print(f"train duration:  {log_dict['charts/train_duration']:.2f} s")
    print(f"episode_reward:  {log_dict['train/episode_reward']:.2f}")
    print(f"episode_success: {log_dict['train/episode_success']:.2f}%")
    if "eval/episode_reward" in log_dict:
        print(f"------------------{log_name.upper()} Testing-----------------------")
        print(f"episode_reward:  {log_dict['eval/episode_reward']:.2f}")
        print(f"episode_success: {log_dict['eval/episode_success']:.2f}%")
    print("******************************************************\n\n")
    log_file.write(json.dumps(log_dict, indent=2) + "\n\n")
    log_file.flush()
