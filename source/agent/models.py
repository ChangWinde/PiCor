import math

import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn

from tools import mlp, weight_init

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(
        self, obs_dim, act_dim, hidden_dim, hidden_depth, log_std_bounds, num_task, device=None
    ):
        super().__init__()

        self.num_task = num_task
        self.log_std_bounds = log_std_bounds
        self.device = device

        # Optimize layer creation with device parameter
        self.trunk = mlp(
            obs_dim, hidden_dim, 2 * act_dim * self.num_task, hidden_depth, layer_norm=False
        )
        if device is not None:
            self.to(device)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, task_id):
        all_mu_std = self.trunk(obs).chunk(self.num_task, dim=-1)
        mu, log_std = all_mu_std[task_id].chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        self.outputs["mu"] = mu
        self.outputs["std"] = std

        dist = SquashedNormal(mu, std)
        return dist

    def get_actions(self, obs, task_id):
        dist = self.forward(obs, task_id)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)  # type: ignore
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim, hidden_depth, num_task, device=None):
        super(Critic, self).__init__()
        self.num_task = num_task
        self.device = device

        self.critic = mlp(obs_dim, hidden_dim, 1 * num_task, hidden_depth, layer_norm=False)
        if device is not None:
            self.to(device)
        self.apply(weight_init)

    def forward(self, obs, task_id):
        value = self.critic(obs).chunk(self.num_task, dim=-1)[task_id]
        return value


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, obs_dim, act_dim, hidden_dim, hidden_depth, num_task, device=None):
        super().__init__()
        self.num_task = num_task
        self.device = device

        self.Q1 = mlp(obs_dim + act_dim, hidden_dim, 1 * num_task, hidden_depth, layer_norm=True)
        self.Q2 = mlp(obs_dim + act_dim, hidden_dim, 1 * num_task, hidden_depth, layer_norm=True)

        if device is not None:
            self.to(device)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, task_id):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action).chunk(self.num_task, dim=-1)[task_id]
        q2 = self.Q2(obs_action).chunk(self.num_task, dim=-1)[task_id]

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f"train_critic/{k}_hist", v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f"train_critic/q1_fc{i}", m1, step)
                logger.log_param(f"train_critic/q2_fc{i}", m2, step)


class QCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, obs_dim, act_dim, hidden_dim, hidden_depth, num_task, device=None):
        super().__init__()
        self.num_task = num_task
        self.device = device

        self.qf = mlp(obs_dim + act_dim, hidden_dim, 1 * num_task, hidden_depth, layer_norm=True)

        if device is not None:
            self.to(device)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, task_id):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.qf(obs_action).chunk(self.num_task, dim=-1)[task_id]

        self.outputs["q1"] = q1

        return q1

    def get_all_values(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        all_q_values = self.qf(obs_action).chunk(self.num_task, dim=-1)

        return torch.stack(all_q_values, dim=0)
