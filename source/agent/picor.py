import os

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict, from_module
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.utils.clip_grad import clip_grad_norm_

from tools import soft_update_params, to_np
from agent.models import Critic, DiagGaussianActor, DoubleQCritic

class PiCor:
    def __init__(self, cfg, algo_cfg, env_cfg, device):
        super().__init__()

        self.cfg = cfg
        self.algo_cfg = algo_cfg
        self.env_cfg = env_cfg
        self.device = device

        self.num_task = self.cfg.num_task
        self.max_ep_len = self.cfg.max_ep_len
        self.target_entropy = -self.env_cfg["act_dim"]
        self.discount = self.algo_cfg["sac"]["discount"]
        self.autotune = self.algo_cfg["sac"]["autotune"]
        self.sac_epoch = self.algo_cfg["sac"]["sac_epoch"]
        self.batch_size = self.algo_cfg["sac"]["batch_size"]
        self.critic_tau = self.algo_cfg["sac"]["critic_tau"]
        self.init_t = self.algo_cfg["sac"]["init_temperature"]
        self.max_grad_norm = self.algo_cfg["sac"]["max_grad_norm"]
        self.actor_update_frequency = self.algo_cfg["sac"]["actor_update_frequency"]
        self.critic_target_update_frequency = self.algo_cfg["sac"]["critic_target_update_frequency"]
        self.action_range = torch.tensor(self.env_cfg["act_range"]).to(self.device)

        self.lamda = self.algo_cfg["corr"]["lamda"]
        self.epsilon = self.algo_cfg["corr"]["epsilon"]
        self.corr_epochs = self.algo_cfg["corr"]["corr_epochs"]

        # instantiate critic and actor
        self.algo_cfg["critic_cfg"]["obs_dim"] = self.env_cfg["obs_dim"]
        self.algo_cfg["critic_cfg"]["act_dim"] = self.env_cfg["act_dim"]
        self.algo_cfg["actor_cfg"]["obs_dim"] = self.env_cfg["obs_dim"]
        self.algo_cfg["actor_cfg"]["act_dim"] = self.env_cfg["act_dim"]
        self.algo_cfg["corr_v_cfg"]["obs_dim"] = self.env_cfg["obs_dim"]

        self.setup()
        self.set_training_mode()

    def setup(self):
        self.capturable = self.cfg.cudagraphs and not self.cfg.compile

        self.amp_enabled = self.cfg.amp and torch.cuda.is_available()
        self.amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.amp_dtype = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16
        self.scaler = GradScaler(enabled=self.amp_enabled and self.amp_dtype == torch.float16)

        # setup actor
        self.actor = DiagGaussianActor(
            num_task=self.num_task, **self.algo_cfg["actor_cfg"], device=self.device
        )
        self.actor_detach = DiagGaussianActor(
            num_task=self.num_task, **self.algo_cfg["actor_cfg"], device=self.device
        )
        from_module(self.actor).data.to_module(self.actor_detach)
        self.policy = self.actor_detach.forward

        # setup critic
        self.qnet = DoubleQCritic(
            num_task=self.num_task, device=self.device, **self.algo_cfg["critic_cfg"]
        )
        self.qnet_target = DoubleQCritic(
            num_task=self.num_task, device=self.device, **self.algo_cfg["critic_cfg"]
        )
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        # setup correction value
        self.corr_v = Critic(
            num_task=self.num_task, device=self.device, **self.algo_cfg["corr_v_cfg"]
        )

        # setup alpha
        self.log_alpha = torch.tensor(
            np.log(np.repeat(self.init_t, self.num_task)),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )

        # setup optimizer
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=self.algo_cfg["sac"]["actor_lr"], capturable=self.capturable
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.qnet.parameters(), lr=self.algo_cfg["sac"]["critic_lr"], capturable=self.capturable
        )
        self.log_alpha_optimizer = torch.optim.AdamW(
            [self.log_alpha], lr=self.algo_cfg["sac"]["alpha_lr"], capturable=self.capturable
        )
        self.corr_v_optimizer = torch.optim.AdamW(
            self.corr_v.parameters(),
            lr=self.algo_cfg["sac"]["critic_lr"],
            capturable=self.capturable,
        )

        # setup compile
        if self.cfg.compile:
            compile_mode = self.cfg.compile_mode

            # for training functions
            self._update_qf = torch.compile(self.update_critic, mode=compile_mode)
            self._update_pi = torch.compile(self.update_actor_and_alpha, mode=compile_mode)
            self._policy = torch.compile(self.policy, mode=None)

            # for nn
            self._actor = torch.compile(self.actor, mode=compile_mode)
            self._qnet = torch.compile(self.qnet, mode=compile_mode)
            self._qnet_target = torch.compile(self.qnet_target, mode=compile_mode)
            self._corr_v = torch.compile(self.corr_v, mode=compile_mode)

            # for other functions
            self._gae = torch.compile(self.gae, mode=compile_mode)
            self._projection = torch.compile(self.projection, mode=compile_mode)
            self._optimizer_update = torch.compile(self.optimizer_update, mode=compile_mode)
        else:
            self._update_qf = self.update_critic
            self._update_pi = self.update_actor_and_alpha
            self._policy = self.act
            self._actor = self.actor
            self._qnet = self.qnet
            self._qnet_target = self.qnet_target
            self._corr_v = self.corr_v
            self._gae = self.gae
            self._projection = self.projection
            self._optimizer_update = self.optimizer_update

    def act(self, obs, task_i, sample=False):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        obs = obs.unsqueeze(0)
        dist = self.policy(obs, task_i)
        action = dist.sample() if sample else dist.mean
        action = tensor_clamp(action, self.action_range[0], self.action_range[1])

        assert action.ndim == 2 and action.shape[0] == 1
        return to_np(action[0])

    def update(self, data, step):
        logs_dict = TensorDict()

        for _ in range(self.sac_epoch):
            logs_dict = self._update_qf(data, logs_dict)

            if step % self.algo_cfg["sac"]["actor_update_frequency"] == 0:
                logs_dict = self._update_pi(data, logs_dict)

            if step % self.algo_cfg["sac"]["critic_target_update_frequency"] == 0:
                soft_update_params(self.qnet, self.qnet_target, self.critic_tau)

            if self.amp_enabled:
                self.scaler.update()

        logs_dict["losses/alpha"] = self.alpha
        return logs_dict

    def update_critic(self, data, logs_dict):
        task_i = data["task_i"][0]
        with autocast(
            device_type=self.amp_device_type, dtype=self.amp_dtype, enabled=self.amp_enabled
        ):
            with torch.no_grad():
                next_actions, log_prob = self._actor.get_actions(data["next_observations"], task_i)
                qf_next_target1, qf_next_target2 = self._qnet_target(
                    data["next_observations"], next_actions, task_i
                )
                target_v = (
                    torch.min(qf_next_target1, qf_next_target2)
                    - self.alpha[task_i].detach() * log_prob
                )
                next_q_values = data["rewards"].unsqueeze(-1) + (
                    ~data["dones"].unsqueeze(-1) * self.discount * target_v
                )

            curr_q_value1, curr_q_value2 = self._qnet(data["observations"], data["actions"], task_i)
            assert (
                next_q_values.shape == curr_q_value1.shape == curr_q_value2.shape
            ), "Shape mismatch {} {} {}".format(
                next_q_values.shape, curr_q_value1.shape, curr_q_value2.shape
            )
            qf_loss = F.mse_loss(curr_q_value1, next_q_values) + F.mse_loss(
                curr_q_value2, next_q_values
            )

        self._optimizer_update(self.critic_optimizer, qf_loss)

        logs_dict["losses/qf_loss"] = qf_loss.detach()
        return logs_dict

    def update_actor_and_alpha(self, data, logs_dict):
        task_i = data["task_i"][0]
        with autocast(
            device_type=self.amp_device_type, dtype=self.amp_dtype, enabled=self.amp_enabled
        ):
            # actor update
            actions, log_prob = self._actor.get_actions(data["observations"], task_i)
            actor_q_value1, actor_q_value2 = self._qnet(data["observations"], actions, task_i)
            actor_q_value = torch.min(actor_q_value1, actor_q_value2)
            actor_loss = (self.alpha[task_i] * log_prob - actor_q_value).mean()

            # alpha update
            if self.autotune:
                alpha_loss = (
                    self.alpha[task_i] * (-log_prob - self.target_entropy).detach()
                ).mean()
                alpha_loss_value = alpha_loss.detach()
            else:
                alpha_loss_value = torch.tensor(0.0, device=self.device)

        grad_norm = self._optimizer_update(self.actor_optimizer, actor_loss)
        if self.autotune and alpha_loss.requires_grad:
            self._optimizer_update(self.log_alpha_optimizer, alpha_loss)

        logs_dict["losses/pi_loss"] = actor_loss.detach()
        logs_dict["losses/entropy"] = -log_prob.detach()
        logs_dict["losses/grad_norm"] = grad_norm
        logs_dict["losses/alpha_loss"] = alpha_loss_value
        return logs_dict

    def projection(self, buffers):
        weights = F.softmax(self.alpha.detach(), dim=0)
        vars = []
        for task_i in range(self.num_task):
            data = buffers[task_i][-self.max_ep_len :]
            advantages, returns = self._gae(data, task_i)
            dist = self._actor(data["observations"], task_i)
            logprob = dist.log_prob(data["actions"]).sum(-1, keepdim=True)
            vars.append((data["observations"], data["actions"], logprob, advantages, returns))

        for _ in range(self.corr_epochs):
            all_loss = []
            for task_i in range(self.num_task):
                obs, action, logprob, adv, returns = vars[task_i]

                # Actor loss
                dist_now = self._actor(obs, task_i)
                logprob_now = dist_now.log_prob(action).sum(-1, keepdim=True)
                ratios = torch.exp(logprob_now - logprob)
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                all_loss.append(actor_loss)

            project_loss = (
                torch.tensor(all_loss, device=self.device, requires_grad=True) * weights
            ).sum()
            self._optimizer_update(self.actor_optimizer, project_loss)

            corr_v_losses = []
            for task_i in range(self.num_task):
                obs, _, _, _, returns = vars[task_i]
                values = self._corr_v(obs, task_i)
                if values.dim() > 1:
                    values = values.squeeze(-1)
                corr_v_loss = F.mse_loss(returns.detach(), values)
                corr_v_losses.append(corr_v_loss)

            total_corr_v_loss = torch.stack(corr_v_losses).mean()
            self._optimizer_update(self.corr_v_optimizer, total_corr_v_loss)

    def gae(self, data, task_i):
        # bootstrap value if not done
        next_obs = data["next_observations"][-1]
        next_value = self._corr_v(next_obs, task_i).reshape(-1)

        values = self._corr_v(data["observations"], task_i).reshape(-1)  # [0, ..., max_ep_len-1]
        vals_unbind = torch.cat([values, next_value]).unbind(0)  # [0, ..., max_ep_len]
        nextnonterminals = (~data["dones"]).float().unbind(0)  # [0, ..., max_ep_len-1]
        rewards = data["rewards"].unbind(0)  # [0, ..., max_ep_len-1]

        lastgaelam = 0
        advantages = []
        for t in range(self.max_ep_len - 1, -1, -1):
            delta = (
                rewards[t]
                + self.discount * vals_unbind[t + 1] * nextnonterminals[t]
                - vals_unbind[t]
            )
            advantages.append(delta + self.discount * self.lamda * nextnonterminals[t] * lastgaelam)
            lastgaelam = advantages[-1]

        advantages = torch.stack(list(reversed(advantages)))
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values
        return advantages, returns

    def optimizer_update(self, optimizer, objective):
        if self.amp_enabled:
            optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(objective).backward()
            if self.max_grad_norm is not None:
                self.scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(
                    parameters=optimizer.param_groups[0]["params"], max_norm=self.max_grad_norm
                )
            else:
                grad_norm = None
            self.scaler.step(optimizer)
        else:
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            if self.max_grad_norm is not None:
                grad_norm = clip_grad_norm_(
                    parameters=optimizer.param_groups[0]["params"], max_norm=self.max_grad_norm
                )
            else:
                grad_norm = None
            optimizer.step()
        return grad_norm

    def set_training_mode(self, training=True):
        self.training = training
        self.actor.train(training)
        self.qnet.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save(self, path):
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pt"))
        torch.save(self.qnet.state_dict(), os.path.join(path, "critic.pt"))
        torch.save(self.log_alpha, os.path.join(path, "log_alpha.pt"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(path, "actor_optimizer.pt"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(path, "critic_optimizer.pt"))
        torch.save(
            self.log_alpha_optimizer.state_dict(), os.path.join(path, "log_alpha_optimizer.pt")
        )

    def load(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pt")))
        self.qnet.load_state_dict(torch.load(os.path.join(path, "critic.pt")))
        self.log_alpha = torch.load(os.path.join(path, "log_alpha.pt"))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(path, "actor_optimizer.pt")))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(path, "critic_optimizer.pt")))
        self.log_alpha_optimizer.load_state_dict(
            torch.load(os.path.join(path, "log_alpha_optimizer.pt"))
        )


# helper functions
@torch.jit.script
def tensor_clamp(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(x, upper), lower)


@torch.jit.script
def unscale_action(action: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return action * scale + bias
