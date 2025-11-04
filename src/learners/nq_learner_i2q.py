import copy
import time
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop, Adam

from components.episode_buffer import EpisodeBatch
from components.epsilon_schedules import FlatThenDecayThenFlatSchedule
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
from utils.th_utils import get_parameters_num
from utils.rl_utils import RunningMeanStd


class QSS(th.nn.Module):
    def __init__(self, obs_len):
        super(QSS, self).__init__()
        self.fc1 = th.nn.Linear(2*obs_len, 256)
        self.fc2 = th.nn.Linear(256, 256)
        self.fc3 = th.nn.Linear(256, 1)

    def forward(self, x, y):
        q = self.fc3(F.relu(self.fc2(F.relu(self.fc1(th.cat((x,y),-1))))))
        return q

class VS(th.nn.Module):
    def __init__(self, obs_len):
        super(VS, self).__init__()
        self.fc1 = th.nn.Linear(obs_len, 256)
        self.fc2 = th.nn.Linear(256, 256)
        self.fc3 = th.nn.Linear(256, 1)

    def forward(self, x):
        q = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
        return q


class NQLearnerI2Q:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = list(mac.agent.parameters())
        
        self.qss = th.nn.ModuleList([QSS(self.args.rnn_hidden_dim) for _ in range(self.args.n_agents)]).cuda()
        self.vs = th.nn.ModuleList([VS(self.args.rnn_hidden_dim) for _ in range(self.args.n_agents)]).cuda()
        self.mixer = None
        
        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
            self.optimiser_qss = Adam(params=self.qss.parameters(), lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
            self.optimiser_vs = Adam(params=self.vs.parameters(), lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        elif self.args.optimizer == 'rmsprop':
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.optimiser_qss = RMSprop(params=self.qss.parameters(), lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.optimiser_vs = RMSprop(params=self.vs.parameters(), lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        else:
            raise "optimizer error"
        
        # 目标网络
        self.target_mac = copy.deepcopy(mac)
        self.target_qss = copy.deepcopy(self.qss)
        self.target_vs = copy.deepcopy(self.vs)
        
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0
        self.avg_time = 0
        
        # 奖励标准化
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.args.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=self.device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        start_time = time.time()
        if self.args.use_cuda and str(self.mac.get_device()) == "cpu":
            self.mac.cuda()

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        self.mac.set_train_mode()
        mac_out = []
        curiosity_entropy = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, curiosity_value = self.mac.forward(batch, t)
            mac_out.append(agent_outs)
            curiosity_entropy.append(curiosity_value.mean())
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # double DQN action
        mac_out[avail_actions == 0] = -9999999
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            # Set target mac to testing mode
            self.target_mac.set_evaluation_mode()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs, _ = self.target_mac.forward(batch, t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time
            target_mac_out[avail_actions == 0] = -9999999
            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.detach()
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, self.args.gamma, self.args.td_lambda)
            targets = targets.detach()

        mask = mask.expand_as(targets).detach()
        mask_sum = mask.sum().detach()
        
        mac_hid = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            mac_hid.append(self.target_mac.forward_hidden(batch, t=t))
        mac_hid = th.stack(mac_hid, dim=1).clone().detach()
        hid = mac_hid[:, :-1]
        next_hid = mac_hid[:,1:]

        v_s = th.stack([self.vs[i](hid[:,:,i]) for i in range(self.args.n_agents)], dim=2).squeeze(-1)
        if self.args.common_reward:
            target_v = th.stack([self.target_vs[i](next_hid[:,:,i]) * self.args.gamma * (1 - terminated) + rewards for i in range(self.args.n_agents)], dim=2).squeeze(-1).detach()
        else:
            target_v = th.stack([self.target_vs[i](next_hid[:,:,i]) * self.args.gamma * (1 - terminated) + rewards[..., i:i + 1] for i in range(self.args.n_agents)], dim=2).squeeze(-1).detach()
        q_ss = th.stack([self.qss[i](hid[:,:,i],next_hid[:,:,i]) for i in range(self.args.n_agents)], dim=2).squeeze(-1)
        loss_qss = (((q_ss - target_v)**2)*mask).sum()/mask_sum
        self.optimiser_qss.zero_grad()
        loss_qss.backward()
        th.nn.utils.clip_grad_norm_(self.qss.parameters(), self.args.grad_norm_clip)
        self.optimiser_qss.step()

        q_ss = th.stack([self.target_qss[i](hid[:,:,i],next_hid[:,:,i]) for i in range(self.args.n_agents)], dim=2).squeeze(-1).detach()
        weights = ((q_ss > v_s).float()*(2*self.args.tau - 1) + 1 - self.args.tau).clone().detach()
        w_mask = weights*mask
        loss_vs = (((v_s - q_ss)**2)*w_mask).sum()/w_mask.sum()
        self.optimiser_vs.zero_grad()
        loss_vs.backward()
        th.nn.utils.clip_grad_norm_(self.vs.parameters(), self.args.grad_norm_clip)
        self.optimiser_vs.step()

        predicted_targets = (self.args.lamb * target_v + (1 - self.args.lamb) * targets).detach()
        td_error = chosen_action_qvals - predicted_targets
        td_error2 = td_error.pow(2)
        masked_td_error = td_error2 * mask
        mask_elems = mask.sum()
        loss = masked_td_error.sum() / mask_elems
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        
        if self.mixer is not None:
            self.mixer.eval()
        self.mac.set_evaluation_mode()  # RL训练完毕

        self.train_t += 1
        self.avg_time += (time.time() - start_time - self.avg_time) / self.train_t
        print("Avg cost {} seconds".format(self.avg_time))
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
        
        curiosity_entropy = sum(curiosity_entropy) / len(curiosity_entropy)
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # For log
            with th.no_grad():
                mask_elems = mask_elems.item()
                td_error_abs = masked_td_error.abs().sum().item() / mask_elems
                q_taken_mean = (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents)
                target_mean = (targets * mask).sum().item() / (mask_elems * self.args.n_agents)
            self.logger.log_stat("train/loss_td", loss.item(), t_env)
            self.logger.log_stat("train/grad_norm", grad_norm, t_env)
            self.logger.log_stat("train/td_error_abs", td_error_abs, t_env)
            self.logger.log_stat("train/q_taken_mean", q_taken_mean, t_env)
            self.logger.log_stat("train/target_mean", target_mean, t_env)
            self.logger.log_stat("train/cuda_memory", th.cuda.max_memory_allocated() / 1024**2, t_env)
            self.logger.log_stat("train/curiosity_entropy", curiosity_entropy, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_qss.load_state_dict(self.qss.state_dict())
        self.target_vs.load_state_dict(self.vs.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        # 将优化器状态移动到 CUDA
        for state in self.optimiser.state.values():
            for k, v in state.items():
                if isinstance(v, th.Tensor):
                    state[k] = v.cuda() 
    
    def log_memory(self, step_name: str):
        if self.args.use_cuda:
            print(f"\n=== {step_name} ===")
            print(f"分配显存: {th.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"显存缓存: {th.cuda.memory_reserved() / 1024**2:.2f} MB")
            print(f"当前显存峰值: {th.cuda.max_memory_allocated() / 1024**2:.2f} MB\n")
