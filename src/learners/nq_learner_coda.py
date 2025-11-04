import copy
import time
import numpy as np
import torch as th
from torch.optim import RMSprop, Adam

from components.episode_buffer import EpisodeBatch
from components.epsilon_schedules import FlatThenDecayThenFlatSchedule
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
from utils.th_utils import get_parameters_num
from utils.rl_utils import RunningMeanStd


class NQLearnerCoda:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = list(mac.agent.parameters())
        self.ask_params = []
        if self.args.shared_ask:
            self.ask_params.extend(list(mac.ask_networks.parameters()))
        else:
            for i in mac.ask_networks:
                self.ask_params.extend(list(i.parameters()))
        self.advice_params = []
        if self.args.shared_agent:
            self.advice_params.extend(list(mac.agent.migrate_fc1.parameters()))
            self.advice_params.extend(list(mac.agent.migrate_fc2.parameters()))
        else:
            for i in mac.agent.agents:
                self.advice_params.extend(list(i.migrate_fc1.parameters())) 
                self.advice_params.extend(list(i.migrate_fc2.parameters()))
        # 混合网络
        self.mixer = None
        # 目标网络
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0
        self.avg_time = 0
        
        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
            self.optimiser_ask = Adam(params=self.ask_params, lr=args.ask_lr, weight_decay=getattr(args, "weight_decay", 0))
            self.optimiser_advice = Adam(params=self.advice_params, lr=args.advice_lr, weight_decay=getattr(args, "weight_decay", 0))
        elif self.args.optimizer == 'rmsprop':
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.optimiser_ask = RMSprop(params=self.ask_params, lr=args.ask_lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.optimiser_advice = RMSprop(params=self.advice_params, lr=args.advice_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        else:
            raise "optimizer error"
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
        
        if t_env > self.args.start_advice:
            ask_actions = batch["ask_actions"][:, :-1]
            # 复原未处理的ask_actions
            original_ask_actions = ask_actions.clone()
            requester_ids = th.arange(self.mac.n_agents, device=original_ask_actions.device).repeat(batch.batch_size, batch.max_seq_length - 1, 1).unsqueeze(-1)
            # 相对老师索引 > 请求者自己的ID
            need_add_mask = (original_ask_actions > requester_ids)
            original_ask_actions[need_add_mask] -= 1
            original_ask_actions += 1
            advice_actions = batch["advice_actions"][:, :-1]
            chosen_actions = batch["chosen_actions"][:, :-1] 
            advice_mask = th.ne(advice_actions, chosen_actions) * (advice_actions != -1)
            # advice_mask = (advice_actions != -1)

        # Calculate estimated Q-Values
        self.mac.set_train_mode()
        mac_out = []
        curiosity_entropy = []
        ask_out = []
        advice_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, ask_probs, advice_probs, curiosity_value = self.mac.forward(batch, t, t_env)
            mac_out.append(agent_outs)
            curiosity_entropy.append(curiosity_value.mean())
            if t_env > self.args.start_advice:
                ask_out.append(ask_probs)
                advice_out.append(advice_probs)
            
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        mac_out[avail_actions == 0] = -9999999
        if t_env > self.args.start_advice:
            ask_out = th.stack(ask_out, dim=1)
            raw_advice_out = th.stack(advice_out, dim=1)  # (B, T, N, N, N_actions), 注意对于每个agent都有给其他agent的建议动作分布
            # 根据ask_actions和raw_advice_out计算每个agent的建议动作分布
            advice_out = th.zeros_like(mac_out)
            for teacher_id in range(self.args.n_agents):
                request_mask_for_teacher = (ask_actions == teacher_id)
                # 如果没人向这位老师请求，就跳过
                if not request_mask_for_teacher.any(): 
                    continue
                student_indices = request_mask_for_teacher.nonzero(as_tuple=False)  # (K, 3)
                advice_out[student_indices[:, 0], student_indices[:, 1], student_indices[:, 2], :] = raw_advice_out[student_indices[:, 0], student_indices[:, 1], teacher_id, student_indices[:, 2], :]
            # 此时advice_actions包含-1，需要掩蔽后再做索引操作
            advice_actions = advice_actions * advice_mask
            advice_mask = advice_mask.squeeze(3)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        if t_env > self.args.start_advice:
            chosen_ask_qvals = th.gather(ask_out[:, :-1], dim=3, index=original_ask_actions).squeeze(3)  # Remove the last dim
            chosen_advice_qvals = th.gather(advice_out[:, :-1], dim=3, index=advice_actions).squeeze(3)  # Remove the last dim
            chosen_origin_qvals = th.gather(mac_out[:, :-1], dim=3, index=chosen_actions).squeeze(3)  # Remove the last dim
        
        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            # Set target mac to testing mode
            self.target_mac.set_evaluation_mode()
            target_mac_out = []
            target_ask_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs, target_ask_probs, _, _ = self.target_mac.forward(batch, t, t_env)
                target_mac_out.append(target_agent_outs)
                if t_env > self.args.start_advice:
                    target_ask_out.append(target_ask_probs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time
            target_mac_out[avail_actions == 0] = -9999999
            if t_env > self.args.start_advice:
                target_ask_out = th.stack(target_ask_out, dim=1)
            
            # Max over target Q-Values/ Double q learning
            cur_max_actions = mac_out.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            if t_env > self.args.start_advice:
                cur_max_ask_actions = ask_out.max(dim=3, keepdim=True)[1]
                target_max_ask_qvals = th.gather(target_ask_out, 3, cur_max_ask_actions).squeeze(3)

            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, self.args.gamma, self.args.td_lambda).detach()
            if t_env > self.args.start_advice:
                rewards_ask = ((chosen_advice_qvals - chosen_origin_qvals) * advice_mask).detach()
                targets_ask = build_td_lambda_targets(rewards_ask, terminated, mask, target_max_ask_qvals, self.args.gamma, self.args.td_lambda).detach()
                targets_advice = targets

        td_error = targets - chosen_action_qvals
        td_error2 = 0.5 * td_error.pow(2)
        mask_rl = mask.expand_as(td_error2)
        masked_td_error_rl = td_error2 * mask_rl
        mask_elems_rl = mask_rl.sum()
        loss_rl = masked_td_error_rl.sum() / mask_elems_rl
        # Optimise
        self.optimiser.zero_grad()
        loss_rl.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        
        if t_env > self.args.start_advice:
            td_error_ask = targets_ask - chosen_ask_qvals
            td_error_advice = targets_advice - chosen_advice_qvals
            td_error_ask2 = 0.5 * td_error_ask.pow(2)
            td_error_advice2 = 0.5 * td_error_advice.pow(2)
            mask_ask = mask.expand_as(td_error_ask2)
            mask_advice = mask.expand_as(td_error_advice2) * advice_mask
            masked_td_error_ask = td_error_ask2 * mask_ask
            masked_td_error_advice = td_error_advice2 * mask_advice
            mask_elems_ask = mask_ask.sum()
            mask_elems_advice = mask_advice.sum()
            # 询问机制的损失
            loss_ask = masked_td_error_ask.sum() / mask_elems_ask
            # Optimise
            self.optimiser_ask.zero_grad()
            loss_ask.backward()
            grad_norm_ask = th.nn.utils.clip_grad_norm_(self.ask_params, self.args.grad_norm_clip)
            self.optimiser_ask.step()
            # 建议机制的损失
            if mask_elems_advice == 0:
                loss_advice = th.tensor(0.0).to(self.device)
            else:
                loss_advice = masked_td_error_advice.sum() / mask_elems_advice
                # Optimise
                self.optimiser_advice.zero_grad()
                loss_advice.backward()
                grad_norm_advice = th.nn.utils.clip_grad_norm_(self.advice_params, self.args.grad_norm_clip)
                self.optimiser_advice.step()
        
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
                mask_elems_rl = mask_elems_rl.item()
                td_error_abs = masked_td_error_rl.abs().sum().item() / mask_elems_rl
                q_taken_mean = (chosen_action_qvals * mask_rl).sum().item() / (mask_elems_rl * self.args.n_agents)
                target_mean = (targets * mask_rl).sum().item() / (mask_elems_rl * self.args.n_agents)
                if t_env > self.args.start_advice:
                    td_error_abs_ask = masked_td_error_ask.abs().sum().item() / mask_elems_ask
                    td_error_abs_advice = masked_td_error_advice.abs().sum().item() / mask_elems_advice
                    reward_ask = th.mean(rewards_ask)
            self.logger.log_stat("train/loss_rl", loss_rl.item(), t_env)
            self.logger.log_stat("train/grad_norm", grad_norm, t_env)
            self.logger.log_stat("train/td_error_abs", td_error_abs, t_env)
            self.logger.log_stat("train/q_taken_mean", q_taken_mean, t_env)
            self.logger.log_stat("train/target_mean", target_mean, t_env)
            if t_env > self.args.start_advice:
                self.logger.log_stat("train/loss_ask", loss_ask.item(), t_env)
                self.logger.log_stat("train/td_error_abs_ask", td_error_abs_ask, t_env)
                self.logger.log_stat("train/loss_advice", loss_advice.item(), t_env)
                self.logger.log_stat("train/td_error_abs_advice", td_error_abs_advice, t_env)
                self.logger.log_stat("train/reward_ask", reward_ask, t_env)
            self.logger.log_stat("train/cuda_memory", th.cuda.max_memory_allocated() / 1024**2, t_env)
            self.logger.log_stat("train/curiosity_entropy", curiosity_entropy, t_env)
            for i in range(self.mac.n_agents):
                self.logger.log_stat("train/budget_agent_{}".format(i), self.mac.ask_budget[i], t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.logger.console_logger.info("Updated target network")
    
    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_mac.parameters(), self.mac.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

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
