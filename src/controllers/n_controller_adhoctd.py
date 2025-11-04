import os

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
from components.epsilon_schedules import DecayThenFlatSchedule
import numpy as np
import random
import math
import torch.nn.functional as F
from collections import Counter

# This multi-agent controller shares parameters between agents
class NMACAdHocTD(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(NMACAdHocTD, self).__init__(scheme, groups, args)
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        # 发送请求和回应请求的预算
        self.ask_budget, self.give_budget = [], []
        # 记录每个智能体见过的观测值及其次数
        self.agent_obs = []
        self.epi_obs = []
        for i in range(self.n_agents):
            self.ask_budget.append(args.budget)
            self.give_budget.append(args.budget)
            self.agent_obs.append({})
            self.epi_obs.append([])
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # 每次rollout开始时，设置为评估模式
        if t_ep == 0:
            self.set_evaluation_mode()
            # 清空epi_obs
            self.epi_obs = []
            for b in range(ep_batch["obs"].shape[0]):
                self.epi_obs.append([])
                for i in range(self.n_agents):
                    self.epi_obs[b].append([])
        self.epsilon = self.schedule.eval(t_env)
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        for i in range(self.n_agents):
            for j in range(ep_batch["obs"].shape[0]):
                if tuple(ep_batch["obs"][j, t_ep, i].tolist()) in self.agent_obs[i]:
                    self.agent_obs[i][tuple(ep_batch["obs"][j, t_ep, i].tolist())] = self.agent_obs[i][tuple(ep_batch["obs"][j, t_ep, i].tolist())] + 1
                else:
                    self.agent_obs[i][tuple(ep_batch["obs"][j, t_ep, i].tolist())] = 1
        # AdHocTD
        qvals[avail_actions==0.0] = -float("inf")
        chosen_actions = th.zeros(size=(len(bs), self.n_agents), dtype=th.long)
        for agent_num in range(self.n_agents):
            qval = qvals[bs, agent_num, :]
            avail_action = avail_actions[bs, agent_num, :]
            if np.random.uniform() >= self.epsilon:
                chosen_actions[:, agent_num] = self.action_selector.select_action_no_epsilon(qval, avail_action, t_env, test_mode=test_mode)
            else:
                # 试图获取建议
                obs = ep_batch["obs"][bs, t_ep, agent_num]
                if t_ep == 0:
                    last_action = ep_batch["actions_onehot"][bs, t_ep, agent_num]
                else:
                    last_action = ep_batch["actions_onehot"][bs, t_ep - 1, agent_num]
                # 如果可以发送请求，则完全按照建议执行
                if t_env > self.args.start_advice and self.ask_budget[agent_num] > 0:  # able to send a request
                    advised_action = self.ask_advice(qval, self.hidden_states[bs, agent_num, :], obs, last_action, agent_num, self.agent_obs, self.epi_obs)
                    for i in range(advised_action.shape[0]):
                        if advised_action[i] == -1:  # knowledge is not available
                            if np.random.uniform() < self.epsilon:
                                advised_action[i] = self.action_selector.select_action_random(qval[i:i + 1], avail_action[i:i + 1], t_env, test_mode=test_mode)[0]
                            else:
                                advised_action[i] = self.action_selector.select_action_no_epsilon(qval[i:i + 1], avail_action[i:i + 1], t_env, test_mode=test_mode)[0]
                    chosen_actions[:, agent_num] = advised_action
                else:  # unable to send a request, perform epsilon-greedy
                    if np.random.uniform() < self.epsilon:
                        chosen_actions[:, agent_num] = self.action_selector.select_action_random(qval, avail_action, t_env, test_mode=test_mode)
                    else:
                        chosen_actions[:, agent_num] = self.action_selector.select_action_no_epsilon(qval, avail_action, t_env, test_mode=test_mode)
        # 更新epi_obs
        for b in range(ep_batch["obs"].shape[0]):
            for i in range(self.n_agents):
                self.epi_obs[b][i].append(ep_batch["obs"][b, t_ep, i].tolist())

        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        curiosity = self.cal_curiosity(agent_outs)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), curiosity
    
    def select_from_advice(self, advised_list):
        if len(advised_list) == 0:
            return None
        elif len(advised_list) == 1:
            return advised_list[0]
        else:
            count = Counter(advised_list).most_common()
            action_index = []
            most_times = count[0][1]
            for tuple in count:
                if tuple[1] < most_times:
                    break
                if tuple[1] == most_times:
                    action_index.append(tuple[0])
            action = random.choices(action_index, k=1)[0]
        return action
    
    def ask_advice(self, out, hidden_state, obs, last_action, i, agent_obs, epi_obs):
        batch_num = obs.shape[0]
        # 1. 计算mask
        need_advice_mask = []
        for b in range(batch_num):
            cond = random.random() < math.pow((1 + self.args.variable_a), -math.sqrt(agent_obs[i].get(tuple(obs[b].tolist()), 0))) and obs[b].tolist() not in epi_obs[b][i]
            need_advice_mask.append(cond)
        # 如果所有智能体都不需要建议，则返回-1
        if need_advice_mask.count(True) == 0:
            return th.full(size=(batch_num,), fill_value=-1, dtype=th.long)
        # 2. 批量构造输入
        s = []
        for b in range(batch_num):
            s_b = []
            for j in range(self.n_agents):
                s_j = np.asarray(obs[b]).flatten()
                if self.args.obs_last_action:
                    s_j = np.hstack((s_j, np.asarray(last_action[b]).flatten()))
                if self.args.shared_agent and self.args.obs_agent_id:
                    s_j = np.hstack((s_j, np.eye(self.n_agents)[j]))
                s_b.append(s_j)
            s.append(s_b)
        obs_adv = th.Tensor(np.array(s))  # [batch, n_agents, obs_dim]
        # 3. 批量神经网络推理
        if self.args.shared_agent:
            out_adv, _ = self.agent(obs_adv, hidden_state.unsqueeze(-2).expand(batch_num, self.n_agents, -1))
        else:
            # 这里如果不是shared_agent，还是要循环
            out_adv = []
            for advisor_i in range(self.n_agents):
                out_adv_i, _ = self.agent.agents[advisor_i](obs_adv[:, advisor_i:advisor_i + 1, :], hidden_state.unsqueeze(-2))
                out_adv.append(out_adv_i)
            out_adv = th.stack(out_adv, dim=1).squeeze(2)  # [batch, n_agents, n_actions]
        out_adv = F.softmax(out_adv, dim=-1)
        # 4. 生成建议动作
        action_list = []
        for b in range(batch_num):
            if not need_advice_mask[b]:
                action_list.append(-1)
                continue
            advised_actions = []
            for j in range(self.n_agents):
                if i == j:
                    continue
                difQ = math.fabs(out_adv[b, j, :].max() - out_adv[b, j, :].min())
                numberVisits = agent_obs[j].get(tuple(obs[b].tolist()), 0)
                value = (math.sqrt(numberVisits) * difQ)
                prob = 1 - (math.pow((1 + self.args.variable_g), -value))
                if random.random() < prob:
                    advised_actions.append(out_adv[b, j, :].argmax().item())
            action = self.select_from_advice(advised_actions)
            if action is None:
                # -1表示没有建议
                action_list.append(-1)
            else:
                self.ask_budget[i] -= 1
                action_list.append(action)
        
        batch_advice = th.tensor(action_list, dtype=th.long)

        return batch_advice
