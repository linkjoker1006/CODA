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
class NMACCONS(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(NMACCONS, self).__init__(scheme, groups, args)
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        # 发送请求和回应请求的预算
        self.ask_budget, self.give_budget = [], []
        # 记录每个智能体见过的观测值及其次数
        self.agent_obs = []
        for i in range(self.n_agents):
            self.ask_budget.append(args.budget)
            self.give_budget.append(args.budget)
            self.agent_obs.append({})
        self.negative_weight = 1
        self.positive_weight = 0
        self.try_adv_times = 0
        self
        
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
        # CONS
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
                    advised_action = self.ask_advice(qval, self.hidden_states[bs, agent_num, :], obs, last_action, agent_num, self.agent_obs, self.epi_obs, t_env)
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
    
    def _targeted_exploration(self, trust, sort, n_actions):
        interval = 1 / (n_actions - 1)
        for i in range(1, n_actions):
            if trust <= interval * i and trust > interval * (i - 1):
                avail_act_num = n_actions - i  # the number of candidate actions
                avail_act_prob = sort[0][-avail_act_num:]  # original probabilities of candidate actions
                avail_act_index = sort[1][-avail_act_num:]  # indexes of candidate actions
                new_act_prob = avail_act_prob / avail_act_prob.sum(dim=-1)
                action = random.choices(avail_act_index.tolist(), weights=new_act_prob, k=1)[0]
                break
        return action
    
    def ask_advice(self, out, hidden_state, obs, last_action, i, agent_obs, epi_obs, total_steps):
        out = F.softmax(out, dim=-1)
        batch_num = obs.shape[0]
        # 1. 计算mask
        need_advice_mask = []
        for b in range(batch_num):
            cond = random.random() < math.pow((1 + self.args.variable_a), -math.sqrt(agent_obs[i].get(tuple(obs[b].tolist()), 0))) and obs[b].tolist() not in epi_obs[b][i]
            need_advice_mask.append(cond)
        # 如果所有智能体都不需要建议，则返回-1
        if need_advice_mask.count(True) == 0:
            return th.full(size=(batch_num,), fill_value=-1, dtype=th.long)
        
        if "pgm" in self.args.env or "ft" in self.args.env or "cleanup" in self.args.env or "pp" in self.args.env:
            max_steps = self.args.env_args["max_steps"]
        elif self.args.env == "gymma":
            max_steps = self.args.env_args["time_limit"]
        elif "sc2" in self.args.env:
            max_steps = self.args.env_args["episode_limit"]
        omega = ((1 / self.args.c_w) - 1) / (1 - (self.args.start_advice / max_steps / self.args.c_ep))
        tmp = self.args.c_ep
        self.negative_weight = tmp / (tmp + omega * (total_steps - self.args.start_advice) / max_steps)
        self.positive_weight = 1 - self.negative_weight
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
        n_actions = out_adv.shape[-1]
        std = th.std(out_adv, dim=-1, unbiased=False)
        # Normalize std using Min-Max normalization to obtain the policy confidence
        trust = std * (n_actions / math.sqrt(n_actions - 1))
        a_best_worst = []  # Store the best & worst actions
        give_adv_list = []
        for b in range(batch_num):
            a_best_worst.append([])
            give_adv_list.append([])
            for j in range(self.n_agents):
                a_best_worst[b].append([-1, -1])
                give_adv_list[b].append(j)
                if i == j:
                    continue
                else:
                    a_best_worst[b][j][0] = out_adv[b, j].argmax().item()
                    a_best_worst[b][j][1] = out_adv[b, j].argmin().item()

        # Identify which agents are eligible for knowledge sharing
        advised_action = []
        for b in range(batch_num):
            if not need_advice_mask[b]:
                advised_action.append(-1)
                continue
            obs_key = tuple(obs[b].tolist())
            for j in range(self.n_agents):
                if (j == i) or (obs_key not in agent_obs[j]):
                    give_adv_list[b].remove(j)
                    continue
                if (agent_obs[j][obs_key] <= agent_obs[i][obs_key]) and (out_adv[b, j].max() <= out_adv[b, i].max()):
                    give_adv_list[b].remove(j)
                    continue
            if len(give_adv_list[b]) > 0:  # There are no agents eligible for knowledge sharing
                get_advice = False
                for k in range(n_actions):  # for each action
                    good_record, bad_record = [], []
                    good_visit, bad_visit = 0, 0
                    for l in give_adv_list[b]:
                        value = math.sqrt(agent_obs[l][obs_key]) * trust[b, l]
                        if (a_best_worst[b][l][0] == k) and (out_adv[b, l, k].item() > out[b, k].item()):  # 智能体l的最佳动作是k,且概率比i的大
                            temp = math.pow(math.e, value)
                            good_visit += temp
                            good_record.append(temp)
                            good_record.append((out_adv[b, l, k].item() - out[b, k].item()) * self.args.tau * self.positive_weight)
                        if (a_best_worst[b][l][1] == k) and (out_adv[b, l, k].item() < out[b, k].item()):  # 智能体l的最差动作是k,且概率比i的小
                            temp = math.pow(math.e, value)
                            bad_visit += temp
                            bad_record.append(temp)
                            bad_record.append((out_adv[b, l, k].item() - out[b, k].item()) * self.args.tau * self.negative_weight)
                    if good_visit != 0 or bad_visit != 0:  # 其他智能体的建议中包含了当前动作k，如果不包含就是0
                        get_advice = True
                        for m in range(int(len(good_record) / 2)):
                            out[b, k] += (good_record[2 * m] / good_visit * good_record[2 * m + 1])
                        for p in range(int(len(bad_record) / 2)):
                            out[b, k] += (bad_record[2 * p] / bad_visit * bad_record[2 * p + 1])
                if get_advice:  # Reference others' knowledge for decision-making
                    out[b] = F.softmax(out[b] / 0.3, dim=-1)  # obtains the new policy by performing softmax normalization
                    # cautiously absorbing the knowledge and sample an action
                    sort = th.sort(out[b])
                    std = th.std(out[b], unbiased=False)
                    adv_trust = std * (n_actions / math.sqrt(n_actions - 1))
                    action = out[b].argmax() if random.random() < adv_trust else self._targeted_exploration(adv_trust, sort, n_actions)
                    advised_action.append(action)
                    self.ask_budget[i] -= 1
                else:  # There is no need to reference others' knowledge for decision-making.
                    advised_action.append(-1)
            else:  # There are no agents eligible for knowledge sharing
                advised_action.append(-1)

        batch_advice = th.tensor(advised_action, dtype=th.long)

        return batch_advice
