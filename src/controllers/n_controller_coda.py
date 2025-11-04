import os

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
from components.epsilon_schedules import FlatThenDecayThenFlatSchedule
import numpy as np
import random
import math
import copy
import json
import torch.nn.functional as F
from collections import Counter
from modules.others.ask import HybridRequestNetwork

# This multi-agent controller shares parameters between agents
class NMACCODA(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(NMACCODA, self).__init__(scheme, groups, args)
        self.ask_budget = [0] * self.n_agents
        # 从start_advice开始慢慢加入建议机制
        self.ask_schedule = FlatThenDecayThenFlatSchedule(0.0, 1.0, max(args.start_advice, 0), args.all_ask, decay="linear")
        self.ask_ratio = self.ask_schedule.eval(0)
        self.advice_schedule_1 = FlatThenDecayThenFlatSchedule(1.0, 0.05, max(args.start_advice, 0), args.all_advice, decay="linear")
        self.advice_ratio = self.advice_schedule_1.eval(0)
        self.advice_schedule_2 = FlatThenDecayThenFlatSchedule(1.0, 0.05, max(args.start_advice, 0), max(args.start_advice, 0) + (args.epsilon_anneal_time * 2), decay="linear")
        self.advice_epsilon = self.advice_schedule_2.eval(0)
        # 构建每个智能体的curiosity模型
        self.build_ask_networks()
        self.replay_file = None
    
    def cal_curiosity(self, agent_outs):
        with th.no_grad():
            cur_logits = agent_outs
            cur_probs = F.softmax(cur_logits, dim=-1)
            cur_log_probs = F.log_softmax(cur_logits, dim=-1)
            # 计算策略熵新奇度
            novelty_entropy = -th.sum(cur_probs * cur_log_probs, dim=-1, keepdim=True)
        
        return novelty_entropy
    
    def build_ask_networks(self):
        if "pgm" in self.args.env or "ft" in self.args.env or "cleanup" in self.args.env or "pp" in self.args.env:  # 3ag/6ag
            self.self_dim = self.args.obs_component[0]
            self.teacher_num = self.args.obs_component[1][0]
            self.teacher_dim = self.args.obs_component[1][1]
        elif "gymma" in self.args.env:  # spread/tag
            self.self_dim = self.args.obs_component[0] + self.args.obs_component[1][0] * self.args.obs_component[1][1]
            self.teacher_num = self.args.obs_component[2][0]
            self.teacher_dim = self.args.obs_component[2][1]
        self_dim = self.self_dim
        if self.args.obs_last_action:
            self_dim += self.args.n_actions
        if self.args.obs_agent_id:
            self_dim += self.n_agents
        if self.args.shared_ask:
            self.ask_networks = HybridRequestNetwork(self_dim, self.teacher_dim, self.teacher_num)
        else:
            self.ask_networks = []
            for i in range(self.n_agents):
                self.ask_networks.append(HybridRequestNetwork(self_dim, self.teacher_dim, self.teacher_num))
    
    def get_ask_probs(self, self_inputs, teacher_inputs, curiosity):
        ask_probs = []
        for i in range(self.n_agents):
            if self.args.shared_ask:
                gate_logit, _ = self.ask_networks(self_inputs[:, i:i + 1], teacher_inputs[:, i:i + 1], curiosity[:, i:i + 1])
            else:
                gate_logit, _ = self.ask_networks[i](self_inputs[:, i:i + 1], teacher_inputs[:, i:i + 1], curiosity[:, i:i + 1])
            ask_probs.append(gate_logit)
        ask_probs = th.cat(ask_probs, dim=1)
        
        return ask_probs
    
    def get_advice_probs(self, all_obs, agent_outs):
        b, n, _ = all_obs.shape
        advice_probs = []
        for teacher_id in range(n):
            advice_logits = []
            for student_id in range(n):
                if self.args.shared_agent:
                    advice_logit = self.agent.give_advice(all_obs[:, student_id], 
                                                          self.hidden_states[:, student_id], 
                                                          all_obs[:, teacher_id], 
                                                          self.hidden_states[:, teacher_id],
                                                          agent_outs[:, student_id])
                else:
                    advice_logit = self.agent.agents[teacher_id].give_advice(all_obs[:, student_id], 
                                                                             self.hidden_states[:, student_id], 
                                                                             all_obs[:, teacher_id], 
                                                                             self.hidden_states[:, teacher_id],
                                                                             agent_outs[:, student_id])
                advice_logits.append(advice_logit.unsqueeze(1))
            advice_logits = th.cat(advice_logits, dim=1)
            advice_probs.append(advice_logits.unsqueeze(1))
        advice_probs = th.cat(advice_probs, dim=1)
        
        return advice_probs
    
    def anwser_ask(self, ask_actions, all_obs, advice_probs, test_mode=False):
        if test_mode:
            self.advice_epsilon = 0.0
        b, n, _ = all_obs.shape
        # 创建一个最终的输出张量，用-1进行填充
        advices = th.full_like(ask_actions, -1)
        # TODO: 以一定概率输出随机建议
        # random_advices = th.randint(0, advice_probs.shape[3], size=(advice_probs.shape[0], advice_probs.shape[1], advice_probs.shape[2]), device=advice_probs.device).long()
        # random_numbers = th.rand(size=(advice_probs.shape[0], advice_probs.shape[1], advice_probs.shape[2]), device=advice_probs.device)
        # pick_random = (random_numbers < self.advice_epsilon).long()
        # all_advice_actions = (1 - pick_random) * th.argmax(advice_probs, dim=-1) + pick_random * random_advices
        all_advice_actions = th.argmax(advice_probs, dim=-1)
        # 按老师ID进行分组处理，以实现高效的批量推理
        for teacher_id in range(n):
            # 1. 找到所有向这位老师请求的学生
            request_mask_for_teacher = (ask_actions == teacher_id)  # (B, N)，值为True的位置代表一个有效请求
            # 如果没人向这位老师请求，就跳过
            if not request_mask_for_teacher.any(): 
                continue
            # 2. 收集所有请求这位老师的“学生”的观测数据
            student_indices = request_mask_for_teacher.nonzero(as_tuple=False)  # (K, 2) 的张量, K是请求数, 每行是[batch_idx, student_idx]
            # 分离出批次索引和学生ID索引
            batch_indices = student_indices[:, 0]  # (K,)
            student_ids = student_indices[:, 1]    # (K,)
            # 3. 收集所有请求这位老师的“学生”的建议
            advice_actions = all_advice_actions[batch_indices, teacher_id, student_ids]  # (K,)
            # 4. 将生成的建议“散播”回最终的输出张量中
            advices[batch_indices, student_ids] = advice_actions
        
        return advices
    
    def make_decision(self, chosen_probs, ask_actions, chosen_actions, all_obs, advice_probs, test_mode=False):
        if test_mode:
            self.advice_ratio = 0.0
        advice_actions = self.anwser_ask(ask_actions, all_obs, advice_probs, test_mode)
        advice_mask = (advice_actions != -1)
        new_advice_probs = th.zeros_like(chosen_probs)
        for i in range(ask_actions.shape[0]):
            for j in range(ask_actions.shape[1]):
                if ask_actions[i, j] >= 0:
                    new_advice_probs[i, j] = advice_probs[i, ask_actions[i, j], j]
        chosen_action_qvals = th.gather(chosen_probs, dim=2, index=chosen_actions.unsqueeze(-1)).squeeze(-1)
        advice_action_qvals = th.gather(new_advice_probs, dim=2, index=(advice_actions * advice_mask).unsqueeze(-1)).squeeze(-1)
        # 以一定概率直接采纳建议
        random_numbers = th.rand(size=(ask_actions.shape[0], ask_actions.shape[1]), device=advice_mask.device)
        pick_random = random_numbers < self.advice_ratio
        gate_mask = advice_mask * (pick_random | (advice_action_qvals > chosen_action_qvals))
        final_actions = advice_actions * gate_mask + chosen_actions * (~gate_mask)
        
        return final_actions, advice_actions
    
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if t_ep == 0:
            self.set_evaluation_mode()
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals, ask_probs, advice_probs, curiosity = self.forward(ep_batch, t_ep, t_env)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        # 在训练start_advice步数之前，所有agent都不请求
        if t_env <= self.args.start_advice:
            ask_actions = th.full((len(bs), self.n_agents), -1, device=qvals.device).long()
            advice_actions = th.full((len(bs), self.n_agents), -1, device=qvals.device).long()
            final_actions = chosen_actions
        else:
            ask_actions = self.action_selector.select_ask_action(ask_probs[bs], self.ask_ratio, t_env, test_mode)
            final_actions, advice_actions = self.make_decision(qvals[bs], ask_actions, chosen_actions, self._build_inputs(ep_batch, t_ep), advice_probs, test_mode)
        # 统计每个智能体的询问次数
        cnt_ask = th.sum((advice_actions != -1), dim=0)
        for i in range(self.n_agents):
            self.ask_budget[i] += cnt_ask[i]
        
        return final_actions, ask_actions, advice_actions, chosen_actions

    def forward(self, ep_batch, t, t_env):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        curiosity = self.cal_curiosity(agent_outs)
        # 在训练start_advice步数之前，所有agent都不请求
        if t_env <= self.args.start_advice:
            ask_probs = None
            advice_probs = None
        else:
            self.ask_ratio = self.ask_schedule.eval(t_env)
            self.advice_ratio = self.advice_schedule_1.eval(t_env)
            self.advice_epsilon = self.advice_schedule_2.eval(t_env)
            self_inputs, teacher_inputs = self._build_attention_inputs(ep_batch, t)
            ask_probs = self.get_ask_probs(self_inputs, teacher_inputs, curiosity)
            advice_probs = self.get_advice_probs(agent_inputs, agent_outs)
        
        return agent_outs, ask_probs, advice_probs, curiosity
    
    def _build_attention_inputs(self, batch, t):
        # 把obs中附近老师的信息分离出来
        bs = batch.batch_size
            
        self_inputs = []
        self_inputs.append(batch["obs"][:, t, :, :self.self_dim])  # b1av
        teacher_inputs = []
        for i in range(self.teacher_num):
            start = self.self_dim + i * self.teacher_dim
            end = start + self.teacher_dim
            teacher_inputs.append(batch["obs"][:, t, :, start:end])
        if self.args.obs_last_action:
            if t == 0:
                self_inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                self_inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            self_inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        self_inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in self_inputs], dim=-1)
        teacher_inputs = th.cat([x.reshape(bs, self.n_agents, 1, -1) for x in teacher_inputs], dim=2)
        
        return self_inputs, teacher_inputs
    
    def cuda(self):
        self.agent.cuda()
        if self.args.shared_ask:
            self.ask_networks.cuda()
        else:
            for network in self.ask_networks:
                network.cuda()

    def cpu(self):
        self.agent.cpu()
        if self.args.shared_ask:
            self.ask_networks.cpu()
        else:
            for network in self.ask_networks:
                network.cpu()
    
    def set_train_mode(self):
        self.agent.train()
        if self.args.shared_ask:
            self.ask_networks.train()
        else:
            for network in self.ask_networks:
                network.train()
    
    def set_evaluation_mode(self):
        self.agent.eval()
        if self.args.shared_ask:
            self.ask_networks.eval()
        else:
            for network in self.ask_networks:
                network.eval()
    
    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        if self.args.shared_ask:
            th.save(self.ask_networks.state_dict(), "{}/ask_networks.th".format(path))
        else:
            for i, network in enumerate(self.ask_networks):
                th.save(network.state_dict(), "{}/ask_networks_{}.th".format(path, i))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        if self.args.shared_ask:
            self.ask_networks.load_state_dict(th.load("{}/ask_networks.th".format(path), map_location=lambda storage, loc: storage))
        else:
            for i, network in enumerate(self.ask_networks):
                network.load_state_dict(th.load("{}/ask_networks_{}.th".format(path, i), map_location=lambda storage, loc: storage))
    
    def game_log(self, ep_batch, t, agent_outs, ask_probs, advice_probs, curiosity, final_actions, ask_actions, chosen_actions, advice_actions):
        if self.replay_file is None:
            self.replay_file = open("replay.log", "w")
        if t == 24:
            self.replay_file.close()
        # 1. 从批次中获取当前时间步的观测值
        pos = ep_batch["obs"][:, t, :, :2]
        # 反归一化，横坐标乘7，纵坐标乘8
        pos = pos * th.tensor([7, 8], device=pos.device)
        # 2. 将所有需要记录的 Pytorch Tensors 移到 CPU 并转为 NumPy 数组
        # 这样做可以避免 GPU 内存占用，并且方便后续处理
        log_data = {
            "pos": pos.cpu().numpy(),
            "agent_outs": agent_outs.cpu().numpy(),
            "ask_probs": ask_probs.cpu().numpy(),
            "advice_probs": advice_probs.cpu().numpy(),
            "curiosity": curiosity.cpu().numpy(),
            "final_actions": final_actions.cpu().numpy(),
            "ask_actions": ask_actions.cpu().numpy(),
            "chosen_actions": chosen_actions.cpu().numpy(),
            "advice_actions": advice_actions.cpu().numpy()
        }
        # 3. 遍历批次中的每一个样本 (agent)
        batch_size = pos.shape[0]
        for i in range(batch_size):
            # 为当前 agent 构建一个 JSON 对象
            record = {
                "timestep": t,
                "agent_index": i, # 记录这是批次中的第几个 agent
                # 使用 .tolist() 将 numpy 数组转为 python 列表，以便 json 序列化
                "pos": log_data["pos"][i].tolist(),
                "curiosity": log_data["curiosity"][i].tolist(),
                "final_action": log_data["final_actions"][i].tolist(),
                "ask_action": log_data["ask_actions"][i].tolist(),
                "chosen_action": log_data["chosen_actions"][i].tolist(),
                "advice_action": log_data["advice_actions"][i].tolist()
            }

            # 4. 将该条记录写入文件，并添加换行符
            self.replay_file.write(json.dumps(record) + '\n')
        