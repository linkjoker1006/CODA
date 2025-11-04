import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm


class NRNNAgent(nn.Module):
    """
    n_rnn 30.412K for 5m_vs_6m
    """
    def __init__(self, input_shape, args):
        super(NRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.migrate_fc1 = nn.Linear(args.rnn_hidden_dim * 2 + args.n_actions, args.rnn_hidden_dim)
        self.migrate_fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def give_advice(self, student_obs, student_hidden_state, teacher_obs, teacher_hidden_state, agent_outs):
        k, d = student_obs.shape
        x = student_obs.reshape(-1, d)
        x = F.relu(self.fc1(x))
        h_x = student_hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h_x = self.rnn(x, h_x)
        
        y = teacher_obs.reshape(-1, d)
        y = F.relu(self.fc1(y))
        h_y = teacher_hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h_y = self.rnn(y, h_y)
        # 因为不更新，所以不如detach了
        h = th.cat([h_x.detach(), h_y.detach(), agent_outs.detach()], dim=-1)
        h = F.relu(self.migrate_fc1(h))
        advice_action_logits = self.migrate_fc2(h)
        
        return advice_action_logits
    
    def forward(self, inputs, hidden_state):
        b, n, d = inputs.shape
        inputs = inputs.reshape(-1, d)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        q = q.reshape(b, n, -1)
        hh = hh.reshape(b, n, -1)
        
        return q, hh