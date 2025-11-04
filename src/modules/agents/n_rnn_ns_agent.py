import torch.nn as nn
import torch as th

from .n_rnn_agent import NRNNAgent


class NRNNNSAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NRNNNSAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_shape = input_shape
        self.agents = th.nn.ModuleList(
            [NRNNAgent(input_shape, args) for _ in range(self.n_agents)]
        )

    def init_hidden(self):
        # make hidden states on same device as model
        return th.cat([a.init_hidden() for a in self.agents], dim=0)

    def forward(self, inputs, hidden_state):
        hiddens = []
        qs = []
        for i in range(self.n_agents):
            q, h = self.agents[i](inputs[:, i:i + 1], hidden_state[:, i:i + 1])
            hiddens.append(h)
            qs.append(q)
        return th.cat(qs, dim=1), th.cat(hiddens, dim=1)

    def cuda(self):
        for a in self.agents:
            a.cuda()
    
    def cpu(self):
        for a in self.agents:
            a.cpu()
