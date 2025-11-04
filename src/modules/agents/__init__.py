from .n_rnn_agent import NRNNAgent
from .n_mlp_agent import NMLPAgent
from .n_rnn_ns_agent import NRNNNSAgent


REGISTRY = {}

REGISTRY["n_mlp"] = NMLPAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["n_rnn_ns"] = NRNNNSAgent