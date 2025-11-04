from .nq_learner import NQLearner
from .nq_learner_i2q import NQLearnerI2Q
from .nq_learner_coda import NQLearnerCoda

REGISTRY = {}

REGISTRY["nq_learner"] = NQLearner
REGISTRY["nq_learner_i2q"] = NQLearnerI2Q
REGISTRY["nq_learner_coda"] = NQLearnerCoda