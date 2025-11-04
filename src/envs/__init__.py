from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

try:
    smac = True
    from .smac_v1.StarCraft2EnvWrapper import StarCraft2EnvWrapper
except Exception as e:
    print(e)
    smac = False

try:
    smacv2 = True
    from .smac_v2.StarCraft2Env2Wrapper import StarCraft2Env2Wrapper
except Exception as e:
    print(e)
    smacv2 = False

try:
    mpe = True
    from .mpe.GymmaEnvWrapper import GymmaEnvWrapper
except Exception as e:
    print(e)
    mpe = False

try:
    ft = True
    from .ft.FindTreasureWrapper import FindTreasureWrapper
except Exception as e:
    print(e)
    ft = False

try:
    pgm = True
    from .pgm.PatientGoldMinerWrapper import PatientGoldMinerWrapper
except Exception as e:
    print(e)
    pgm = False
    
try:
    cleanup = True
    from .cleanup.CleanUpWrapper import CleanUpWrapper
except Exception as e:
    print(e)
    cleanup = False

try:
    pp = True
    from .pp.PredatorPreyWrapper import PredatorPreyWrapper
except Exception as e:
    print(e)
    pp = False

def __check_and_prepare_smac_kwargs(kwargs):
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    assert kwargs[
        "common_reward"
    ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
    del kwargs["common_reward"]
    del kwargs["reward_scalarisation"]
    assert "map_name" in kwargs, "Please specify the map_name in the env_args"
    return kwargs

def smac_fn(env, **kwargs) -> MultiAgentEnv:
    kwargs = __check_and_prepare_smac_kwargs(kwargs)
    return env(**kwargs)

def gymma_fn(env, **kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return env(**kwargs)

def ft_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

def pgm_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

def cleanup_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

def pp_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}

if smac:
    REGISTRY["sc2"] = partial(smac_fn, env=StarCraft2EnvWrapper)
    if sys.platform == "linux":
        os.environ.setdefault("SC2PATH",
                              os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
else:
    print("SMAC V1 is not supported...")

if smacv2:
    REGISTRY["sc2_v2"] = partial(smac_fn, env=StarCraft2Env2Wrapper)
    if sys.platform == "linux":
        os.environ.setdefault("SC2PATH",
                              os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
else:
    print("SMAC V2 is not supported...")

if mpe:
    REGISTRY["gymma"] = partial(gymma_fn, env=GymmaEnvWrapper)
else:
    print("MPE is not supported...")
    
if ft:
    REGISTRY["ft"] = partial(ft_fn, env=FindTreasureWrapper)
else:
    print("FT is not supported...")

if pgm:
    REGISTRY["pgm3ag"] = partial(pgm_fn, env=PatientGoldMinerWrapper)
    REGISTRY["pgm6ag"] = partial(pgm_fn, env=PatientGoldMinerWrapper)
else:
    print("PGM is not supported...")

if cleanup:
    REGISTRY["cleanup"] = partial(cleanup_fn, env=CleanUpWrapper)
else:
    print("Cleanup is not supported...")
    
if pp:
    REGISTRY["pp"] = partial(pp_fn, env=PredatorPreyWrapper)
else:
    print("PP is not supported...")

print("Supported environments:", REGISTRY)
