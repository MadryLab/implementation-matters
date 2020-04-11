import os
import json

import sys
#sys.path.append("../")
from utils import dict_product, iwt

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Walker2d-v2"],
    "mode": ["ppo"],
    "clip_eps": [1e32],
    "out_dir": ["results/ppo_noclip_walker/agents"],
    "norm_rewards": ["rewards"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "value_clipping": [False],
    "ppo_lr_adam": [0.725e-4] * 40,
    "entropy_coeff": [-0.01],
    "clip_rewards": [30],
    "clip_observations": [-1],
    "lambda": [0.85],
    "val_lr": [6e-4],
    "cpu": [True],
    "clip_grad_norm": [0.1],
    "advanced_logging": [True]
}

all_configs = [{**BASE_CONFIG, **p} for p in dict_product(PARAMS)]
if os.path.isdir("agent_configs/") or os.path.isdir("agents/"):
    raise ValueError("Please delete the 'agent_configs/' and 'agents/' directories")
os.makedirs("agent_configs/")
os.makedirs("agents/")

for i, config in enumerate(all_configs):
    with open(f"agent_configs/{i}.json", "w") as f:
        json.dump(config, f)
