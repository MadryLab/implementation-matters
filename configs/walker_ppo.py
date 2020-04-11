import os
import json

import sys
from utils import dict_product, iwt

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Walker2d-v2"],
    "mode": ["ppo"],
    "out_dir": ["results/ppo_walker/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "value_clipping": [True],
    "ppo_lr_adam": [4e-4] * 40,
    "val_lr": [3e-4],
    "cpu": [True],
    "save_iters": [150],
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
