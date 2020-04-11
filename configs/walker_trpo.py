import os
import json

import sys
from utils import dict_product, iwt

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Walker2d-v2"],
    "mode": ["trpo"],
    "out_dir": ["results/trpo_walker/agents"],
    "norm_rewards": ["none"],
    "initialization": ["xavier"],
    "anneal_lr": [False],
    "value_clipping": [False],
    "max_kl": [0.04] * 40,
    "max_kl_final": [0.04],
    "val_lr": [3e-4],
    "clip_rewards": [-1],
    "clip_observations" : [-1],
    "cpu": [True],
}

all_configs = [{**BASE_CONFIG, **p} for p in dict_product(PARAMS)]
if os.path.isdir("agent_configs/") or os.path.isdir("agents/"):
    raise ValueError("Please delete the 'agent_configs/' and 'agents/' directories")
os.makedirs("agent_configs/")
os.makedirs("agents/")

for i, config in enumerate(all_configs):
    with open(f"agent_configs/{i}.json", "w") as f:
        json.dump(config, f)
