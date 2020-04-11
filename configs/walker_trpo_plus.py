import os
import json

import sys
from utils import dict_product, iwt

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Walker2d-v2"],
    "mode": ["trpo"],
    "out_dir": ["results/trpo_plus_walker/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [False],
    "value_clipping": [False],
    "max_kl": [0.07] * 40,
    "max_kl_final": [0.04],
    "val_lr": [1e-4],
    "cpu": [True],
    "clip_grad_norm": [1.0],
    "save_iters": [150],
}

all_configs = [{**BASE_CONFIG, **p} for p in dict_product(PARAMS)]
if os.path.isdir("agent_configs/") or os.path.isdir("agents/"):
    raise ValueError("Please delete the 'agent_configs/' and 'agents/' directories")
os.makedirs("agent_configs/")
os.makedirs("agents/")

for i, config in enumerate(all_configs):
    with open(f"agent_configs/{i}.json", "w") as f:
        json.dump(config, f)
