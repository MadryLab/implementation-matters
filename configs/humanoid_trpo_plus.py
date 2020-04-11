import os
import json

import sys
from utils import dict_product, iwt

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Humanoid-v2"],
    "mode": ["trpo"],
    "out_dir": ["results/trpo_plus_humanoid/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["xavier"],
    "anneal_lr": [False],
    "value_clipping": [True],
    "max_kl": [0.1] * 40,
    "max_kl_final": [0.05],
    "val_lr": [5e-5],
    "clip_grad_norm": [0.5],
    "lambda": [0.85],
    "cpu": [True]
}

all_configs = [{**BASE_CONFIG, **p} for p in dict_product(PARAMS)]
if os.path.isdir("agent_configs/") or os.path.isdir("agents/"):
    raise ValueError("Please delete the 'agent_configs/' and 'agents/' directories")
os.makedirs("agent_configs/")
os.makedirs("agents/")

for i, config in enumerate(all_configs):
    with open(f"agent_configs/{i}.json", "w") as f:
        json.dump(config, f)
