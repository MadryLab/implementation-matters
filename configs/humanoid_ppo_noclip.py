import os
import json

import sys
sys.path.append("../")
from utils import dict_product, iwt

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Humanoid-v2"],
    "mode": ["ppo"],
    "clip_eps": [1e32],
    "out_dir": ["results/ppo_noclip_humanoid/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["xavier"],
    "anneal_lr": [False],
    "value_clipping": [True],
    "entropy_coeff": [0.005],
    "ppo_lr_adam": [2e-5] * 40,
    "clip_grad_norm": [0.5],
    "val_lr": [5e-5],
    "lambda": [0.85],
    "cpu": [True],
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
