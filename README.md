# Code for "Implementation Matters in Deep RL: A Case Study on PPO and TRPO"

This repository contains our implementation of PPO and TRPO, with manual toggles
for the code-level optimizations described in our paper. We assume that the user
has a machine with MuJoCo and mujoco_py properly set up and installed, i.e.
you should be able to run the following command on your system without errors:

```python
import gym
gym.make_env("Humanoid-v2")
```

The code itself is quite simple to use. To run the ablation case study discussed
in our paper, you can run the following list of commands:

1. ``cd configs/``
2. ``mkdir PATH_TO_OUT_DIR`` and change ``out_dir`` to this in the relevant config file. By default agents will be written to ``results/{env}_{algorithm}/agents/``. 
3. ``python {config_name}.py``
4. ``cd ..``
5. Edit the ``NUM_THREADS`` variables in the ``run_agents.py`` file according to your local machine.
6. Train the agents: ``python run_agents.py PATH_TO_OUT_DIR/agent_configs``
7. The outputs will be in the ``agents`` subdirectory of ``OUT_DIR``, readable
with the ``cox`` python library.

See the ``MuJoCo.json`` file for a full list of adjustable parameters.

