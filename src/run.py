from policy_gradients.agent import Trainer
import git
import numpy as np
import os
import argparse
from policy_gradients import models
import sys
import json
import torch
from cox.store import Store, schema_from_dict


# Tee object allows for logging to both stdout and to file
class Tee(object):
    def __init__(self, file_path, stream_type, mode='a'):
        assert stream_type in ['stdout', 'stderr']

        self.file = open(file_path, mode)
        self.stream_type = stream_type
        self.errors = 'chill'

        if stream_type == 'stdout':
            self.stream = sys.stdout
            sys.stdout = self
        else:
            self.stream = sys.stderr
            sys.stderr = self

    def write(self, data):
        self.file.write(data)
        self.stream.write(data)

    def flush(self):
        self.file.flush()
        self.stream.flush()

def main(params):
    for k, v in zip(params.keys(), params.values()):
        assert v is not None, f"Value for {k} is None"

    # #
    # Setup logging
    # #
    metadata_schema = schema_from_dict(params)
    base_directory = params['out_dir']
    store = Store(base_directory)

    # redirect stderr, stdout to file
    """
    def make_err_redirector(stream_name):
        tee = Tee(os.path.join(store.path, stream_name + '.txt'), stream_name)
        return tee

    stderr_tee = make_err_redirector('stderr')
    stdout_tee = make_err_redirector('stdout')
    """

    # Store the experiment path and the git commit for this experiment
    metadata_schema.update({
        'store_path':str,
        'git_commit':str
    })

    repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)),
                    search_parent_directories=True)

    metadata_table = store.add_table('metadata', metadata_schema)
    metadata_table.update_row(params)
    metadata_table.update_row({
        'store_path':store.path,
        'git_commit':repo.head.object.hexsha
    })

    metadata_table.flush_row()

    # Table for checkpointing models and envs
    if params['save_iters'] > 0:
        store.add_table('checkpoints', {
            'val_model':store.PYTORCH_STATE,
            'policy_model':store.PYTORCH_STATE,
            'envs':store.PICKLE,
            'policy_opt': store.PYTORCH_STATE,
            'val_opt': store.PYTORCH_STATE,
            'iteration':int
        })

    # The trainer object is in charge of sampling trajectories and
    # taking PPO/TRPO optimization steps
    p = Trainer.agent_from_params(params, store=store)
    rewards = []

    # Table for final results
    final_table = store.add_table('final_results', {
        'iteration':int,
        '5_rewards':float,
        'terminated_early':bool
    })

    def finalize_table(iteration, terminated_early, rewards):
        final_5_rewards = np.array(rewards)[-5:].mean()
        final_table.append_row({
            'iteration':iteration,
            '5_rewards':final_5_rewards,
            'terminated_early':terminated_early
        })

    # Try-except so that we save if the user interrupts the process
    try:
        for i in range(params['train_steps']):
            print('Step %d' % (i,))
            if params['save_iters'] > 0 and i % params['save_iters'] == 0:
                store['checkpoints'].append_row({
                    'iteration':i,
                    'val_model': p.val_model.state_dict(),
                    'policy_model': p.policy_model.state_dict(),
                    'policy_opt': p.POLICY_ADAM.state_dict(),
                    'val_opt': p.val_opt.state_dict(),
                    'envs':p.envs
                })
            
            mean_reward = p.train_step()
            rewards.append(mean_reward)

        finalize_table(i, False, rewards)
    except KeyboardInterrupt:
        torch.save(p.val_model, 'saved_experts/%s-expert-vf' % (params['game'],))
        torch.save(p.policy_model, 'saved_experts/%s-expert-pol' % (params['game'],))

        finalize_table(i, True, rewards)
    store.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate experiments to be run.')
    # Basic setup
    parser.add_argument('--config-path', type=str, required=True,
                        help='json for this config')
    parser.add_argument('--game', type=str, help='gym game')
    parser.add_argument('--mode', type=str, choices=['ppo', 'trpo'],
                        help='pg alg')
    parser.add_argument('--out-dir', type=str,
                        help='out dir for store + logging')
    parser.add_argument('--advanced-logging', type=bool, const=True, nargs='?')
    parser.add_argument('--kl-approximation-iters', type=int,
                        help='how often to do kl approx exps')
    parser.add_argument('--log-every', type=int)
    parser.add_argument('--policy-net-type', type=str,
                        choices=models.POLICY_NETS.keys())
    parser.add_argument('--value-net-type', type=str,
                        choices=models.VALUE_NETS.keys())
    parser.add_argument('--train-steps', type=int,
                        help='num agent training steps')
    parser.add_argument('--cpu', type=bool, const=True, nargs='?')

    # Which value loss to use
    parser.add_argument('--value-calc', type=str,
                        help='which value calculation to use')
    parser.add_argument('--initialization', type=str)

    # General Policy Gradient parameters
    parser.add_argument('--num-actors', type=int, help='num actors (serial)',
                        choices=[1])
    parser.add_argument('--t', type=int,
                        help='num timesteps to run each actor for')
    parser.add_argument('--gamma', type=float, help='discount on reward')
    parser.add_argument('--lambda', type=float, help='GAE hyperparameter')
    parser.add_argument('--val-lr', type=float, help='value fn learning rate')
    parser.add_argument('--val-epochs', type=int, help='value fn epochs')

    # PPO parameters
    parser.add_argument('--adam-eps', type=float, choices=[0, 1e-5], help='adam eps parameter')

    parser.add_argument('--num-minibatches',type=int,
                        help='num minibatches in ppo per epoch')
    parser.add_argument('--ppo-epochs', type=int)
    parser.add_argument('--ppo-lr', type=float,
                        help='if nonzero, use gradient descent w this lr')
    parser.add_argument('--ppo-lr-adam', type=float,
                        help='if nonzero, use adam with this lr')
    parser.add_argument('--anneal-lr', type=bool,
                        help='if we should anneal lr linearly from start to finish')
    parser.add_argument('--clip-eps', type=float, help='ppo clipping')
    parser.add_argument('--entropy-coeff', type=float,
                        help='entropy weight hyperparam')
    parser.add_argument('--value-clipping', type=bool,
                        help='should clip values (w/ ppo eps)')
    parser.add_argument('--value-multiplier', type=float,
                        help='coeff for value loss in combined step ppo loss')
    parser.add_argument('--share-weights', type=bool,
                        help='share weights in valnet and polnet')
    parser.add_argument('--clip-grad-norm', type=float,
                        help='gradient norm clipping (-1 for no clipping)')
    
    # TRPO parameters
    parser.add_argument('--max-kl', type=float, help='trpo max kl hparam')
    parser.add_argument('--max-kl-final', type=float, help='trpo max kl final')
    parser.add_argument('--fisher-frac-samples', type=float,
                        help='frac samples to use in fisher vp estimate')
    parser.add_argument('--cg-steps', type=int,
                        help='num cg steps in fisher vp estimate')
    parser.add_argument('--damping', type=float, help='damping to use in cg')
    parser.add_argument('--max-backtrack', type=int, help='max bt steps in fvp')

    # Normalization parameters
    parser.add_argument('--norm-rewards', type=str, help='type of rewards normalization', 
                        choices=['rewards', 'returns', 'none'])
    parser.add_argument('--norm-states', type=bool, help='should norm states')
    parser.add_argument('--clip-rewards', type=float, help='clip rews eps')
    parser.add_argument('--clip-observations', type=float, help='clips obs eps')

    # Saving
    parser.add_argument('--save-iters', type=int, help='how often to save model (0 = no saving)')

    # For grid searches only
    # parser.add_argument('--cox-experiment-path', type=str, default='')
    
    args = parser.parse_args()

    json_params = json.load(open(args.config_path))

    # Override the JSON config with the argparse config
    params = vars(args)
    missing_keys = []
    for key in json_params:
        if key not in params:
            missing_keys.append(key)
    assert not missing_keys, "Following keys not in args: " + str(missing_keys)

    missing_keys = []
    for key in params:
        if key not in json_params and key != "config_path":
            missing_keys.append(key)
    assert not missing_keys, "Following keys not in JSON: " + str(missing_keys)

    json_params.update({k: params[k] for k in params if params[k] is not None})
    params = json_params

    main(params)

