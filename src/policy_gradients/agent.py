import torch
import tqdm
import time
import dill
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
import gym
from .models import *
from .torch_utils import *
from .steps import value_step, step_with_mode
from .logging import *

from multiprocessing import Process, Queue
from .custom_env import Env

class Trainer():
    '''
    This is a class representing a Policy Gradient trainer, which 
    trains both a deep Policy network and a deep Value network.
    Exposes functions:
    - advantage_and_return
    - multi_actor_step
    - reset_envs
    - run_trajectories
    - train_step
    Trainer also handles all logging, which is done via the "cox"
    library
    '''
    def __init__(self, policy_net_class, value_net_class, params,
                 store, advanced_logging=True, log_every=5):
        '''
        Initializes a new Trainer class.
        Inputs;
        - policy, the class of policy network to use (inheriting from nn.Module)
        - val, the class of value network to use (inheriting from nn.Module)
        - step, a reference to a function to use for the policy step (see steps.py)
        - params, an dictionary with all of the required hyperparameters
        '''
        # Parameter Loading
        self.params = Parameters(params)

        # Whether or not the value network uses the current timestep
        time_in_state = self.VALUE_CALC == "time"

        # Whether to use GPU (as opposed to CPU)
        if not self.CPU:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

        # Environment Loading
        def env_constructor():
            # Whether or not we should add the time to the state
            horizon_to_feed = self.T if time_in_state else None
            return Env(self.GAME, norm_states=self.NORM_STATES,
                       norm_rewards=self.NORM_REWARDS,
                       params=self.params,
                       add_t_with_horizon=horizon_to_feed,
                       clip_obs=self.CLIP_OBSERVATIONS,
                       clip_rew=self.CLIP_REWARDS)

        self.envs = [env_constructor() for _ in range(self.NUM_ACTORS)]
        self.params.AGENT_TYPE = "discrete" if self.envs[0].is_discrete else "continuous"
        self.params.NUM_ACTIONS = self.envs[0].num_actions
        self.params.NUM_FEATURES = self.envs[0].num_features 
        self.policy_step = step_with_mode(self.MODE)
        self.params.MAX_KL_INCREMENT = (self.params.MAX_KL_FINAL - self.params.MAX_KL) / self.params.TRAIN_STEPS
        self.advanced_logging = advanced_logging
        self.n_steps = 0
        self.log_every = log_every

        # Instantiation
        self.policy_model = policy_net_class(self.NUM_FEATURES, self.NUM_ACTIONS,
                                             self.INITIALIZATION,
                                             time_in_state=time_in_state)

        opts_ok = (self.PPO_LR == -1 or self.PPO_LR_ADAM == -1)
        assert opts_ok, "One of ppo_lr and ppo_lr_adam must be -1 (off)."
        # Whether we should use Adam or simple GD to optimize the policy parameters
        if self.PPO_LR_ADAM != -1:
            kwargs = {
                'lr':self.PPO_LR_ADAM,
            }

            if self.params.ADAM_EPS > 0:
                kwargs['eps'] = self.ADAM_EPS

            self.params.POLICY_ADAM = optim.Adam(self.policy_model.parameters(),
                                                 **kwargs)
        else:
            self.params.POLICY_ADAM = optim.SGD(self.policy_model.parameters(), lr=self.PPO_LR)

        # If using a time dependent value function, add one extra feature
        # for the time ratio t/T
        if time_in_state:
            self.params.NUM_FEATURES = self.NUM_FEATURES + 1

        # Value function optimization
        self.val_model = value_net_class(self.NUM_FEATURES, self.INITIALIZATION)
        self.val_opt = optim.Adam(self.val_model.parameters(), lr=self.VAL_LR, eps=1e-5) 
        assert self.policy_model.discrete == (self.AGENT_TYPE == "discrete")

        # Learning rate annealing
        # From OpenAI hyperparametrs:
        # Set adam learning rate to 3e-4 * alpha, where alpha decays from 1 to 0 over training
        if self.ANNEAL_LR:
            lam = lambda f: 1-f/self.TRAIN_STEPS
            ps = optim.lr_scheduler.LambdaLR(self.POLICY_ADAM, 
                                                    lr_lambda=lam)
            vs = optim.lr_scheduler.LambdaLR(self.val_opt, lr_lambda=lam)
            self.params.POLICY_SCHEDULER = ps
            self.params.VALUE_SCHEDULER = vs

        if store is not None:
            self.setup_stores(store)

    def setup_stores(self, store):
        # Logging setup
        self.store = store
        self.store.add_table('optimization', {
            'mean_reward':float,
            'final_value_loss':float,
            'mean_std':float
        })

        if self.advanced_logging:
            paper_constraint_cols = {
                'avg_kl':float,
                'max_ratio':float,
                'opt_step':int
            }

            value_cols = {
                'heldout_gae_loss':float,
                'heldout_returns_loss':float,
                'train_gae_loss':float,
                'train_returns_loss':float
            }

            self.store.add_table('paper_constraints_train',
                                        paper_constraint_cols)
            self.store.add_table('paper_constraints_heldout',
                                        paper_constraint_cols)
            self.store.add_table('value_data', value_cols)


    def __getattr__(self, x):
        '''
        Allows accessing self.A instead of self.params.A
        '''
        if x == 'params':
            return {}
        try:
            return getattr(self.params, x)
        except KeyError:
            raise AttributeError(x)

    def advantage_and_return(self, rewards, values, not_dones):
        """
        Calculate GAE advantage, discounted returns, and 
        true reward (average reward per trajectory)

        GAE: delta_t^V = r_t + discount * V(s_{t+1}) - V(s_t)
        using formula from John Schulman's code:
        V(s_t+1) = {0 if s_t is terminal
                   {v_s_{t+1} if s_t not terminal and t != T (last step)
                   {v_s if s_t not terminal and t == T
        """
        assert shape_equal_cmp(rewards, values, not_dones)
        
        V_s_tp1 = ch.cat([values[:,1:], values[:, -1:]], 1) * not_dones
        deltas = rewards + self.GAMMA * V_s_tp1 - values

        # now we need to discount each path by gamma * lam
        advantages = ch.zeros_like(rewards)
        returns = ch.zeros_like(rewards)
        indices = get_path_indices(not_dones)
        for agent, start, end in indices:
            advantages[agent, start:end] = discount_path( \
                    deltas[agent, start:end], self.LAMBDA*self.GAMMA)
            returns[agent, start:end] = discount_path( \
                    rewards[agent, start:end], self.GAMMA)

        return advantages.clone().detach(), returns.clone().detach()

    def reset_envs(self, envs):
        '''
        Resets environments and returns initial state with shape:
        (# actors, 1, ... state_shape)
	    '''
        if self.CPU:
            return cpu_tensorize([env.reset() for env in envs]).unsqueeze(1)
        else:
            return cu_tensorize([env.reset() for env in envs]).unsqueeze(1)

    def multi_actor_step(self, actions, envs):
        '''
        Simulate a "step" by several actors on their respective environments
        Inputs:
        - actions, list of actions to take
        - envs, list of the environments in which to take the actions
        Returns:
        - completed_episode_info, a variable-length list of final rewards and episode lengths
            for the actors which have completed
        - rewards, a actors-length tensor with the rewards collected
        - states, a (actors, ... state_shape) tensor with resulting states
        - not_dones, an actors-length tensor with 0 if terminal, 1 otw
        '''
        normed_rewards, states, not_dones = [], [], []
        completed_episode_info = []
        for action, env in zip(actions, envs):
            gym_action = action[0].cpu().numpy()
            new_state, normed_reward, is_done, info = env.step(gym_action)
            if is_done:
                completed_episode_info.append(info['done'])
                new_state = env.reset()

            # Aggregate
            normed_rewards.append([normed_reward])
            not_dones.append([int(not is_done)])
            states.append([new_state])

        tensor_maker = cpu_tensorize if self.CPU else cu_tensorize
        data = list(map(tensor_maker, [normed_rewards, states, not_dones]))
        return [completed_episode_info, *data]

    def run_trajectories(self, num_saps, return_rewards=False, should_tqdm=False):
        """
        Resets environments, and runs self.T steps in each environment in 
        self.envs. If an environment hits a terminal state, the env is
        restarted and the terminal timestep marked. Each item in the tuple is
        a tensor in which the first coordinate represents the actor, and the
        second coordinate represents the time step. The third+ coordinates, if
        they exist, represent additional information for each time step.
        Inputs: None
        Returns:
        - rewards: (# actors, self.T)
        - not_dones: (# actors, self.T) 1 in timestep if terminal state else 0
        - actions: (# actors, self.T, ) indices of actions
        - action_logprobs: (# actors, self.T, ) log probabilities of each action
        - states: (# actors, self.T, ... state_shape) states
        """
        # Arrays to be updated with historic info
        envs = self.envs
        initial_states = self.reset_envs(envs)

        # Holds information (length and true reward) about completed episodes
        completed_episode_info = []
        traj_length = int(num_saps // self.NUM_ACTORS)

        shape = (self.NUM_ACTORS, traj_length)
        all_zeros = [ch.zeros(shape) for i in range(3)]
        rewards, not_dones, action_log_probs = all_zeros

        actions_shape = shape + (self.NUM_ACTIONS,)
        actions = ch.zeros(actions_shape)

        states_shape = (self.NUM_ACTORS, traj_length+1) + initial_states.shape[2:]
        states =  ch.zeros(states_shape)

        iterator = range(traj_length) if not should_tqdm else tqdm.trange(traj_length)

        assert self.NUM_ACTORS == 1

        states[:, 0, :] = initial_states
        last_states = states[:, 0, :]
        for t in iterator:
            # assert shape_equal([self.NUM_ACTORS, self.NUM_FEATURES], last_states)
            # Retrieve probabilities 
            # action_pds: (# actors, # actions), prob dists over actions
            # next_actions: (# actors, 1), indices of actions
            # next_action_probs: (# actors, 1), prob of taken actions
            action_pds = self.policy_model(last_states)
            next_actions = self.policy_model.sample(action_pds)
            next_action_log_probs = self.policy_model.get_loglikelihood(action_pds, next_actions)

            next_action_log_probs = next_action_log_probs.unsqueeze(1)
            # shape_equal([self.NUM_ACTORS, 1], next_action_log_probs)

            # if discrete, next_actions is (# actors, 1) 
            # otw if continuous (# actors, 1, action dim)
            next_actions = next_actions.unsqueeze(1)
            # if self.policy_model.discrete:
            #     assert shape_equal([self.NUM_ACTORS, 1], next_actions)
            # else:
            #     assert shape_equal([self.NUM_ACTORS, 1, self.policy_model.action_dim])

            ret = self.multi_actor_step(next_actions, envs)

            # done_info = List of (length, reward) pairs for each completed trajectory
            # (next_rewards, next_states, next_dones) act like multi-actor env.step()
            done_info, next_rewards, next_states, next_not_dones = ret
            # assert shape_equal([self.NUM_ACTORS, 1], next_rewards, next_not_dones)
            # assert shape_equal([self.NUM_ACTORS, 1, self.NUM_FEATURES], next_states)

            # If some of the actors finished AND this is not the last step
            # OR some of the actors finished AND we have no episode information
            if len(done_info) > 0 and (t != self.T - 1 or len(completed_episode_info) == 0):
                completed_episode_info.extend(done_info)

            # Update histories
            # each shape: (nact, t, ...) -> (nact, t + 1, ...)

            pairs = [
                (rewards, next_rewards),
                (not_dones, next_not_dones),
                (actions, next_actions),
                (action_log_probs, next_action_log_probs),
                (states, next_states)
            ]

            last_states = next_states[:, 0, :]
            for total, v in pairs:
                if total is states:
                    total[:, t+1] = v
                else:
                    total[:, t] = v

        # Calculate the average episode length and true rewards over all the trajectories
        infos = np.array(list(zip(*completed_episode_info)))
        if infos.size > 0:
            _, ep_rewards = infos
            avg_episode_length, avg_episode_reward = np.mean(infos, axis=1)
        else:
            ep_rewards = [-1]
            avg_episode_length = -1
            avg_episode_reward = -1

        # Last state is never acted on, discard
        states = states[:,:-1,:]
        trajs = Trajectories(rewards=rewards, 
            action_log_probs=action_log_probs, not_dones=not_dones, 
            actions=actions, states=states)

        to_ret = (avg_episode_length, avg_episode_reward, trajs)
        if return_rewards:
            to_ret += (ep_rewards,)

        return to_ret

    def collect_saps(self, num_saps, should_log=True, return_rewards=False,
                     should_tqdm=False):
        with torch.no_grad():
            # Run trajectories, get values, estimate advantage
            output = self.run_trajectories(num_saps,
                                           return_rewards=return_rewards,
                                           should_tqdm=should_tqdm)

            if not return_rewards:
                avg_ep_length, avg_ep_reward, trajs = output
            else:
                avg_ep_length, avg_ep_reward, trajs, ep_rewards = output

            # If we are sharing weights between the policy network and 
            # value network, we use the get_value function of the 
            # *policy* to # estimate the value, instead of using the value
            # net
            if not self.SHARE_WEIGHTS:
                values = self.val_model(trajs.states).squeeze(-1)
            else:
                values = self.policy_model.get_value(trajs.states).squeeze(-1)

            # Calculate advantages and returns
            advantages, returns = self.advantage_and_return(trajs.rewards,
                                            values, trajs.not_dones)

            trajs.advantages = advantages
            trajs.returns = returns
            trajs.values = values

            assert shape_equal_cmp(trajs.advantages, 
                            trajs.returns, trajs.values)

            # Logging
            if should_log:
                msg = "Current mean reward: %f | mean episode length: %f"
                print(msg % (avg_ep_reward, avg_ep_length))
                self.store.log_table_and_tb('optimization', {
                    'mean_reward': avg_ep_reward
                })

            # Unroll the trajectories (actors, T, ...) -> (actors*T, ...)
            saps = trajs.unroll()

        to_ret = (saps, avg_ep_reward, avg_ep_length)
        if return_rewards:
            to_ret += (ep_rewards,)

        return to_ret

    def take_steps(self, saps, logging=True):
        # Begin advanged logging code
        assert saps.unrolled
        should_adv_log = self.advanced_logging and \
                     self.n_steps % self.log_every == 0 and logging

        self.params.SHOULD_LOG_KL = self.advanced_logging and \
                        self.KL_APPROXIMATION_ITERS != -1 and \
                        self.n_steps % self.KL_APPROXIMATION_ITERS == 0
        store_to_pass = self.store if should_adv_log else None
        # End logging code

        if should_adv_log:
            num_saps = saps.advantages.shape[0]
            val_saps = self.collect_saps(num_saps, should_log=False)[0]

            out_train = self.policy_model(saps.states)
            out_val = self.policy_model(val_saps.states)

            old_pds = select_prob_dists(out_train, detach=True)
            val_old_pds = select_prob_dists(out_val, detach=True)

        # Update the value function before unrolling the trajectories
        # Pass the logging data into the function if applicable
        val_loss = ch.tensor(0.0)
        if not self.SHARE_WEIGHTS:
            val_loss = value_step(saps.states, saps.returns, 
                saps.advantages, saps.not_dones, self.val_model,
                self.val_opt, self.params, store_to_pass).mean()

        if logging:
            self.store.log_table_and_tb('optimization', {
                'final_value_loss': val_loss
            })


        # Take optimizer steps
        args = [saps.states, saps.actions, saps.action_log_probs,
                saps.rewards, saps.returns, saps.not_dones, 
                saps.advantages, self.policy_model, self.params, 
                store_to_pass, self.n_steps]

        self.MAX_KL += self.MAX_KL_INCREMENT 

        # Policy optimization step
        surr_loss = self.policy_step(*args).mean()

        # If the anneal_lr option is set, then we decrease the 
        # learning rate at each training step
        if self.ANNEAL_LR:
            self.POLICY_SCHEDULER.step()
            self.VALUE_SCHEDULER.step()

        if should_adv_log:
            log_value_losses(self, val_saps, 'heldout')
            log_value_losses(self, saps, 'train')
            paper_constraints_logging(self, saps, old_pds,
                            table='paper_constraints_train')
            paper_constraints_logging(self, val_saps, val_old_pds,
                            table='paper_constraints_heldout')

            self.store['paper_constraints_train'].flush_row()
            self.store['paper_constraints_heldout'].flush_row()
            self.store['value_data'].flush_row()

        return surr_loss, val_loss


    def train_step(self):
        '''
        Take a training step, by first collecting rollouts, then 
        calculating advantages, then taking a policy gradient step, and 
        finally taking a value function step.

        Inputs: None
        Returns: 
        - The current reward from the policy (per actor)
        '''
        print("-" * 80)
        start_time = time.time()

        num_saps = self.T * self.NUM_ACTORS
        saps, avg_ep_reward, avg_ep_length = self.collect_saps(num_saps)
        surr_loss, val_loss = self.take_steps(saps)

        # Logging code
        print("Surrogate Loss:", surr_loss.item(), 
                        "| Value Loss:", val_loss.item())
        print("Time elapsed (s):", time.time() - start_time)
        if not self.policy_model.discrete:
            mean_std = ch.exp(self.policy_model.log_stdev).mean()
            print("Agent stdevs: %s" % mean_std)
            self.store.log_table_and_tb('optimization', {
                'mean_std': mean_std
            })
        else:
            self.store['optimization'].update_row({
                'mean_std':np.nan
            })

        self.store['optimization'].flush_row()
        # End logging code

        self.n_steps += 1
        return avg_ep_reward

    @staticmethod
    def agent_from_data(store, row, cpu):
        '''
        Initializes an agent from serialized data (via cox)
        Inputs:
        - store, the name of the store where everything is logged
        - row, the exact row containing the desired data for this agent
        - cpu, True/False whether to use the CPU (otherwise sends to GPU)
        Outputs:
        - agent, a constructed agent with the desired initialization and
              parameters
        - agent_params, the parameters that the agent was constructed with
        '''
        ckpts = store[CKPTS_TABLE]

        get_item = lambda x: list(row[x])[0]

        items = ['val_model', 'policy_model', 'val_opt', 'policy_opt']
        names = {i: get_item(i) for i in items}

        param_keys = list(store['metadata'].df.columns)
        param_values = list(store['metadata'].df.iloc[0,:])

        def process_item(v):
            try:
                return v.item()
            except:
                return v

        param_values = [process_item(v) for v in param_values]
        agent_params = {k:v for k, v in zip(param_keys, param_values)}

        if 'adam_eps' not in agent_params: 
            agent_params['adam_eps'] = 1e-5
        if 'cpu' not in agent_params:
            agent_params['cpu'] = cpu

        agent = Trainer.agent_from_params(agent_params)

        def load_state_dict(model, ckpt_name):
            mapper = ch.device('cuda:0') if not cpu else ch.device('cpu')
            state_dict = ckpts.get_state_dict(ckpt_name, map_location=mapper)
            model.load_state_dict(state_dict)

        load_state_dict(agent.policy_model, names['policy_model'])
        load_state_dict(agent.val_model, names['val_model'])
        if agent.ANNEAL_LR:
            agent.POLICY_SCHEDULER.last_epoch = get_item('iteration')
            agent.VALUE_SCHEDULER.last_epoch = get_item('iteration')
        load_state_dict(agent.POLICY_ADAM, names['policy_opt'])
        load_state_dict(agent.val_opt, names['val_opt'])
        agent.envs = ckpts.get_pickle(get_item('envs'))

        return agent, agent_params

    @staticmethod
    def agent_from_params(params, store=None):
        '''
        Construct a trainer object given a dictionary of hyperparameters.
        Trainer is in charge of sampling trajectories, updating policy network,
        updating value network, and logging.
        Inputs:
        - params, dictionary of required hyperparameters
        - store, a cox.Store object if logging is enabled
        Outputs:
        - A Trainer object for training a PPO/TRPO agent
        '''
        agent_policy = policy_net_with_name(params['policy_net_type'])
        agent_value = value_net_with_name(params['value_net_type'])

        advanced_logging = params['advanced_logging'] and store is not None
        log_every = params['log_every'] if store is not None else 0

        if params['cpu']:
            torch.set_num_threads(1)
        p = Trainer(agent_policy, agent_value, params, store, log_every=log_every,
                    advanced_logging=advanced_logging)

        return p

