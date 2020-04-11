import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box as Continuous
import gym
from .torch_utils import RunningStat, ZFilter, Identity, StateWithTime, RewardFilter

class Env:
    '''
    A wrapper around the OpenAI gym environment that adds support for the following:
    - Rewards normalization
    - State normalization
    - Adding timestep as a feature with a particular horizon T
    Also provides utility functions/properties for:
    - Whether the env is discrete or continuous
    - Size of feature space
    - Size of action space
    Provides the same API (init, step, reset) as the OpenAI gym
    '''
    def __init__(self, game, norm_states, norm_rewards, params, add_t_with_horizon=None, clip_obs=None, clip_rew=None):
        self.env = gym.make(game)
        clip_obs = None if clip_obs < 0 else clip_obs
        clip_rew = None if clip_rew < 0 else clip_rew

        # Environment type
        self.is_discrete = type(self.env.action_space) == Discrete
        assert self.is_discrete or type(self.env.action_space) == Continuous

        # Number of actions
        action_shape = self.env.action_space.shape
        assert len(action_shape) <= 1 # scalar or vector actions
        self.num_actions = self.env.action_space.n if self.is_discrete else 0 \
                            if len(action_shape) == 0 else action_shape[0]
        
        # Number of features
        assert len(self.env.observation_space.shape) == 1
        self.num_features = self.env.reset().shape[0]

        # Support for state normalization or using time as a feature
        self.state_filter = Identity()
        if norm_states:
            self.state_filter = ZFilter(self.state_filter, shape=[self.num_features], \
                                            clip=clip_obs)
        if add_t_with_horizon is not None:
            self.state_filter = StateWithTime(self.state_filter, horizon=add_t_with_horizon)
        
        # Support for rewards normalization
        self.reward_filter = Identity()
        if norm_rewards == "rewards":
            self.reward_filter = ZFilter(self.reward_filter, shape=(), center=False, clip=clip_rew)
        elif norm_rewards == "returns":
            self.reward_filter = RewardFilter(self.reward_filter, shape=(), gamma=params.GAMMA, clip=clip_rew)

        # Running total reward (set to 0.0 at resets)
        self.total_true_reward = 0.0

    def reset(self):
        # Reset the state, and the running total reward
        start_state = self.env.reset()
        self.total_true_reward = 0.0
        self.counter = 0.0
        self.state_filter.reset()
        self.reward_filter.reset()
        return self.state_filter(start_state, reset=True)

    def step(self, action):
        state, reward, is_done, info = self.env.step(action)
        state = self.state_filter(state)
        self.total_true_reward += reward
        self.counter += 1
        _reward = self.reward_filter(reward)
        if is_done:
            info['done'] = (self.counter, self.total_true_reward)
        return state, _reward, is_done, info

    
