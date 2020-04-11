import torch.nn as nn
import math
import torch as ch
import torch.nn.functional as F
from .torch_utils import *

'''
Neural network models for estimating value and policy functions
Contains:
- Initialization utilities
- Value Network(s)
- Policy Network(s)
- Retrieval Function
'''

########################
### INITIALIZATION UTILITY FUNCTIONS:
# initialize_weights
########################

HIDDEN_SIZES = (64, 64)
ACTIVATION = nn.Tanh
STD = 2**0.5

def initialize_weights(mod, initialization_type, scale=STD):
    '''
    Weight initializer for the models.
    Inputs: A model, Returns: none, initializes the parameters
    '''
    for p in mod.parameters():
        if initialization_type == "normal":
            p.data.normal_(0.01)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")
            

########################
### INITIALIZATION UTILITY FUNCTIONS:
# Generic Value network, Value network MLP
########################

class ValueDenseNet(nn.Module):
    '''
    An example value network, with support for arbitrarily many
    fully connected hidden layers (by default 2 * 128-neuron layers),
    maps a state of size (state_dim) -> a scalar value.
    '''
    def __init__(self, state_dim, init, hidden_sizes=(64, 64)):
        '''
        Initializes the value network.
        Inputs:
        - state_dim, the input dimension of the network (i.e dimension of state)
        - hidden_sizes, an iterable of integers, each of which represents the size
        of a hidden layer in the neural network.
        Returns: Initialized Value network
        '''
        super().__init__()
        self.activation = ACTIVATION()
        self.affine_layers = nn.ModuleList()

        prev = state_dim
        for h in hidden_sizes:
            l = nn.Linear(prev, h)
            initialize_weights(l, init)
            self.affine_layers.append(l)
            prev = h

        self.final = nn.Linear(prev, 1)
        initialize_weights(self.final, init, scale=1.0)

    def forward(self, x):
        '''
        Performs inference using the value network.
        Inputs:
        - x, the state passed in from the agent
        Returns:
        - The scalar (float) value of that state, as estimated by the net
        '''
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        value = self.final(x)
        return value

    def get_value(self, x):
        return self(x)

########################
### POLICY NETWORKS
# Discrete and Continuous Policy Examples
########################

'''
A policy network can be any class which is initialized 
with a state_dim and action_dim, as well as optional named arguments.
Must provide:
- A __call__ override (or forward, for nn.Module): 
    * returns a tensor parameterizing a distribution, given a 
    BATCH_SIZE x state_dim tensor representing shape
- A function calc_kl(p, q): 
    * takes in two batches tensors which parameterize probability 
    distributions (of the same form as the output from __call__), 
    and returns the KL(p||q) tensor of length BATCH_SIZE
- A function entropies(p):
    * takes in a batch of tensors parameterizing distributions in 
    the same way and returns the entropy of each element in the 
    batch as a tensor
- A function sample(p): 
    * takes in a batch of tensors parameterizing distributions in
    the same way as above and returns a batch of actions to be 
    performed
- A function get_likelihoods(p, actions):
    * takes in a batch of parameterizing tensors (as above) and an 
    equal-length batch of actions, and returns a batch of probabilities
    indicating how likely each action was according to p.
'''

class DiscPolicy(nn.Module):
    '''
    A discrete policy using a fully connected neural network.
    The parameterizing tensor is a categorical distribution over actions
    '''
    def __init__(self, state_dim, action_dim, init, hidden_sizes=HIDDEN_SIZES, time_in_state=False, share_weights=False):
        '''
        Initializes the network with the state dimensionality and # actions
        Inputs:
        - state_dim, dimensionality of the state vector
        - action_dim, # of possible discrete actions
        - hidden_sizes, an iterable of length #layers,
            hidden_sizes[i] = number of neurons in layer i
        - time_in_state, a boolean indicating whether the time is 
            encoded in the state vector
        '''
        super().__init__()
        self.activation = ACTIVATION()
        self.time_in_state = time_in_state

        self.discrete = True
        self.affine_layers = nn.ModuleList()
        prev_size = state_dim
        for i in hidden_sizes:
            lin = nn.Linear(prev_size, i)
            initialize_weights(lin, init)
            self.affine_layers.append(lin)
            prev_size = i

        self.final = nn.Linear(prev_size, action_dim)

        # For the case where we want to share parameters 
        # between the policy and value networks
        self.share_weights = share_weights
        if share_weights:
            if time_in_state:
                self.final_value = nn.Linear(prev_size + 1, 1)
            else:
                self.final_value = nn.Linear(prev_size, 1)

            initialize_weights(self.final_value, init, scale=1.0)

    def forward(self, x):
        '''
        Outputs the categorical distribution (via softmax)
        by feeding the state through the neural network
        '''
        # If the time is in the state, discard it
        if self.time_in_state:
            x = x[:,:-1]
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        
        probs = F.softmax(self.final(x))
        return probs

    def calc_kl(self, p, q, get_mean=True): # TODO: does not return a list
        '''
        Calculates E KL(p||q):
        E[sum p(x) log(p(x)/q(x))]
        Inputs:
        - p, first probability distribution (NUM_SAMPLES, NUM_ACTIONS)
        - q, second probability distribution (NUM_SAMPLES, NUM_ACTIONS)
        Returns:
        - Empirical KL from p to q
        '''
        p, q = p.squeeze(), q.squeeze()
        assert shape_equal_cmp(p, q)
        kl = (p * (ch.log(p) - ch.log(q))).sum(-1)
        return kl

    def entropies(self, p):
        '''
        p is probs of shape (batch_size, action_space). return mean entropy
        across the batch of states
        '''
        entropies = (p * ch.log(p)).sum(dim=1)
        return entropies

    def get_loglikelihood(self, p, actions):
        '''
        Inputs:
        - p, batch of probability tensors
        - actions, the actions taken
        '''
        try:
            dist = ch.distributions.categorical.Categorical(p)
            return dist.log_prob(actions)
        except Exception as e:
            raise ValueError("Numerical error")
    
    def sample(self, probs):
        '''
        given probs, return: actions sampled from P(.|s_i), and their
        probabilities
        - s: (batch_size, state_dim)
        Returns actions:
        - actions: shape (batch_size,)
        '''
        dist = ch.distributions.categorical.Categorical(probs)
        actions = dist.sample()
        return actions.long()

    def get_value(self, x):
        # If the time is in the state, discard it
        assert self.share_weights, "Must be sharing weights to use get_value"
        t = None
        if self.time_in_state:
            t = x[...,-1:]
            x = x[...,:-1]

        for affine in self.affine_layers:
            x = self.activation(affine(x))

        if self.time_in_state:
            return self.final_value(ch.cat((x, t), -1))
        else:
            return self.final_value(x)


class CtsPolicy(nn.Module):
    '''
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and standard deviation vector, 
    which parameterize a gaussian distribution.
    '''
    def __init__(self, state_dim, action_dim, init, hidden_sizes=HIDDEN_SIZES,
                 time_in_state=False, share_weights=False):
        super().__init__()
        self.activation = ACTIVATION()
        self.action_dim = action_dim
        self.discrete = False
        self.time_in_state = time_in_state

        self.affine_layers = nn.ModuleList()
        prev_size = state_dim
        for i in hidden_sizes:
            lin = nn.Linear(prev_size, i)
            initialize_weights(lin, init)
            self.affine_layers.append(lin)
            prev_size = i

        self.final_mean = nn.Linear(prev_size, action_dim)
        initialize_weights(self.final_mean, init, scale=0.01)
        
        # For the case where we want to share parameters 
        # between the policy and value networks
        self.share_weights = share_weights
        if share_weights:
            if time_in_state:
                self.final_value = nn.Linear(prev_size + 1, 1)
            else:
                self.final_value = nn.Linear(prev_size, 1)

            initialize_weights(self.final_value, init, scale=1.0)

        stdev_init = ch.zeros(action_dim)
        self.log_stdev = ch.nn.Parameter(stdev_init)

    def forward(self, x):
        # If the time is in the state, discard it
        if self.time_in_state:
            x = x[:,:-1]
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        
        means = self.final_mean(x)
        std = ch.exp(self.log_stdev)

        return means, std 

    def get_value(self, x):
        assert self.share_weights, "Must be sharing weights to use get_value"

        # If the time is in the state, discard it
        t = None
        if self.time_in_state:
            t = x[...,-1:]
            x = x[...,:-1]

        for affine in self.affine_layers:
            x = self.activation(affine(x))

        if self.time_in_state:
            return self.final_value(ch.cat((x, t), -1))
        else:
            return self.final_value(x)

    def sample(self, p):
        '''
        Given prob dist (mean, var), return: actions sampled from p_i, and their
        probabilities. p is tuple (means, var). means shape 
        (batch_size, action_space), var (action_space,), here are batch_size many
        prboability distributions you're sampling from

        Returns tuple (actions, probs):
        - actions: shape (batch_size, action_dim)
        - probs: shape (batch_size, action_dim)
        '''
        means, std = p
        return (means + ch.randn_like(means)*std).detach()

    def get_loglikelihood(self, p, actions):
        try:    
            mean, std = p
            nll =  0.5 * ((actions - mean) / std).pow(2).sum(-1) \
                   + 0.5 * np.log(2.0 * np.pi) * actions.shape[-1] \
                   + self.log_stdev.sum(-1)
            return -nll
        except Exception as e:
            raise ValueError("Numerical error")

    def calc_kl(self, p, q):
        '''
        Get the expected KL distance between two sets of gaussians over states -
        gaussians p and q where p and q are each tuples (mean, var)
        - In other words calculates E KL(p||q): E[sum p(x) log(p(x)/q(x))]
        - From https://stats.stackexchange.com/a/60699
        '''
        p_mean, p_std = p
        q_mean, q_std = q
        p_var, q_var = p_std.pow(2), q_std.pow(2)
        assert shape_equal([-1, self.action_dim], p_mean, q_mean)
        assert shape_equal([self.action_dim], p_var, q_var)

        d = q_mean.shape[1]
        diff = q_mean - p_mean

        log_quot_frac = ch.log(q_var).sum() - ch.log(p_var).sum()
        tr = (p_var / q_var).sum()
        quadratic = ((diff / q_var) * diff).sum(dim=1)

        kl_sum = 0.5 * (log_quot_frac - d + tr + quadratic)
        assert kl_sum.shape == (p_mean.shape[0],)
        return kl_sum

    def entropies(self, p):
        '''
        Get entropies over the probability distributions given by p
        p_i = (mean, var), p mean is shape (batch_size, action_space),
        p var is shape (action_space,)
        '''
        _, std = p
        detp = determinant(std)
        d = std.shape[0]
        entropies = ch.log(detp) + .5 * (d * (1. + math.log(2 * math.pi)))
        return entropies

## Retrieving networks
# Make sure to add newly created networks to these dictionaries!

POLICY_NETS = {
    "DiscPolicy": DiscPolicy,
    "CtsPolicy": CtsPolicy
}

VALUE_NETS = {
    "ValueNet": ValueDenseNet,
}

def policy_net_with_name(name):
    return POLICY_NETS[name]

def value_net_with_name(name):
    return VALUE_NETS[name]
