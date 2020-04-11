import torch as ch
import numpy as np
from .torch_utils import *
from torch.nn.utils import parameters_to_vector as flatten
from torch.nn.utils import vector_to_parameters as assign
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from .steps import value_loss_returns, value_loss_gae, adv_normalize

#####
# Understanding TRPO approximations for KL constraint
#####

def paper_constraints_logging(agent, saps, old_pds, table):
    new_pds = agent.policy_model(saps.states)
    new_log_ps = agent.policy_model.get_loglikelihood(new_pds,
                                                    saps.actions)

    ratios = ch.exp(new_log_ps - saps.action_log_probs)
    max_rat = ratios.max()

    kls = agent.policy_model.calc_kl(old_pds, new_pds)
    avg_kl = kls.mean()

    row = {
        'avg_kl':avg_kl,
        'max_ratio':max_rat,
        'opt_step':agent.n_steps,
    }

    for k in row:
        if k != 'opt_step':
            row[k] = float(row[k])

    agent.store.log_table_and_tb(table, row)

##
# Treating value learning as a supervised learning problem:
# How well do we do?
##
def log_value_losses(agent, saps, label_prefix, table='value_data'):
    '''
    Computes the validation loss of the value function modeling it 
    as a supervised learning of returns. Calculates the loss using 
    all three admissible loss functions (returns, consistency, mixed).
    Inputs: None
    Outputs: None, logs to the store 
    '''
    with ch.no_grad():
        # Compute validation loss
        new_values = agent.val_model(saps.states).squeeze(-1)
        args = [new_values, saps.returns, saps.advantages, saps.not_dones,
                agent.params, saps.values]
        returns_loss, returns_mre, returns_msre = value_loss_returns(*args, re=True)
        gae_loss, gae_mre, gae_msre = value_loss_gae(*args, re=True)

        agent.store.log_table_and_tb(table, {
            ('%s_returns_loss' % label_prefix): returns_loss,
            ('%s_gae_loss' % label_prefix): gae_loss,
        })
