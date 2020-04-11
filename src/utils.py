import numpy as np
import itertools

def dict_product(d):
    '''
    Implementing itertools.product for dictionaries.
    E.g. {"a": [1,4],  "b": [2,3]} -> [{"a":1, "b":2}, {"a":1,"b":3} ..]
    Inputs:
    - d, a dictionary {key: [list of possible values]}
    Returns;
    - A list of dictionaries with every possible configuration
    '''
    keys = d.keys()
    vals = d.values()
    prod_values = list(itertools.product(*vals))
    all_dicts = map(lambda x: dict(zip(keys, x)), prod_values)
    return all_dicts

def iwt(start, end, interval, trials):
    return list(np.arange(start, end, interval))*trials
