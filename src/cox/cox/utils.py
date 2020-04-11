import GPUtil
import argparse
import json
from collections import namedtuple
import os
import dill as pickle
import codecs
import itertools

def consistent(old, new):
    '''
    Asserts that either first argument is None or
    both arguments are equal, and returns the non-None
    argument.
    '''
    if old is None:
        return new
    assert old == new
    return old

def override_json(args, json_path, check_consistency=False):
    json_params = json.load(open(json_path))

    params = args.as_dict()

    if check_consistency:
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
    for k in [k for k in params if params[k] is None and not (k in json_params)]:
        json_params[k] = None

    params = json_params
    # for k, v in params.items():
    #     assert v is not None, k

    args = Parameters(params)
    return args

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

class Parameters():
    '''
    Parameters class, just a nice way of accessing a dictionary
    > ps = Parameters({"a": 1, "b": 3})
    > ps.A # returns 1
    > ps.B # returns 3
    '''
    def __init__(self, params):
        super().__setattr__('params', params)

        # ensure no overlapping (in case) params
        collisions = set()
        for k in self.params.keys():
            collisions.add(k.lower())

        assert len(collisions) == len(self.params.keys())

    def as_dict(self):
        return self.params

    def __getattr__(self, x):
        if x in vars(self):
            return vars(self)[x]

        k = x.lower()
        if k not in self.params:
            return None

        return self.params[k]

    def __setattr__(self, x, v):
        if x in vars(self):
            vars(self)[x.lower()] = v

        self.params[x.lower()] = v

    def __delattr__ (self, key):
        del self.params[key]

    def __iter__ (self):
        return iter(self.params)

    def __len__ (self):
        return len(self.params)

    def __str__(self):
        return json.dumps(self.params, indent=2)

    def __repr__(self):
        return str(self)

    def __getstate__(self):
        return self.params

    def __contains__(self, x):
        return x in self.params

    def __setstate__(self, x):
        self.params = x

def mkdirp(x, should_msg=False):
    '''
    Tries to make a directory, but doesn't error if the
    directory exists/can't be created.
    '''
    try:
        os.makedirs(x)
    except Exception as e:
        if should_msg:
            print("Failed to make directory (might already exist). \
            Exact message was %s" % (e.message,))

def obj_to_string(obj):
    return codecs.encode(pickle.dumps(obj), "base64").decode()

def string_to_obj(s):
    if s is None or s == "":
        return None
    if not isinstance(s, str):
        return s
    try:
        return pickle.loads(codecs.decode(s.encode(), "base64"))
    except Exception as e:
        return s

def available_gpus(frac_cpu=0.5, frac_mem=0.5):
    gpus = GPUtil.getGPUs()

    def should_use_gpu(v):
        return v.load < frac_cpu and v.memoryUsed/v.memoryTotal < frac_mem 

    return [i for i, v in enumerate(gpus) if should_use_gpu(v)]
