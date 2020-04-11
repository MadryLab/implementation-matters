from .utils import dict_product
from uuid import uuid4
import pandas as pd
import json
import os

TMUX_STRING = \
"""tmux new-session -d -s {0} -n 0
tmux send-keys -t {0}:0 "source ~/.exp_rc" Enter
"""

TMUX_JOB = """
tmux send-keys -t {0}:0 "{1}" Enter
"""

DELETE_ALL_JOBS = """
rm -f {0}
"""

PATH_KEY = "cox_experiment_path"

def consolidate_experiment(exp_csv, command_maker, num_sess, delete_after=True, session_prefix="sess"):
    exp_df = pd.read_csv(exp_csv)
    job_str = ""
    for worker in range(min(num_sess, len(exp_df))):
        job_str += TMUX_STRING.format(session_prefix + str(worker))

    for i, exp_path in enumerate(exp_df[PATH_KEY]):
        assert type(exp_path) == str
        params = json.load(open(exp_path))
        job_str += TMUX_JOB.format(session_prefix + str(i % num_sess), command_maker(i, params))
        if delete_after:
            job_str += TMUX_JOB.format(session_prefix + str(i % num_sess), "rm -f %s" % (exp_path,))
    return job_str


def generate_experiments(base, params_to_vary, exp_dir, rules=[], sort_by=None):
    '''
    Given a function to test, a base set of hyperparameters to vary, 
    we generate the appropriate config files and scripts to run the 
    tests and collect the results.
    The results will be collected over a grid of hyperparameters given
    by a cartesian product, but filtered by a set of specified rules.
    Inputs:
    - base, a dictionary {str: val} with the default values for all  
     the necessary hyperparameters 
    - params, a dictionary {str: list[val]} with the values to be tested
    for each hyperparameters
    - exp_dir, a string path to the directory where the experiments
    are to be saved
    '''
    params_df_cols = set(base.keys()) | set(params_to_vary.keys()) | {PATH_KEY}
    params_df = pd.DataFrame(columns=list(params_df_cols))

    all_test_params = dict_product(params_to_vary)
    if sort_by:
        if not isinstance(sort_by, list):
            sort_by = [sort_by]
        for i in reversed(sort_by):
            all_test_params = sorted(all_test_params, key=lambda d:-d[i])

    filt_params = all_test_params
    exp_uid = str(uuid4())

    for rule in rules:
        filt_params = list(filter(rule, filt_params))

    for i, fd in enumerate(filt_params):
        job_name = "job%d-%s" % (i, exp_uid)
        new_job = dict(base, **fd)
        fname = os.path.join(exp_dir, "%s.json" % (job_name,))
        new_job[PATH_KEY] = fname
        params_df.loc[i,list(new_job.keys())] = list(new_job.values())
        json.dump(new_job, open(fname, "w"))

    csv_path = os.path.join(exp_dir, "all_experiments.csv")
    params_df.to_csv(csv_path)
    return csv_path

if  __name__=="__main__":
    base = {"a": 1, "b":2, "c": 3}
    params = {"a": [1,2,3], "b": [4,5,6,7]}
    cmd_maker = lambda i, d: "CUDA_VISIBLE_DEVICES=%d python %s --base-config %s" \
                                % (i % 8, "main.py", d["cox_experiment_path"])
    rules = [lambda d: (d["b"] % d["a"] == 0)]
    csv_path = generate_experiments("main.py", base, params, '/tmp/silly', rules=rules)
    consolidate_experiment(csv_path, cmd_maker, 8)
