from .generator import generate_experiments, PATH_KEY, consolidate_experiment
import os
import tempfile
import subprocess
import json
import argparse
from importlib.util import spec_from_file_location, module_from_spec

def main():
    parser = argparse.ArgumentParser(description='Generate experiments to be run.')
    parser.add_argument('-b', '--base-config-path', type=str)
    parser.add_argument('-o', '--out-dir', type=str, required=True)
    parser.add_argument('-e', '--experiment-config-path', type=str, required=True)
    parser.add_argument('-r', '--run', action='store_true')
    parser.add_argument('-p', '--session-prefix', type=str, default='sess')
    args = parser.parse_args()

    spec = spec_from_file_location("config", args.experiment_config_path)
    config = module_from_spec(spec)
    spec.loader.exec_module(config)

    try:
        assert not any([x is None for x in [config.NUM_SESSIONS, config.CMD_MAKER, \
                                                config.RULES, config.PARAMS]])
    except:
        raise ValueError("experiment-config-path should be a pointer to a python file \
                                with NUM_SESSIONS, CMD_MAKER, RULES, and PARAMS defined. \
                                To generate an example experimental config file with comments, \
                                run 'python -m cox.generate-config.py'.")

    if args.base_config_path:
        base = json.load(open(args.base_config_path))
    else:
        base = {}
    base['out_dir'] = args.out_dir

    gen_kwargs = {
        'rules':config.RULES
    }

    try:
        gen_kwargs['sort_by'] = config.SORT_BY
    except:
        pass

    csv_path = generate_experiments(base, config.PARAMS, args.out_dir, **gen_kwargs)
    job_str = consolidate_experiment(csv_path, config.CMD_MAKER, config.NUM_SESSIONS,
                                     delete_after=False, session_prefix=args.session_prefix)

    if not args.run:
        open("run.sh", "w").write(job_str)
    else:
        new_file, filename = tempfile.mkstemp()
        os.write(new_file, str.encode(job_str))
        subprocess.call("bash %s" % filename, shell=True)

if  __name__=="__main__":
    main()
