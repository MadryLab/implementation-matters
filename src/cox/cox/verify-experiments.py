import argparse
import pandas as pd
import os
import shutil
import json
from .readers import CollectionReader
from .generator import PATH_KEY, consolidate_experiment
from importlib.util import spec_from_file_location, module_from_spec

if  __name__=="__main__":
    parser = argparse.ArgumentParser(description='Verify that all experiments ran successfully (either by config file or by directory)')
    parser.add_argument('-d', '--dir', type=str, required=True)
    parser.add_argument('-o', '--out', type=str)
    parser.add_argument('-m', '--metadata-table', type=str, 
                            default='metadata', 
                            help='The name of the table containing metadata')
    parser.add_argument('-e', '--experiment-config-path', type=str)
    parser.add_argument('-p', '--session-prefix', type=str, default='sess')
    parser.add_argument('-i', '--ignore', nargs='*')
    args = parser.parse_args()

    reader = CollectionReader(args.dir)

    print("Reader initialized. Reading data...")
    df = reader.df(args.metadata_table)
    print("Data has been loaded. Verifying experiments...")

    csv_path = os.path.join(args.dir, "all_experiments.csv")
    params_df = pd.read_csv(csv_path)
    params_dicts = params_df.to_dict(orient='records')
    bad_files = []
    for d in params_dicts:
        q_df = df
        di = list(d.items())
        for k, v in di[1:]:
            if args.ignore is None or not k in args.ignore:
                q_df = q_df[q_df[k] == v]
        if len(q_df) == 0:
            bad_files.append(d)

    if args.out is not None:
        remaining_df = params_df[params_df[PATH_KEY].isin(bad_files)]

        assert args.experiment_config_path is not None
        spec = spec_from_file_location("config", args.experiment_config_path)
        config = module_from_spec(spec)
        spec.loader.exec_module(config)

        [shutil.copy2(x, args.out) for x in remaining_df[PATH_KEY].tolist()]
        for json_file in os.listdir(args.out):
            if not ".json" in json_file:
                continue
            full_path = os.path.join(args.out, json_file)
            d = json.load(open(full_path))
            d['out_dir'] = args.out
            remaining_df.loc[remaining_df[PATH_KEY] == d[PATH_KEY], [PATH_KEY]] = full_path 
            d[PATH_KEY] = full_path
            json.dump(d, open(full_path, "w"))
        csv_path = os.path.join(args.out, "all_experiments.csv")
        remaining_df.to_csv(csv_path)
        job_str = consolidate_experiment(csv_path, config.CMD_MAKER, 
                        config.NUM_SESSIONS, delete_after=False, 
                        session_prefix=args.session_prefix)
        open("run.sh", "w").write(job_str)
    else:
        print("BAD_FILES=", bad_files)
        import pdb; pdb.set_trace()

