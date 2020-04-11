import pandas as pd
import os
import tqdm

from .store import STORE_BASENAME, Store, OBJECT, PICKLE

## collection tools
class CollectionReader:
    '''
    Class for collecting, viewing, and manipulating directories of stores.
    '''
    def __init__(self, directory, log_warnings=True, mode='r', exp_filter=None):
        '''Initialize the CollectionReader object. This will immediately open
        each store in `directory` and see which table are available for viewing.

        Args:
            directory (str) : Path to directory with stores in it. The
                directory should contain directories corresponding to stores.
            log_warnings (bool) : Log warnings if tables with the same name have
                different schemas
            mode (str) : mode to open stores in. Default 'r' (read only), if you
                want to write you will need to make the mode 'a' (append only)
                or 'w' (write).
            exp_filter (method) : Call exp_filter on the experiment id of each
                store, excludes store from collection if it returns `false`.
        '''
        self.stores = {}
        table_list = set()

        self.schemas = {}
        self.directory = directory

        for exp_id in tqdm.tqdm(os.listdir(directory)):
            if exp_filter is not None and not exp_filter(exp_id):
                continue
            store_path = os.path.join(directory, exp_id, STORE_BASENAME)
            if not os.path.exists(store_path):
                continue

            store = Store(self.directory, exp_id, new=False, mode='r')
            self.stores[exp_id] = store
            this_table_list = set(store.tables.keys())
            for i in this_table_list:
                self.schemas[i] = store[i].schema

            table_list |= this_table_list
            #store.close()

        self.tables = table_list

    def close(self):
        '''
        Closes all the stores opened by the collection reader.
        '''
        for v in self.stores.values():
            v.close()

    def __del__(self):
        self.close()

    def __str__(self):
        schemas = self.schemas.items()
        msg = '-- Table: %s --\n%s'
        return '\n'.join(msg % (k, v.keys()) for k,v in schemas)

    def df(self, key, append_exp_id=True, keep_serialized=[],
           union_schemas=False, exp_filter=None, skip_errors=False):
        '''Makes a large concatenated PD dataframe from all the stores' tables
        matching this table key.

        Args:
            key (str) : name of table to collect
            append_exp_id (bool) : if true, append corresponding experiment id
                to each row.
            keep_serialized (list of strings) : list corresponding to column
                names. If in this list, do not unserialize the string within the
                column name and make it a python object within the pandas table.
            union_schemas (bool) : If true, union columns of all collected
                tables, otherwise error out.
            exp_filter (method) : If function of exp_id returns false, ignore
                this store. Otherwise include.
            skip_errors (bool) : If true, skip an experiment upon error occurs.

        Returns: Concatenated dataframe of all corresponding tables in the
            dataframes matching the key.
        '''
        tables = []
        schema = None

        candidates = None
        to_unserialize = None

        for exp_id, store in self.stores.items():
            try:
                if exp_filter is not None and not exp_filter(exp_id):
                    continue

                try:
                    table = store[key]
                except Exception as e:
                    print("Warning: exp_id %s has no table '%s'. Skipping." % (exp_id, key))
                    continue

                df = table.df
                if append_exp_id:
                    df['exp_id'] = exp_id

                this_schema = table.schema

                if not (schema == this_schema or schema is None):
                    if schema is None:
                        print("schema for %s is None!" % exp_id)
                    else:
                        missing_schema_keys = set(schema.keys()) - set(this_schema.keys())
                        new_schema_keys = set(this_schema.keys()) - set(schema.keys())

                        if (len(missing_schema_keys) == 0 and len(new_schema_keys) == 0):
                            for k in this_schema:
                                if this_schema[k] != schema[k]:
                                    if set([this_schema[k], schema[k]]) == set([int, float]):
                                        schema[k] = float
                        else:
                            tup = (missing_schema_keys, new_schema_keys)
                            print('new schema missing keys: %s, new keys: %s' % tup)
                    if union_schemas:
                        schema.update(this_schema)

                schema = schema or this_schema

                if candidates is None:
                    candidates = (k for k, v in schema.items() if v in (store.OBJECT, store.PICKLE))
                    to_unserialize = set(candidates) - set(keep_serialized)

                for col in to_unserialize:
                    replace_col = []
                    if schema[col] == OBJECT:
                        unserialize = lambda x: table.get_object(x)
                    elif schema[col] == store.PICKLE:
                        unserialize = lambda x: table.get_pickle(x)

                    if col in df:
                        for v in df[col]:
                            replace_col.append(unserialize(v))

                    replace_col = pd.Series(replace_col)
                    df[col] = replace_col

                tables.append(df)
            except Exception as e:
                print(f"Encountered error with loading experiment {exp_id}")
                if not skip_errors:
                    raise e

        catted = pd.concat(tables)
        return catted

