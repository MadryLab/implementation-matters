import pandas as pd
import torch as ch
import numpy as np
import dill as pickle
from uuid import uuid4
from .utils import *
import os
import warnings
from tensorboardX import SummaryWriter

TABLE_OBJECT_DIR = '.table_objects'
SAVE_DIR = 'save'
STORE_BASENAME = 'store.h5'
TENSORBOARD_DIR = 'tensorboard'
COX_DATA_KEY = 'COX_DATA'
PICKLE = '__pickle__'
OBJECT = '__object__'
PYTORCH_STATE = '__pytorch_state__'

pd.set_option('io.hdf.default_format','table')
from pandas.io.pytables import PerformanceWarning
warnings.simplefilter(action="ignore", category=PerformanceWarning)

class Store():
    '''Serializes and saves data from experiment runs. Automatically makes a
    tensorboard. Access the tensorboard field, and refer to the TensorboardX
    documentation for more information about how to manipulate it (it is a
    tensorboardX object).

    Directly saves: int, float, torch scalar, string
    Saves and links: np.array, torch tensor, python object (via pickle or
        pytorch serialization)

    Note on python object serialization: you can choose one of three options to
    serialize using: `OBJECT` (store as python serialization inline), `PICKLE`
    (store as python serialization on disk), or `PYTORCH_STATE` (save as
    pytorch serialization on disk). All these types are represented as
    properties, i.e. `store_instance.PYTORCH_STATE`. You will need to manually
    decode the objects using the static methods found in the `Table` class
    (`get_pytorch_state`, `get_object`, `get_pickle`).
    '''

    OBJECT = OBJECT
    PICKLE = PICKLE
    PYTORCH_STATE = PYTORCH_STATE
    def __init__(self, storage_folder, exp_id=None, new=False, mode='a'):
        '''
        Make new experiment store in `storage_folder`, within its subdirectory
        `exp_id` (if not none). If an experiment exists already with this
        corresponding directory, open it for reading.

        Args:
            storage_folder (str) : parent folder in which we will put a folder
                with all our experiment data (this store).
            exp_id (str) : dir name in `storage_folder` under which we will
                store experimental data.
            new (str): enforce that this store has never been created before.
            mode (str) : mode for accessing tables. a is append only, r is read
                only, w is write.

        '''
        if not exp_id:
            exp_id = str(uuid4())

        exp_path = os.path.join(storage_folder, exp_id)
        if os.path.exists(exp_path) and new:
            raise ValueError("This experiment has already been run.")

        if not os.path.exists(exp_path):
            mkdirp(exp_path)
            print('Logging in: %s' % os.path.abspath(exp_path))

        # Start HDF file
        self.store = pd.HDFStore(os.path.join(exp_path, STORE_BASENAME), mode=mode)

        # Setup
        self.exp_id = exp_id
        self.path = os.path.abspath(exp_path)
        self.save_dir = os.path.join(exp_path, SAVE_DIR)
        self.tb_dir = os.path.join(exp_path, TENSORBOARD_DIR)

        # Where to save table objects
        self._table_object_dir = os.path.join(exp_path, TABLE_OBJECT_DIR)

        if mode != 'r':
            self.tensorboard = SummaryWriter(self.tb_dir)
            mkdirp(self.save_dir)
            mkdirp(self._table_object_dir)
            mkdirp(self.tb_dir)

        self.tables = Table._tables_from_store(self.store, self._table_object_dir)
        self.keys = self.tables.keys()

    def close(self):
        '''
        Closes underlying HDFStore of this store.
        '''
        self.store.close()

    def __str__(self):
        s = []
        for table_name, table in self.tables.items():
            s.append('-- Table: %s --' % table_name)
            s.append(str(table))
            s.append('')

        return '\n'.join(s)

    def get_table(self, table_id):
        '''
        Gets table with key `table_id`.

        Args:
            table_id (str) : id of table to get from this store.

        Returns:
            The corresponding table (Table object).
        '''
        return self.tables[table_id]

    def __getitem__(self, table_id):
        '''
        Gets table with key `table_id`.

        Args:
            table_id (str) : id of table to get from this store.

        Returns:
            The corresponding table (Table object).
        '''
        return self.get_table(table_id)

    def add_table(self, table_name, schema):
        '''
        Add a new table to the experiment.

        Args:
            table_name (str) : a name for the table
            schema (dict) : a dict for the schema of the table. The entries
                should be of the form name:type. For example, if we wanted to
                add a float column in the table named acc, we would have an
                entry `'acc':float`.

        Returns:
            The table object of the new table.
        '''
        table = Table(table_name, schema, self._table_object_dir, self.store)
        self.tables[table_name] = table
        return table

    def add_table_like_example(self, table_name, example, alternative=OBJECT):
        '''
        Add a new table to the experiment, using an example dictionary as the
        basis for the types of the columns.

        Args:
            table_name (str) : a name for the table
            example (dict) : example for the schema of the table. Make a table
                with columns with types corresponding to the types of the
                objects in the dictionary.
            alternative (self.OBJECT|self.PICKLE|self.PYTORCH_STATE) : how to
                store columns that are python objects.
        '''
        schema = schema_from_dict(example, alternative=alternative)
        return self.add_table(table_name, schema)

    def log_table_and_tb(self, table_name, update_dict, summary_type='scalar'):
        '''
        Log to a table and also a tensorboard.

        Args:
            table_name (str) : which table to log to
            update_dict (dict) : values to log and store as a dictionary of
                column mapping to value.
            summary_type (str) : what type of summary to log to tensorboard as
        '''

        table = self.tables[table_name]
        update_dict = _clean_dict(update_dict, table.schema)

        tb_func = getattr(self.tensorboard, 'add_%s' % summary_type)
        iteration = table.nrows

        for name, value in update_dict.items():
            tb_func('/'.join([table_name, name]), value, iteration)

        table.update_row(update_dict)

class Table():
    '''
    A class representing a single storer table, to be written to by
    the experiment. This is essentially a single HDFStore table.
    '''

    def _tables_from_store(store, table_obj_dir):
        tables = {}
        for key in store.keys():
            storer = store.get_storer(key)
            if COX_DATA_KEY in storer.attrs:
                data = storer.attrs[COX_DATA_KEY]
                name = data['name']
                table = Table(name, data['schema'], table_obj_dir, store,
                              has_initialized=True)
                tables[name] = table

        return tables

    def __str__(self):
        s = str(self.df)
        if len(s.split('\n')) > 5:
            s = str(self.df[:4]) + '\n ... (%s rows hidden)' % self.df.shape[0]
        return s

    def __init__(self, name, schema, table_obj_dir, store,
                 has_initialized=False):
        '''
        Create a new Table object.

        Args:
            name (str) : name of table
            schema (dict) : schema of table (as described in `store` class)
            table_obj_dir (str) : where to store serialized objects on disk
            store (Store) : parent store.
            has_initialized (bool) : has this table been created yet.
        '''
        self._name = name
        self._schema = schema
        self._HDFStore = store
        self._curr_row_data = None
        self._table_obj_dir = table_obj_dir
        self._has_initialized = has_initialized

        self._create_row()

    @property
    def df(self):
        '''
        Access the underlying pandas dataframe for this table.
        '''
        if self._has_initialized:
            return self._HDFStore[self._name]
        else:
            return pd.DataFrame(columns=self._schema.keys())

    @property
    def schema(self):
        '''
        Access the underlying schema for this table.
        '''
        return dict(self._schema)

    @property
    def nrows(self):
        '''
        How many rows this table has.
        '''
        if self._has_initialized:
            return self._HDFStore.get_storer(self._name).nrows
        else:
            return 0

    def _initialize_nonempty_table(self):
        self._HDFStore.get_storer(self._name).attrs[COX_DATA_KEY] = {
            'schema':self._schema,
            'name':self._name,
        }

        self._has_initialized = True

    def append_row(self, data):
        '''
        Write a dictionary with format column name:value as a row to the table.
        Must have a value for each column. See `update_row` for more mechanics.

        Args:
            data (dict) : dictionary with format `column name`:`value`.
        '''
        self.update_row(data)
        self.flush_row()

    def _create_row(self):
        assert self._curr_row_data is None

        curr_row_dict = {s: None for s in self._schema}
        self._curr_row_data = curr_row_dict

    def update_row(self, data):
        '''
        Update the currently considered row in the data store. Our database is
        append only using the `Table` API. We can update this single row as much
        as we desire, using column:value mappings in `data`. Eventually, the
        currently considered row must be written to the database using
        `flush_row`. This model allows for writing rows easily when not all the
        values are known in a single context. Each `data` object does not need
        to contain every column, but by the time that the row is flushed every
        column must obtained a value. This update model is stateful.

        Python primitives (`int`, `float`, `str`, `bool`), and their numpy
        equivalents are written automatically to the row. All other objects are
        serialized (see `Store`).

        Args:
            data (dict) : a dictionary with format `column name`:`value`.
        '''
        # Data sanity checks
        assert self._curr_row_data is not None
        assert len(set(data.keys())) == len(data.keys())

        if any([k not in self._schema for k in data]):
            raise ValueError("Got keys that are undeclared in schema")

        for k, v in data.items():
            v_type = self._schema[k]
            if v_type == OBJECT:
                to_store = obj_to_string(v)
            elif v_type == PICKLE or v_type == PYTORCH_STATE:
                uid = str(uuid4())
                fname = os.path.join(self._table_obj_dir, uid)
                if v_type == PICKLE:
                    with open(fname, 'wb') as f:
                        pickle.dump(v, f)
                else:
                    if 'state_dict' in dir(v):
                        v = v.state_dict()
                    ch.save(v, fname, pickle_module=pickle)
                to_store = uid
            else:
                to_store = v_type(v)
                assert to_store is not None

            self._curr_row_data[k] = to_store

    def get_pickle(self, uid):
        '''
        Unserialize object of store.PICKLE type (a pickled object stored as a
        string on disk).

        Args:
            uid (str) : identifier corresponding to stored object in the table.
        '''
        fname = os.path.join(self._table_obj_dir, uid)
        with open(fname, 'rb') as f:
            obj = pickle.load(f)

        return obj

    def get_state_dict(self, uid, **kwargs):
        '''
        Unserialize object of store.PYTORCH_STATE type (object stored using
        pytorch's serialization system).

        Args:
            uid (str) : identifier corresponding to stored object in the table.
        '''
        fname = os.path.join(self._table_obj_dir, uid)
        kwargs['pickle_module'] = pickle
        return ch.load(fname, **kwargs)

    def get_object(self, s):
        '''
        Unserialize object of store.OBJECT type (a pickled object stored as a
        string in the table).

        Args:
            s (str) : pickle string to unpickle into a python object.
        '''
        return string_to_obj(s)

    def flush_row(self):
        '''
        Writes the current row we have staged (using `update_row`) to the table.
        Another row is immediately staged for `update_row` to act on.
        '''
        self._curr_row_data = _clean_dict(self._curr_row_data, self._schema)

        for k in self._schema:
            try:
                assert self._curr_row_data[k] is not None
            except:
                dne = not (k in self._curr_row_data)
                if dne:
                    msg = 'Col %s does not exist!' % k
                else:
                    msg = 'Col %s is None!' % k

                raise ValueError(msg)

        for k, v in self._curr_row_data.items():
            self._curr_row_data[k] = [v]

        df = pd.DataFrame(self._curr_row_data)

        try:
            nrows = self._HDFStore.get_storer(self._name).nrows
        except:
            nrows = 0

        df.index += nrows
        self._HDFStore.append(self._name, df, table=True)

        if not self._has_initialized:
             self._initialize_nonempty_table()

        self._curr_row_data = None
        self._create_row()

def schema_from_dict(d, alternative=OBJECT):
    '''
    Given a dictionary mapping column names to values, make a corresponding
    schema.

    Args:
        d (dict) : dict of values we are going to infer the schema from
        alternative (self.OBJECT|self.PICKLE|self.PYTORCH_STATE) : how to
            store columns that are python objects.
    '''
    natural_types = set([int, str, float, bool])
    schema = {}
    for k, v in d.items():
        t = type(v)
        if t in natural_types:
            schema[k] = t
        else:
            schema[k] = alternative

    return schema

def _clean_dict(d, schema):
    d = dict(d)
    for k, v in d.items():
        v_type = schema[k]
        if v_type in [int, float, bool]:
            if type(v) == ch.Tensor or type(v) == np.ndarray:
                if v.shape == ():
                    v = v_type(v)
                    d[k] = v
    return d
