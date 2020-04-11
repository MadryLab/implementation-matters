import shutil
from . import readers
import os
import numpy as np
import sys
import tempfile

from . import store

d = tempfile.mkdtemp()

def make_store(exp_id):
    s = store.Store(d, exp_id=exp_id)
    dtypes = {
        'int':int,
        'str':str,
        'pickle':store.PICKLE,
        'obj':store.OBJECT
    }

    s.add_table('table1', dtypes)
    s['table1'].append_row({
        'int':999,
        'str':'Non, merci!',
        'pickle':np.random.rand(1, 5),
        'obj':np.random.rand(8, 3),
    })
    
    s.add_table('table2', dtypes)
    t2 = s['table2']
    t2.append_row({
        'int':103,
        'str':'foo',
        'pickle':np.random.rand(103, 5),
        'obj':np.random.rand(200, 31),
    })

    s['table2'].update_row({
        'int':103,
        'str':'foo',
    })

    s['table2'].update_row({
        'int':444,
        'str':'bar',
        'pickle':np.random.rand(3, 7),
        'obj':np.random.rand(91, 9),
    })
    
    s['table2'].flush_row()

make_store('andrew.ilyas')
make_store('ILYAS')

reader = readers.CollectionReader(d)
df = reader.df('table2')
print(df)

print(df[df['exp_id'] == 'ILYAS'])

