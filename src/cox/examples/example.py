import sys
import os
from cox.store import Store
from cox.readers import CollectionReader
import shutil
import numpy as np
import itertools

OUT_DIR = '/tmp/cox_example/'

try:
    shutil.rmtree(OUT_DIR)
except:
    pass

os.mkdir(OUT_DIR)

def f(x):
    return (x - 2.03)**2 + 3

if __name__ == "__main__":
    # Three parameters: initial guess for x, step size, tolerance
    combos = itertools.product(np.linspace(-15, 15, 3), np.linspace(1, 1e-5, 3),
                                [1e-5])

    for x, step, tol in combos:
        store = Store(OUT_DIR)
        store.add_table('metadata', {
            'step_size': float,
            'tolerance': float,
            'initial_x': float,
            'out_dir': str
        })

        store.add_table('result', {
            'final_x': float,
            'final_opt':float
        })

        store['metadata'].append_row({
            'step_size': step,
            'tolerance': tol,
            'initial_x': x,
            'out_dir': '/tmp/'
        })


        for _ in range(1000):
            # Take a uniform step in the direction of decrease
            if f(x + step) < f(x - step):
                x += step
            else:
                x -= step

            # If the difference between the directions
            # is less than the tolerance, stop
            if f(x + step) - f(x - step) < tol:
                break

        store['result'].update_row({
            'final_x': x
        })

        store['result'].update_row({
            'final_opt':f(x)
        })

        store['result'].flush_row()

        print("Done", x, f(x))
        store.close()

    ### Collection reading
    reader = CollectionReader(OUT_DIR)
    print(reader.df('result'))

