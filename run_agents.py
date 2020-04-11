from multiprocessing import Process, JoinableQueue
import sys
from glob import glob
from os import path
from run import main
import json

agent_configs = sys.argv[1]
q = JoinableQueue()
NUM_THREADS = 24

def run_single_config(queue):
    while True:
        conf_path = queue.get()
        params = json.load(open(conf_path))
        try:
            main(params)
        except Exception as e:
            print("ERROR", e)
            raise e
        queue.task_done()

for i in range(NUM_THREADS):
    worker = Process(target=run_single_config, args=(q,))
    worker.daemon = True
    worker.start()

for fname in glob(path.join(agent_configs, "*.json")):
    q.put(fname)

q.join()
