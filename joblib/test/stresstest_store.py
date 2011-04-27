__doc__ = """
Script to stress-test 
"""

import time
import random
import joblib
from joblib.store import *

def stresstest(path, end_time, seed, result_queue):
    rng = random.Random(seed)
    store = DirectoryStore(path)
    counts = {MUST_COMPUTE: 0, COMPUTED: 0, WAIT: 0}
    while time.time() < end_time:
        item = store.get(['foo', str(rng.randrange(0, 1000))])
        try:
            blocking = rng.uniform(0, 1) < 0.5
            state = item.attempt_compute_lock(blocking=blocking)
            counts[state] += 1
            if state == MUST_COMPUTE:
                item.persist_input(dict(a=2, b=4))
                item.persist_output((1, 2, 3))
                if rng.uniform(0, 1) < 0.9:
                    item.commit()
                else:
                    item.rollback()
            elif state == COMPUTED:
                assert item.fetch_output() == (1, 2, 3)
            elif state == WAIT:
                assert not blocking
                state = item.attempt_compute_lock(blocking=True)
                assert state == COMPUTED, state
                assert item.fetch_output() == (1, 2, 3)
            else:
                assert False
        finally:
            item.close()
    result_queue.put(counts)

if __name__ == '__main__':
    import argparse
    import multiprocessing
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--storepath', type=str, default='_store')
    parser.add_argument('--nprocs', type=int, default=1000)
    parser.add_argument('--walltime', type=float, default=3)
    args = parser.parse_args()
    end_time = time.time() + args.walltime

    q = multiprocessing.Queue()
    processes = [multiprocessing.Process(
                     target=stresstest,
                     args=(args.storepath, end_time, random.randint(0, 2**31), q))
                 for x in range(args.nprocs)]
    for x in processes:
        x.start()
    for x in processes:
        x.join()

    counts = {MUST_COMPUTE: 0, COMPUTED: 0, WAIT: 0}
    while not q.empty():
        x = q.get()
        for key in counts:
            counts[key] += x[key]
    print counts
