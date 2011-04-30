__doc__ = """
Script to stress-test 
"""

import time
import random
import joblib
from joblib.store import *

def stresstest(args, end_time, seed, result_queue, callid):
    path = args.storepath
    rng = random.Random(seed)
    store = DirectoryStore(path)
    counts = {MUST_COMPUTE: 0, COMPUTED: 0, WAIT: 0}
    while time.time() < end_time:
        item = store.get(['foo', str(rng.randrange(0, 0.5 * args.nprocs))])
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
    result_queue.put((callid, counts))

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
                     args=(args, end_time, random.randint(0, 2**31), q, callid))
                 for callid in range(args.nprocs)]
    for x in processes:
        x.start()

    # We must simultanously check for killed processes, and
    # reduce the contents put in queue by living processes.
    counts = {MUST_COMPUTE: 0, COMPUTED: 0, WAIT: 0}
    killed = set()
    returned = set()
    while len(killed) + len(returned) < args.nprocs:
        while not q.empty():
            procid, result = q.get()
            returned.add(procid)
            for key in counts:
                counts[key] += result[key]
        for idx, proc in enumerate(processes):
            if idx in returned:
                continue
            if not proc.is_alive():
                killed.add(idx)
        time.sleep(0.1)
            
        
    print counts


    for x in processes:
        x.join()

