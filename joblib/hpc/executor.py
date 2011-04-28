"""
Futures-style executor targeted for clusters
"""

import os
from concurrent.futures import Executor

from ..func_inspect import filter_args
from ..hashing import hash
from .. import numpy_pickle

__all__ = ['ClusterExecutor', 'SlurmExecutor']

def execute_job(path):
    pass

class ClusterExecutor(Executor):
    """
    Options for targeting a specific cluster/location are provided
    through subclassing (system, address, etc.), while options that
    affect a single run (number of nodes, queue account, etc.) are
    provided in constructor arguments.
    """
    configuration_keys = ('task_path',)
    default_task_path = os.path.realpath(os.environ.get('TASKPATH', None))
    
    def __init__(self, **kw):
        for key in self.__class__.configuration_keys:
            value = kw.get(key, None)
            if value is None:
                value = getattr(self.__class__, 'default_%s' % key, None)
            if value is None:
                raise TypeError('Argument %s not provided' % key)
            setattr(self, key, value)
        if self.local_task_path is None:
            self.local_task_path = os.environ[]
    
    def submit(self, func, *args, **kwargs):
        if not hasattr(func, 'version_info'):
            raise ValueError('func does not have @versioned decorator')

        # Compute hashes to find target task path
        args_dict = filter_args(func, func.version_info['ignore_args'],
                                *args, **kwargs)
        args_hash = hash(args_dict, hash_name='sha1')
        func_hash = '%s-%s' % (func.__name__, func.version_info['hexdigest'])
        target_path = os.path.join(func_hash, args_hash)
        local_target_path = os.path.join(self.local_task_path, target_path)
        
        # If it already exists, we're fine
        if not func.version_info['ignore_deps']:
            raise NotImplementedError()
        if os.path.exists(local_target_path):
            # TODO: Check a map of possible dependencies executed for
            # this run here, in case a depdency just got changed. This
            # should be done by consulting a pickled on-file database
            # AFAICT.
            output = numpy_pickle.load(os.path.join(local_target_path, 'output.pkl'))
        else:
            self._persist_call(func, args, kwargs, args_dict)
            self._submit_job(target_path)

        pass
        

    def _persist_call(self, target_path, func, args, kwargs, filtered_args_dict):
        input_repr = dict((k, repr(v)) for k, v in filtered_args_dict.iteritems())
        
        

class SlurmExecutor(ClusterExecutor):
    configuration_keys = ClusterExecutor.configuration_keys + (
        'account', 'nodes', 'mem_per_cpu', 'time',
        'tmp', 'ntasks_per_node', 'omp_num_threads')
    default_nodes = 1
    default_mem_per_cpu = '2000M'
    default_time = '01:00:00' # max run time
    default_tmp = '100M' # scratch space
    default_omp_num_threads = '$SLURM_NTASKS_PER_NODE'
    
    
