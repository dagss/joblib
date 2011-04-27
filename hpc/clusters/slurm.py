from concurrent.futures import Executor

class SlurmExecutor(Executor):
    """
    Configured for a cluster/location through subclassing,
    configured for individual uses through constructor arguments.
    """
    setting_keys = ('account', 'nodes', 'mem_per_cpu', 'time',
                    'tmp', 'ntasks_per_node', 'omp_num_threads')

    default_nodes = 1
    default_mem_per_cpu = '2000M'
    default_time = '01:00:00' # max run time
    default_tmp = '100M' # scratch space
    default_omp_num_threads = '$SLURM_NTASKS_PER_NODE'
    
    def __init__(self, **kw):
        for key in setting_keys:
            value = kw.get(key, None)
            if value is None:
                value = getattr(self.__class__, 'default_%s' % key, None)
            if value is None:
                raise TypeError('Argument %s not provided' % key)
            setattr(self, key, value)

    
    def submit(self, func, *args, **kwargs):
        
