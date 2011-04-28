"""
Futures-style executor targeted for clusters
"""

import os
import socket
import errno
import shutil
import tempfile
import base64
from concurrent.futures import Executor
from textwrap import dedent

from ..func_inspect import filter_args
from ..hashing import NumpyHasher
from .. import numpy_pickle

__all__ = ['ClusterExecutor', 'SlurmExecutor']

# Process invariants
hostname = socket.gethostname()
pid = os.getpid()

def execute_job(path):
    pass

class ClusterFuture(object):
    """
    Note that we do not currently inherit from Future,
    since it contains a certain amount of implementation
    details geared towards thread-safety that we do not
    currently worry about. This should probably change.
    """
    def resubmit(self):
        raise NotImplementedError()

class ClusterExecutor(object):
    """
    Options for targeting a specific cluster/location are provided
    through subclassing (system, address, etc.), while options that
    affect a single run (number of nodes, queue account, etc.) are
    provided in constructor arguments.
    """
    configuration_keys = ()
    
    def __init__(self, **kw):
        for key in self.__class__.configuration_keys:
            value = kw.get(key, None)
            if value is None:
                value = getattr(self.__class__, 'default_%s' % key, None)
            if value is None:
                raise TypeError('Argument %s not provided' % key)
            setattr(self, key, value)
    
    def submit(self, func, *args, **kwargs):
        if not hasattr(func, 'version_info'):
            raise ValueError('func does not have @versioned decorator')

        # Compute hashes to find target job path
        args_dict = filter_args(func, func.version_info['ignore_args'],
                                *args, **kwargs)
        
        future = self._create_future(func, args, kwargs, args_dict)
        return future

    def _create_future(self, target_path):
        raise NotImplementedError()


class DirectoryExecutor(ClusterExecutor):
    configuration_keys = ClusterExecutor.configuration_keys + (
        'store_path',)
    default_store_path = os.path.realpath(os.environ.get('JOBSTORE', None))

    def _create_future(self, func, args, kwargs, filtered_args_dict):
        if not func.version_info['ignore_deps']:
            # TODO: Check a map of possible dependencies executed for
            # this run here, in case a depdency just got changed. This
            # should be done by consulting a pickled on-file database
            # AFAICT.
            raise NotImplementedError()

        # Make job_path containing hashes
        h = NumpyHasher('sha1')
        h.hash(filtered_args_dict)
        args_hash = base64.b32encode(h._hash.digest()).lower()
        func_hash = '%s-%s' % (func.__name__,
                               base64.b32encode(func.version_info['digest']).lower())
        job_name = os.path.join(func_hash, args_hash)
        # Construct job dir if not existing. Remember that we may
        # race for this; if we got to create the directory, we have
        # the lock.
        self._ensure_job_dir(job_name, func, args, kwargs)
        return self._create_future_from_job_dir(job_name)

    def _ensure_job_dir(self, job_name, func, args, kwargs):
        """
        Returns
        -------

        Bool ``existed`` indicating if the job already existed in
        the file system.
        """
        jobpath = os.path.join(self.store_path, job_name)
        if os.path.exists(jobpath):
            return True
        parentpath = os.path.dirname(jobpath)
        ensuredir(parentpath)
        # Guard against crashes & races: Pickle input to temporary
        # directory, then do an atomic rename.
        workpath = tempfile.mkdtemp(prefix='%s-%d-%s-' %
                                    (hostname, pid, os.path.basename(job_name)),
                                    dir=parentpath)
        try:
            # Dump call to file
            call_info = (func, args, kwargs)
            numpy_pickle.dump(call_info, os.path.join(workpath, 'input.pkl'))

            # Create job script
            self._create_jobscript(func.__name__, job_name, workpath)

            # Commit: rename directory
            existed = False
            try:
                os.rename(workpath, jobpath)
            except OSError, e:
                if e.errno != errno.EEXIST:
                    raise
                else:
                    # There was a race; that's fine
                    existed = True
        except:
            shutil.rmtree(workpath) # rollback
            raise

        return existed

class DirectoryFuture(ClusterFuture):
    """
    Cluster job based on preserving state in a directory in a local
    file system. This is an abstract class meant for subclassing, in
    particular it needs an implementation of ``_is_job_running`` that
    can query the system for whether the job has died or not.
    """
    # TODO: Make store_path a dictionary of host_patterns -> paths,
    # and ensure that unpickling this object on a different
    # host changes self.path accordingly.
    
    def __init__(self, store_path, job_path):
        self.path = os.path.realpath(os.path.join(store_path, job_path))
        self._constructed = os.path.exists(self.path)


class SlurmExecutor(DirectoryExecutor):
    configuration_keys = DirectoryExecutor.configuration_keys + (
        'account', 'nodes', 'mem_per_cpu', 'time',
        'tmp', 'ntasks_per_node', 'omp_num_threads')
    default_nodes = 1
    default_mem_per_cpu = '2000M'
    default_time = '01:00:00' # max run time
    default_tmp = '100M' # scratch space
    default_omp_num_threads = '$SLURM_NTASKS_PER_NODE'
    default_ntasks_per_node = 1

    # Configurable by base classes for specific clusters
    pre_command = ''
    post_command = ''
    python_command = 'python'
    
    def get_launch_command(self, fullpath):
        return dedent("""\
        {python} <<END
        from joblib.hpc.executor import execute_job
        execute_job(\"{fullpath}\")
        END
        """.format(python=self.python_command,
                   fullpath=fullpath))

    def _create_jobscript(self, human_name, job_name, work_path):
        jobscriptpath = os.path.join(work_path, 'sbatchscript')
        script = make_slurm_script(
            jobname=human_name,
            command=self.get_launch_command(os.path.join(self.store_path, job_name)),
            logfile=os.path.join(self.store_path, job_name, 'log'),
            precmd=self.pre_command,
            postcmd=self.post_command,
            queue=self.account,
            ntasks=self.ntasks_per_node,
            nodes=self.nodes,
            openmp=self.omp_num_threads,
            time=self.time)
        with file(jobscriptpath, 'w') as f:
            f.write(script)

    def _create_future_from_job_dir(self, job_path):
        return SlurmFuture(self, job_path)

    
class SlurmFuture(DirectoryFuture):
    def __init__(self, config, target_path):
        self.config = config
        self.target_path = target_path
        
        
def make_slurm_script(jobname, command, logfile, where=None, ntasks=1, nodes=None,
                      openmp=1, time='24:00:00', constraints=(),
                      queue='astro', precmd='', postcmd='',
                      mpi=None):
    settings = []
    for c in constraints:
        settings.append('--constraint=%s' % c)
    if nodes is not None:
        if ntasks % nodes != 0:
            raise ValueError('ntasks=%d not divisible by nnodes=%d' % (ntasks, nodes))
        settings.append('--ntasks-per-node=%d' % (ntasks // nodes))
        settings.append('--nodes=%d' % nodes)
        if mpi is None:
            mpi = True
    else:
        settings.append('--ntasks=%d' % ntasks)
        if mpi is None:
            mpi = ntasks > 1
    lst = '\n'.join(['#SBATCH %s' % x for x in settings])
    if where is None:
        where = '$HOME'
    mpicmd = 'mpirun ' if mpi else ''
    
    template = dedent("""\
        #!/bin/bash
        #SBATCH --job-name={jobname}
        #SBATCH --account={queue}
        #SBATCH --time={time}
        #SBATCH --mem-per-cpu=2000
        #SBATCH --tmp=100M
        #SBATCH --output={logfile}
        {lst}
        
        source /site/bin/jobsetup
        export OMP_NUM_THREADS={openmp}
        source $HOME/masterenv

        cd {where}
        {precmd}
        {mpicmd}{command}
        {postcmd}
        """)
    return template.format(**locals())

def ensuredir(path):
    try:
        os.makedirs(path)
    except OSError, e:
        if e.errno != errno.EEXIST:
            raise
