"""
Futures-style executor targeted for clusters
"""

import os
import socket
import errno
import shutil
import tempfile
import base64
import time
from concurrent.futures import Executor, TimeoutError
from textwrap import dedent
from os.path import join as pjoin

from ..func_inspect import filter_args
from ..hashing import NumpyHasher
from .. import numpy_pickle

__all__ = ['ClusterExecutor', 'SlurmExecutor']

# Process invariants
hostname = socket.gethostname()
pid = os.getpid()

class ClusterFuture(object):
    """
    Note that we do not currently inherit from Future,
    since it contains a certain amount of implementation
    details geared towards thread-safety that we do not
    currently worry about. This should probably change.
    """
    def submit(self):
        raise NotImplementedError()

    def submitted(self):
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
        
        future = self._create_future(func, args, kwargs, args_dict, should_submit=True)
        return future

    def _create_future(self, func, args, kwargs, filtered_args_dict, should_submit):
        raise NotImplementedError()



class DirectoryExecutor(ClusterExecutor):
    configuration_keys = ClusterExecutor.configuration_keys + (
        'store_path', 'poll_interval')
    default_store_path = (os.environ['JOBSTORE']
                          if 'JOBSTORE' in os.environ
                          else None)
    default_poll_interval = 60

    def _encode_digest(self, digest):
        # Use base64 but ensure there's no padding (pad digest up front).
        # Replace / with _ in alphabet.
        while (len(digest) * 8 % 6) != 0:
            digest += '\0'
        return base64.b64encode(digest, '_+')

    def _create_future(self, func, args, kwargs, filtered_args_dict, should_submit):
        if not func.version_info['ignore_deps']:
            # TODO: Check a map of possible dependencies executed for
            # this run here, in case a depdency just got changed. This
            # should be done by consulting a pickled on-file database
            # AFAICT.
            raise NotImplementedError()

        # Make job_path containing hashes
        h = NumpyHasher('sha1')
        h.hash(filtered_args_dict)
        args_hash = self._encode_digest(h._hash.digest())
        func_hash = '%s-%s' % (func.__name__,
                               self._encode_digest(func.version_info['digest']))
        job_name = pjoin(func_hash, args_hash)
        # Construct job dir if not existing. Remember that we may
        # race for this; if we got to create the directory, we have
        # the lock.
        we_made_it = self._ensure_job_dir(job_name, func, args, kwargs)
        future = self._create_future_from_job_dir(job_name)
        if should_submit and we_made_it:
            future.submit()
        return future

    def _ensure_job_dir(self, job_name, func, args, kwargs):
        """
        Returns
        -------

        Bool ``we_made_it`` indicating if the job had to be created;
        if False, somebode else did.
        """
        jobpath = pjoin(self.store_path, job_name)
        if os.path.exists(jobpath):
            return False
        parentpath = os.path.dirname(jobpath)
        ensuredir(parentpath)
        # Guard against crashes & races: Pickle input to temporary
        # directory, then do an atomic rename.
        workpath = tempfile.mkdtemp(prefix='%s-%d-%s-' %
                                    (hostname, pid, os.path.basename(job_name)),
                                    dir=parentpath)
        try:
            # Dump call to file
            call_info = dict(func=func, args=args, kwargs=kwargs)
            numpy_pickle.dump(call_info, pjoin(workpath, 'input.pkl'))

            # Create job script
            self._create_jobscript(func.__name__, job_name, workpath)

            # Commit: rename directory
            try:
                os.rename(workpath, jobpath)
                we_made_it = True
            except OSError, e:
                if e.errno != errno.EEXIST:
                    raise
                else:
                    # There was a race; that's fine
                    we_made_it = False
        except:
            shutil.rmtree(workpath) # rollback
            raise

        return we_made_it

def execute_directory_job(path):
    input = numpy_pickle.load(pjoin(path, 'input.pkl'))
    func, args, kwargs = [input[x] for x in ['func', 'args', 'kwargs']]
    try:
        output = ('finished', func(*args, **kwargs))
    except BaseException, e:
        output = ('exception', e)
    # Do an atomic pickle; if output.pkl is present then it is complete
    fd, workfile = tempfile.mkstemp(prefix='output.pkl-', dir=path)
    try:
        os.close(fd)
        numpy_pickle.dump(output, workfile)
        os.rename(workfile, pjoin(path, 'output.pkl'))
    except:
        if os.path.exists(workfile):
            os.unlink(workfile)
        raise

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
    
    def __init__(self, executor, job_name):
        self.job_name = job_name
        self._executor = executor
        self.job_path = os.path.realpath(pjoin(self._executor.store_path, job_name))

    def cancel(self):
        raise NotImplementedError()
    
    def cancelled(self):
        return False
    
    def running(self):
        pass

    def done(self):
        return self._finished() or self.cancelled()
    
    def result(self, timeout=None):
        sleeptime = self._executor.poll_interval
        if timeout is None:
            sleeptime = min(sleeptime, timeout)
        else:
            endtime = time.time() + timeout            
        while True:
            if self._finished():
                status, output = self._load_output()
                if status == 'exception':
                    raise output
                elif status == 'finished':
                    return output
            if timeout is not None and time.time() >= endtime:
                raise TimeoutError()
            time.sleep(sleeptime)
    
    def exception(timeout=None):
        pass

    def _finished(self):
        return os.path.exists(pjoin(self.job_path, 'output.pkl'))

    def _load_output(self):
        return numpy_pickle.load(pjoin(self.job_path, 'output.pkl'))
        


class SlurmExecutor(DirectoryExecutor):
    configuration_keys = DirectoryExecutor.configuration_keys + (
        'account', 'nodes', 'mem_per_cpu', 'time',
        'tmp', 'ntasks_per_node', 'omp_num_threads',
        'sbatch_command_pattern')
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

    def _slurm(self, scriptfile):
        cmd = "sbatch '%s'" % scriptfile
        if os.system(cmd) != 0:
            raise RuntimeError('command failed: %s' % cmd)
    
    def get_launch_command(self, fullpath):
        return dedent("""\
        {python} <<END
        from joblib.hpc.executor import execute_directory_job
        execute_directory_job(\"{fullpath}\")
        END
        """.format(python=self.python_command,
                   fullpath=fullpath))

    def _create_jobscript(self, human_name, job_name, work_path):
        jobscriptpath = pjoin(work_path, 'sbatchscript')
        script = make_slurm_script(
            jobname=human_name,
            command=self.get_launch_command(pjoin(self.store_path, job_name)),
            logfile=pjoin(self.store_path, job_name, 'log'),
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


import re
from time import strftime

class TitanOsloExecutor(SlurmExecutor):
    default_sbatch_command_pattern = 'ssh titan.uio.no "bash -ic \'sbatch %s\'"'
    _sbatch_re = re.compile(r'Submitted batch job ([0-9]+)')

    def _slurm(self, scriptfile):
        from subprocess import Popen, PIPE
        pp = Popen(['ssh', '-T', 'titan.uio.no'], stdin=PIPE, stderr=PIPE, stdout=PIPE)
        pp.stdin.write("sbatch '%s'" % scriptfile)
        pp.stdin.close()
        #TODO: Could this deadlock when output is larger than buffer size?
        err = pp.stderr.read()
        out = pp.stdout.read()
        pp.stderr.close()
        pp.stdout.close()
        retcode = pp.wait()
        if retcode != 0:
            raise RuntimeError('Return code %d: %s\nError log:\n%s' % (retcode, ' '.join(cmd),
                                                                       err))
        m = self._sbatch_re.search(out)
        if not m:
            raise RuntimeError('Sbatch provided unexpected output: %s' % out)
        return m.group(1)
        
class SlurmFuture(DirectoryFuture):
    def submit(self):
        scriptfile = pjoin(self.job_path, 'sbatchscript')
        jobid = self._executor._slurm(scriptfile)
        with file(pjoin(self.job_path, 'jobid'), 'w') as f:
            f.write(jobid + '\n')
        with file(pjoin(self.job_path, 'log'), 'w') as f:
            f.write('%s submitted job (%s), waiting to start\n' % (strftime('%Y-%m-%d %H:%M:%S'), jobid))
        
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
