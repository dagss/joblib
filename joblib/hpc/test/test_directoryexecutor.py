"""Test DirectoryExecutor
"""

from concurrent.futures import ProcessPoolExecutor
import logging
import tempfile
import shutil
import os
import contextlib
from os.path import join as pjoin
from nose.tools import ok_, eq_

from ..executor import DirectoryExecutor, DirectoryFuture, execute_directory_job
from .. import versioned
from ...numpy_pickle import load, dump


# Debug settings
KEEPSTORE = False

#
# Create mock executors that simply forwards to ProcessPoolExecutor
# HOWEVER, communication doesn't happen the normal way, but
# rather through the disk-pickles of DirectoryExecutor. The
# only direct IPC is the path to the job directory as a string.
#

class MockExecutor(DirectoryExecutor):
    configuration_keys = DirectoryExecutor.configuration_keys + (
        'before_submit_hook',)
    
    def __init__(self, *args, **kw):
        DirectoryExecutor.__init__(self, *args, **kw)
        self.subexecutor = ProcessPoolExecutor()
        self.given_work_paths = []
        self.submit_count = 0

    def _create_future_from_job_dir(self, job_name):
        return MockFuture(self, job_name)

    def _create_jobscript(self, human_name, job_name, work_path):
        self.given_work_paths.append(work_path)
        with file(pjoin(work_path, 'jobscript'), 'w') as f:
            f.write('jobscript for job %s\n' % human_name)


class MockFuture(DirectoryFuture):
    def _submit(self):
        self._executor.before_submit_hook(self)
        self._executor.submit_count += 1
        self._executor.logger.debug('Submitting job')
        procex = self._executor.subexecutor
        assert isinstance(self.job_path, str)
        self.subfuture = procex.submit(execute_directory_job, self.job_path)
        return 'job-%d' % (self._executor.submit_count - 1)

#
# Test utils
#

@contextlib.contextmanager
def working_directory(path):
    old = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old)

def ls(path='.'):
    return set(os.listdir(path))

def ne_(a, b):
    assert a != b, "%r == %r" % (a, b)

def filecontents(filename):
    with file(filename) as f:
        return f.read()

#
# Test context
#
def setup_module():
    global logger, store_path

    store_path = tempfile.mkdtemp(prefix='jobstore-')
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info('JOBSTORE=%s' % store_path)

def teardown_module():
    if KEEPSTORE:
        shutil.rmtree(store_path)


@versioned(1, ignore_deps=True)
def f(x, y):
    return x + y

def test_basic():
    tests = []
    PRE_SUBMIT_LS = set(['input.pkl', 'jobscript'])
    COMPUTED_LS = PRE_SUBMIT_LS.union(['output.pkl', 'jobid', 'log'])
    
    def before_submit(fut):
        with working_directory(fut.job_path):
            tests.append((eq_, ls(), PRE_SUBMIT_LS))
            input = load('input.pkl')
            tests.append((eq_, input, dict(
                args=(1, 1),
                func=f,
                kwargs={})))
            tests.append((eq_, 'jobscript for job f\n',
                          filecontents('jobscript')))
    
    executor = MockExecutor(store_path=store_path,
                            logger=logger,
                            poll_interval=1e-1,
                            before_submit_hook=before_submit)

    # Run a single job, check that it executes, and check input/output
    fut = executor.submit(f, 1, 1)    
    yield eq_, executor.submit_count, 1
    yield eq_, fut.result(), 2
    yield ne_, executor.given_work_paths[0], fut.job_path
    with working_directory(fut.job_path):
        output = load('output.pkl')
        yield eq_, output, ('finished', 2)
        yield eq_, ls(), COMPUTED_LS
        yield eq_, len(executor.given_work_paths), 1
        yield eq_, filecontents('jobid'), 'job-0\n'

    # Re-run and check that result is loaded from cache
    fut = executor.submit(f, 1, 1)
    yield eq_, fut.result(), 2
    yield eq_, executor.submit_count, 1

    # Run yet again with different input
    executor.before_submit_hook = lambda x: None
    fut2 = executor.submit(f, 1, 2)
    yield eq_, fut2.result(), 3
    yield eq_, executor.submit_count, 2
    yield ne_, fut2.job_path, fut.job_path

    # Run tests queued by closures
    yield eq_, len(tests), 3
    for x in tests:
        yield x
    


    
