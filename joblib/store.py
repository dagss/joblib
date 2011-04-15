# Author: Dag Sverre Selejbotn <d.s.seljebotn@astro.uio.no>
# Copyright (c) 2011 Dag Sverre Seljebotn
# License: BSD Style, 3 clauses.

"""
A default file-based data store for Memory.

Design
------

The DirectoryStore is reentrant and very simple (multiple threads or
processes can use the same instance). The get() method returns a
TaskData instance, which should not be shared across threads or
processes. Example usage::

    mystore = DirectoryStore('/tmp/joblib')
    res = mystore.get(['mymod', 'myfunc', 'a3242bdef323423'])
    # Now, *always* call attempt_compute_lock() or is_computed() first
    status = res.attempt_compute_lock(blocking=False)
    # status can now be MUST_COMPUTE, COMPUTED or WAIT. Assume MUST_COMPUTE:
    res.persist_input({'a' : 1})
    res.persist_output((1, 2, 3))
    res.commit()
    res.close() # put this in a finally block!!


Locking
-------

Three forms of locking are used. The two first are pessimistic locks;
they are present so that processes/threads can go do something else
(or go to sleep) if somebody else is already working on the same
task. Even if these two mechanisms both fail, it will simply mean more
CPU time wasted, because the last mechanism will ensure correctness.

 - Thread locking: Largely because Unix doesn't support file locking
   in-process, the first lock mechanism used is in-process thread
   locks.  The global thread lock is very transitory and should not be
   a problem, considering the GIL. All waiting happen per-task.

 - File locking: On Unix platforms, fcntl.lockf will be used, which
   has support on single-node as well as on some network
   filesystems. Importantly, it is safe against dead-locks. Windows
   locking is still unimplemented.

 - Optimistic: All output is generated to a temporary directory, and
   on commit() the temporary directory is renamed (which is an atomic
   operation on OSes I checked). If such a rename fails, it will be
   assumed it is due to a race condition and the computed results
   are simply dropped. 

"""

# System imports
import os
import sys
import shutil
from socket import gethostname
try:
    # Import fcntl, only available on Unix.
    #
    # Note: Test suite will temporarily overwrite this with its
    # own stub function to simulate a broken lockf, or None
    # to simulate non-Unix
    #
    # Note: "flock" is completely different from "lockf". Only the
    # latter works across (some) network filesystems. Do not mix the
    # two.
    from fcntl import lockf as fcntl_lockf, LOCK_EX, LOCK_SH, LOCK_UN, LOCK_NB
    from errno import EACCES, EAGAIN    
except ImportError:
    fcntl_lockf = None
try:
    import cPickle as pickle
except ImportError:
    import pickle
from contextlib import contextmanager
import tempfile
import threading
try:
    # json is in the standard library for Python >= 2.6
    import json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        # Not the end of the world: we'll do without this functionality
        json = None


# Relative imports
from . import numpy_pickle

# Process invariants
hostname = gethostname()
pid = os.getpid()

# Enums. Feature is simply that they compare by id()
class Enum(object):
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return '<%s>' % self.name
    
COMPUTED = Enum('COMPUTED')
MUST_COMPUTE = Enum('MUST_COMPUTE')
WAIT = Enum('WAIT')

#
# In-process synchronization
#
_inproc_events_lock = threading.Lock()
_inproc_events = {}

class IllegalOperationError(Exception):
    pass

class DirectoryStore(object):
    def __init__(self,
                 path,
                 save_npy=True, 
                 mmap_mode=None,
                 use_thread_locks=True,
                 use_file_locks=True):
        self.store_path = os.path.abspath(path)
        self.save_npy = save_npy
        self.mmap_mode = mmap_mode
        self._use_thread_locks = use_thread_locks
        self._use_file_locks = use_file_locks

    def get(self, path):
        if not isinstance(path, list):
            raise TypeError("path must be a list of strings")
        return TaskData(os.path.join(*([self.store_path] + path)),
                        self.save_npy,
                        self.mmap_mode,
                        self._use_thread_locks,
                        self._use_file_locks
                        )

class TaskData(object):
    """
    *always* call close on one of these
    """
    def __init__(self, task_path, save_npy, mmap_mode,
                 use_thread_locks=True, use_file_locks=True):
        self.task_path = os.path.realpath(task_path)
        self.save_npy = save_npy
        self.mmap_mode = mmap_mode
        self._lockfilename = '%s.lock' % self.task_path
        self._lockfile = None
        self._work_path = None
        self._done_event = None
        self._use_thread_locks = use_thread_locks
        self._use_file_locks = use_file_locks

    def __del__(self):
        if self._lockfile is not None or self._work_path is not None:
            import warnings
            warnings.warn('Did not commit() or rollback(), fix your code!')

    def persist_input(self, input_repr):
        if self._work_path is None:
            raise IllegalOperationError("call attemt_compute_lock first")
        if json is not None:
            with file(os.path.join(self._work_path, 'input_args.json'), 'w') as f:
                json.dump(input_repr, f)

    def persist_output(self, output):
        if self._work_path is None:
            raise IllegalOperationError("call attemt_compute_lock first")
        filename = os.path.join(self._work_path, 'output.pkl')
        if 'numpy' in sys.modules and self.save_npy:
            numpy_pickle.dump(output, filename) 
        else:
            with file(filename, 'w') as f:
                pickle.dump(output, f, protocol=2)

    def fetch_output(self):
        if not self.is_computed():
            raise IllegalOperationError('output not stored')
        filename = os.path.join(self.task_path, 'output.pkl')
        if self.save_npy:
            return numpy_pickle.load(filename, 
                                     mmap_mode=self.mmap_mode)
        else:
            with file(filename, 'r') as f:
                return pickle.load(f)

    def is_computed(self):
        return os.path.exists(self.task_path)

    def attempt_compute_lock(self, blocking=True):
        """
        Call this to simultaneously probe whether a task is computed
        and offer to compute it if not.

        Returns either of COMPUTED, MUST_COMPUTE, or WAIT. If
        it returns MUST_COMPUTE then you *must* call commit() or
        rollback(). If 'blocking' is True, then WAIT can not be
        returned.
        """
        # First check if it has already been computed
        if self.is_computed():
            return COMPUTED

        # Nope. At this point we may be first, so ensure parent
        # directory exists.
        parent_path, dirname = os.path.split(self.task_path)
        try:
            os.makedirs(parent_path)
        except OSError:
            if os.path.exists(parent_path):
                pass
            else:
                raise

        self._lockfile = None
        self._work_path = None
        try:
            # Try to detect other threads in the same process doing
            # this; if so we can
            if self._use_thread_locks:
                with _inproc_events_lock:
                    done_event = _inproc_events.get(self.task_path, None)
                    if done_event is None:
                        self._done_event = _inproc_events[self.task_path] = threading.Event()

                if done_event is not None:
                    if not blocking:
                        self.rollback()
                        return WAIT
                    else:
                        done_event.wait()
                        if self.is_computed():
                            return COMPUTED
                        # Else, other thread did a rollback() in the
                        # end. Fall through to case below.
                else:
                    # This is needed to guard against a race condition.
                    # This is so that we keep _inproc_events mostly
                    # empty...
                    if self.is_computed():
                        self.rollback()
                        return COMPUTED

            if self._use_file_locks:
                # Try to acquire pessimistic file lock. We don't rely on this, but
                # it can save us CPU time if it does work
                if fcntl_lockf is not None:
                    self._lockfile = file(self._lockfilename, 'a')
                    try:
                        # Get lock
                        fcntl_lockf(self._lockfile, LOCK_EX | (LOCK_NB if not blocking else 0))
                    except IOError, e:
                        self._lockfile = None
                        self.rollback()
                        if not blocking and e.errno in (EACCES, EAGAIN):
                            # Somebody else has the lock
                            return WAIT
                        else:
                            raise # don't know what could cause this...
                    # OK, apparently we got the lock. Is that because we
                    # blocked until some other process released it, so that
                    # the result is already computed?
                    if self.is_computed():
                        self.rollback()
                        return COMPUTED

            # We're first or pessimistic locking is not working
            # (this includes the case where fcntl_lockf is not None but is
            # broken!). Either way we proceed with computation;
            # commit() may then discover there was concurrent computation anyway.
            self._work_path = tempfile.mkdtemp(prefix='%s-%d-%s-' %
                                               (hostname, pid, dirname),
                                               dir=parent_path)
            return MUST_COMPUTE
        except:
            self.rollback()
            raise

    def commit(self):
        """
        returns False if another process committed in the meantime
        """
        # Move the temporary directory to the target. Other
        # processes may be doing the same thing...
        try:
            try:
                os.rename(self._work_path, self.task_path)
                self._work_path = None
            except OSError:
                if self.is_computed():
                    return False
                else:
                    raise # something else
            else:
                return True
        finally:
            # Release locks, and clean up tempdir if the above move failed
            self.rollback()

    def rollback(self):
        if self._lockfile is not None:
            try:
                fcntl_lockf(self._lockfile, LOCK_UN)
            except (OSError, IOError):
                pass
            try:
                self._lockfile.close()
            except OSError:
                pass
            try:
                os.unlink(self._lockfilename)
            except OSError:
                pass
            self._lockfile = None

        if self._work_path is not None:
            try:
                shutil.rmtree(self._work_path)
            except OSError:
                pass
            self._work_path = None

        if self._done_event is not None:
            with _inproc_events_lock:
                try:
                    del _inproc_events[self.task_path]
                except KeyError:
                    pass
            self._done_event.set()
            self._done_event = None

    def close(self):
        self.rollback()
        self.task_path = None
