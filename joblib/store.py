# Author: Dag Sverre Selejbotn <d.s.seljebotn@astro.uio.no>
# Copyright (c) 2011 Dag Sverre Seljebotn
# License: BSD Style, 3 clauses.

"""
A default file-based data store for Memory.

Locking
-------

Two forms of locking are both used:

 - Pessimistic: On Unix platforms, fcntl locking will be used, which
   supports [...]

 - Optimistic: Even if the pessimistic locking [...]


File layout
-----------

Each computable result is stored in $STOREDIR/$functionname/$hash

 

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
except ImportError:
    fcntl_lockf = None
try:
    import cPickle as pickle
except ImportError:
    import pickle
from contextlib import contextmanager
import tempfile

# Relative imports
from . import numpy_pickle

# Process invariants
hostname = gethostname()
pid = os.getpid()

# Flag object
COMPUTED, MUST_COMPUTE, WAIT = object(), object(), object()

class IllegalOperationError(Exception):
    pass


class DirectoryStore(object):
    def __init__(self, path, save_npy, mmap_mode):
        self.store_path = os.path.abspath(path)
        self.save_npy = save_npy
        self.mmap_mode = mmap_mode

    def get(self, path):
        if not isinstance(path, list):
            raise TypeError("path must be a list of strings")
        return TaskData(os.path.join(*([self.store_path] + path)),
                        self.save_npy,
                        self.mmap_mode)
                        
class TaskData(object):
    """
    *always* call close on one of these
    """
    def __init__(self, task_path, save_npy, mmap_mode):
        self.task_path = os.path.realpath(task_path)
        self.save_npy = save_npy
        self.mmap_mode = mmap_mode
        self._lockfilename = '%s.lock' % self.task_path
        self._lockfile = None
        self._temp_path = None

    def __del__(self):
        if self._lockfile is not None or self._temp_path is not None:
            import warnings
            warnings.warn('Did not commit() or rollback(), fix your code!')

    def _check_not_closed(self):
        if self._status == CLOSED:
            raise IllegalOperationError('Resources is closed')

    def persist_input(self, args, kw):
        raise NotImplementedError()

    def persist_output(self, output):
        filename = os.path.join(self._temp_path, 'output.pkl')
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

    def status(self):
        return self._status

    def is_computed(self):
        return os.path.exists(self.task_path)

    def attempt_compute_lock(self, blocking=True):
        """
        Call this to simultaneously probe whether a task is computed
        and offer to compute it if not.

        Returns either of COMPUTED, MUST_COMPUTE, or WAIT. If
        it returns MUST_COMPUTE then you *must* call commit() or
        rollback(). If 'blocking' is True, then WAIT can not be
        computed.
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
        self._temp_path = None
        try:
            # Try to acquire pessimistic lock. We don't rely on this, but
            # it can save us CPU time if it does work.
            if fcntl_lockf is not None:
                self._lockfile = file(self._lockfilename, 'a')
                try:
                    # Get lock
                    fcntl_lockf(self._lockfile, LOCK_EX | (LOCK_NB if not blocking else 0))
                except OSError, e:
                    if not blocking and e.errno in (os.EACCES, os.EAGAIN):
                        # Somebody else has the lock
                        self.rollback()
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
            self._temp_path = tempfile.mkdtemp(prefix='%s-%d-%s-' %
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
                os.rename(self._temp_path, self.task_path)
                self._temp_path = None
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
            except:
                pass
            try:
                self._lockfile.close()
            except:
                pass
            try:
                os.unlink(self._lockfilename)
            except OSError:
                pass
            self._lockfile = None

        if self._temp_path is not None and os.path.exists(self._temp_path):
            try:
                shutil.rmtree(self._temp_path)
            except:
                pass
            self._temp_path = None

    def close(self):
        self.rollback()
        self.task_path = None
