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
WAIT = object()

class DirectoryStore(object):
    def __init__(self, path, save_npy, mmap_mode, logger):
        self.store_path = os.path.abspath(path)
        self.save_npy = save_npy
        self.mmap_mode = mmap_mode
        self.logger = logger

    def fetch_or_compute_output(self, task_name_list, input_hash,
                                compute_callback,
                                callback_args=(),
                                callback_kw={},
                                should_block=True):
        """ Call to look up a cached function output.

            This call is a non-trivial in order to support optimistic
            locking.

            Returns
            -------
            The stored or computed data tuple, or, if should_block == False,
            the ``WAIT`` object if somebody else is computing the data.
            
            ## status: string
            ##     Either 'gotoutput', 'gotlock' or 'inprogress'. 'inprogress'
            ##     is only possible if ``blocking == True``.
            ## data: tuple or None
            ##     If ``status == 'gotoutput'``, this is the stored data tuple.
            ##     Otherwise, this is None.
        """
        work_dir = os.path.join(*([self.store_path] + task_name_list))
        output_dir = os.path.join(work_dir, input_hash)
        lockfile = '%s.lock' % output_dir

        if os.path.exists(output_dir):
            # We're in luck, result is already computed
            return self._load_output(output_dir)
        
        try:
            os.makedirs(work_dir)
        except OSError:
            if os.path.exists(work_dir):
                pass # race condition for creating it
            else:
                raise

        # We may have to compute the result. Wrap the computation in a
        # pessimistic lock; we don't rely on this, but it can save us CPU
        # time if it does work.
        with pessimistic_filebased_lock(lockfile, should_block) as got_lock:
            if not got_lock:
                assert not should_block
                return WAIT
            # OK, apparently we got the lock. Is that because we
            # blocked until some other process released it?
            if not os.path.exists(output_dir):
                # No, either we're first or pessimistic locking is not working.
                # Either way we proceed with computation and store it.
                output = compute_callback(*callback_args, **callback_kw)
                self._atomic_persist_output(output_dir, output)
                return output
            else:
                pass # fall through; we want to release lock before loading results

        # We reach this point only if we blocked for another process
        # to compute the results. So now we can load them.
        return self._load_output(output_dir)           

    def _atomic_persist_output(self, path, output):
        # Create the files to a temporary directory in the same location,
        # and then do an atomic directory rename
        parent_path, dirname = os.path.split(path)
        target_path = os.path.join(parent_path, dirname)
        temp_path = tempfile.mkdtemp(prefix='%s-%d-%s-' %
                                     (hostname, pid, dirname))
        try:
            # Dump pickle to temporary directory
            self._persist_output(temp_path, output)
            # Move the temporary directory to the target. Other
            # processes may be doing the same thing...
            try:
                os.rename(temp_path, target_path)
            except OSError:
                if os.path.exists(target_path):
                    self.logger.info('Another process simultaneously computed %s' % target_path)
                else:
                    raise # something else
        finally:
            # Never leave the temporary directory around
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)

    def _persist_output(self, path, output):
        filename = os.path.join(path, 'output.pkl')
        if 'numpy' in sys.modules and self.save_npy:
            numpy_pickle.dump(output, filename) 
        else:
            with file(filename, 'w') as f:
                pickle.dump(output, f, protocol=2)

    def _load_output(self, output_dir):
        filename = os.path.join(output_dir, 'output.pkl')
        if self.save_npy:
            return numpy_pickle.load(filename, 
                                     mmap_mode=self.mmap_mode)
        else:
            with file(filename, 'r') as f:
                return pickle.load(f)

@contextmanager
def pessimistic_filebased_lock(lock_filename, should_block):
    """ A context manager to lock using a filename.

        We use optimistic locking in addition in case this is not
        present; e.g., on Windows we simply pretend we have the lock.
        
    """
    if fcntl_lockf is None:
        yield True
    else:
        try:
            with file(lock_filename, 'a') as fhandle:
                try:
                    # Get lock
                    fcntl_lockf(fhandle, LOCK_EX | (LOCK_NB if not should_block else 0))
                except OSError, e:
                    if not should_block and e.errno in (os.EACCES, os.EAGAIN):
                        # Somebody else has the lock
                        yield False
                    else:
                        raise # don't know what could cause this...
                else:
                    # At this point we have the lock
                    yield True
                    # Release lock (probably redundant as we'll proceed to close the file...
                    fcntl_lockf(fhandle, LOCK_UN)
        finally:
            try:
                os.unlink(lock_filename) # Clean up after ourself
            except OSError:
                pass


## def path_splitall(path):
##     def _split(x):
##         if x == '':
##             return []
##         else:
##             heads, tail = os.path.split(x)
##             return _split(heads) + [tail]    
##     return _split(os.path.normpath(path))

## def racesafe_ensure_path(path):
##     # TODO: Test on Windows etc.
##     if os.path.exists(path):
##         return path

##     pathlist = path_splitall(path)
##     for i in range(1, len(pathlist)):
##         leading = os.path.join(pathlist[:i])
##         try:
##             os.mkdir(leading)
##         except OSError:
##             pass

    
