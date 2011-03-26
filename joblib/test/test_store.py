# Author: Dag Sverre Selejbotn <d.s.seljebotn@astro.uio.no>
# Copyright (c) 2011 Dag Sverre Seljebotn
# License: BSD Style, 3 clauses.

import tempfile
import shutil
import os
from nose.tools import ok_, eq_
from nose import SkipTest


from .. import store
from ..store import COMPUTED, MUST_COMPUTE, WAIT

#
# Test fixture
#
class MockLogger(object):
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(msg)

tempdir = store_instance = mock_logger = None

def setup_store():
    global tempdir, store_instance, mock_logger
    tempdir = tempfile.mkdtemp()
#    mock_logger = MockLogger()
    store_instance = store.DirectoryStore(tempdir,
                                          save_npy=True,
                                          mmap_mode=None)
#                                          logger=mock_logger)
                                          
def teardown_store():
    global tempdir, store_instance, mock_logger
    shutil.rmtree(tempdir)
    tempdir = mock_logger = store_instance = None
    
PATH = ['foo', 'bar', '1234ae']


import functools

def with_store(func):
    @functools.wraps(func)
    def inner():
        setup_store()
        try:
            for x in func():
                yield x
        finally:
            teardown_store()
    return inner

#
# Tests
#

@with_store
def test_basic():
    td = store_instance.get(PATH)
    yield eq_, td.is_computed(), False
    yield eq_, td.attempt_compute_lock(blocking=False), MUST_COMPUTE
    td.persist_output((1, 2, 3))
    td.rollback()

    yield eq_, td.is_computed(), False
    yield eq_, td.attempt_compute_lock(blocking=False), MUST_COMPUTE
    td.persist_output((1, 2, 3))
    td.commit()
    yield eq_, td.attempt_compute_lock(blocking=False), COMPUTED    
    yield eq_, td.is_computed(), True
    yield eq_, td.fetch_output(), (1, 2, 3)

    td.close()
    
    td = store_instance.get(PATH)
    yield eq_, td.is_computed(), True
    yield eq_, td.fetch_output(), (1, 2, 3)
    yield eq_, td.attempt_compute_lock(blocking=False), COMPUTED    
    td.close()

    # Check contents of dir
    yield eq_, os.listdir(os.path.join(*([tempdir] + PATH))), ['output.pkl']


def do_pessimistic_lock_tests():
    # Simple checks. Must also be stress-tested for race conditions,
    # this is not sufficient.
    a = store_instance.get(PATH)
    b = store_instance.get(PATH)
    yield eq_, a.attempt_compute_lock(blocking=False), MUST_COMPUTE
    yield eq_, b.attempt_compute_lock(blocking=False), WAIT
    a.persist_output((1, 2, 3))
    a.rollback()
    yield eq_, b.is_computed(), False
    yield eq_, b.attempt_compute_lock(blocking=False), MUST_COMPUTE
    b.persist_output((1, 2, 3))
    yield eq_, a.attempt_compute_lock(blocking=False), WAIT
    b.commit()
    yield eq_, a.attempt_compute_lock(blocking=False), COMPUTED
    

@with_store
def test_thread_locking():
    old_lockf = store.fcntl_lockf
    try:
        # Temporarily disable the lockf mechanism, make
        # sure we rely on threading only. Probably not necessarry...
        store.fcntl_lockf = None
        for x in do_pessimistic_lock_tests():
            yield x
        # TODO: Construct a thread that actually waits.
    finally:
        store.fcntl_lockf = old_lockf

@with_store
def test_fcntl_locking():
    try:
        import fcntl
    except ImportError:
        raise SkipTest("Currently only on POSIX platforms")
#    do_pessimistic_lock_tests()

@with_store
def test_optimistic_locking():
    # Temporarily disable any lockf present
    old_fcntl_lockf = store.fcntl_lockf
    try:
        store.fcntl_lockf = None
        for x in _do_optimistic_locking(): yield x
    finally:
        fcntl_lockf = old_fcntl_lockf

def _do_optimistic_locking():
    # Simulate concurrent access by doing a nested call from
    # within the callback -- the store can't tell the difference
    nested_checks = []

    def do_fail():
        nested_checks.append((ok_, False, "Should not try to recompute"))

    def compute_inner():
        # Actually return something different for the same hash (!!)
        return ("compute_inner",)

    def compute_outer():
        # The store thinks we're busy computing -- now,
        # hit it again for the same resource.
        r = store_instance.fetch_or_compute_output(NAMELST, HASH, compute_inner)
        nested_checks.append((eq_, r, ("compute_inner",)))
        return ("compute_outer",)

    r = store_instance.fetch_or_compute_output(NAMELST, HASH, compute_outer)
    # At this point, the return value of compute_outer is passed through.
    yield eq_, r, ("compute_outer",)    

    # ...BUT, what is actually persisted is the result of
    # compute_inner which got there first.
    r = store_instance.fetch_or_compute_output(NAMELST, HASH, do_fail)
    yield eq_, r, ("computer_inner",)

    for x in nested_checks: yield x    
