# Author: Dag Sverre Selejbotn <d.s.seljebotn@astro.uio.no>
# Copyright (c) 2011 Dag Sverre Seljebotn
# License: BSD Style, 3 clauses.

import tempfile
import shutil
import os
from nose.tools import ok_, eq_, with_setup
from nose import SkipTest

from .. import store

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
    mock_logger = MockLogger()
    store_instance = store.DirectoryStore(tempdir,
                                          save_npy=True,
                                          mmap_mode=None,
                                          logger=mock_logger)
                                          
def teardown_store():
    global tempdir, store_instance, mock_logger
    shutil.rmtree(tempdir)
    tempdir = mock_logger = store_instance = None
    
NAMELST = ['foo', 'bar']
HASH = '1234ae'

with_store = with_setup(setup_store, teardown_store)

#
# Tests
#

@with_setup(setup_store, teardown_store)
def test_basic():

    timescalled = [0]
    
    def compute():
        timescalled[0] += 1
        return (1, 2, 3)

    # First time we compute
    r = store_instance.fetch_or_compute_output(
        ['foo', 'bar'], HASH, compute)
    yield eq_, r, (1, 2, 3)
    yield eq_, timescalled[0], 1

    # Simply fetch from cache
    r = store_instance.fetch_or_compute_output(
        ['foo', 'bar'], HASH, compute)
    yield eq_, r, (1, 2, 3)
    yield eq_, timescalled[0], 1

    # Check contents of dir
    yield eq_, os.listdir(os.path.join(*([tempdir] + NAMELST + [HASH]))), ['output.pkl']

@with_store
def test_pessimistic_locking():
    try:
        import fcntl
    except ImportError:
        raise SkipTest("Currently only on POSIX platforms")

    # We simulate concurrent access by doing a nested call from
    # within the callback -- the store can't tell the difference

    nested_checks = []

    def do_fail():
        nested_checks.append((ok_, False, "Should not try to recompute"))

    def compute():
        # The store thinks we're busy computing -- now,
        # hit it with a nonblocking call for the same resource
        r = store_instance.fetch_or_compute_output(NAMELST, HASH, do_fail,
                                                   should_block=False)
        nested_checks.append((eq_, r, store.WAIT, "Should report wait status"))
        return (1, 2, 3)

    r = store_instance.fetch_or_compute_output(NAMELST, HASH, compute)
    yield eq_, r, (1, 2, 3)
    for x in nested_checks: yield x

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
