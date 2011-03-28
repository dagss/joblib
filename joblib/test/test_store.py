# Author: Dag Sverre Selejbotn <d.s.seljebotn@astro.uio.no>
# Copyright (c) 2011 Dag Sverre Seljebotn
# License: BSD Style, 3 clauses.

import tempfile
import shutil
import os
from nose.tools import ok_, eq_, assert_raises
from nose import SkipTest
import threading
from time import sleep
import functools

from .. import store
from ..store import COMPUTED, MUST_COMPUTE, WAIT, IllegalOperationError

#
# Test fixture
#
class MockLogger(object):
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(msg)

tempdir = store_instance = mock_logger = None

def setup_store(**kw):
    global tempdir, store_instance, mock_logger
    tempdir = tempfile.mkdtemp()
    store_instance = store.DirectoryStore(tempdir,
                                          save_npy=True,
                                          mmap_mode=None,
                                          **kw)
                                          
def teardown_store():
    global tempdir, store_instance, mock_logger
    shutil.rmtree(tempdir)
    tempdir = mock_logger = store_instance = None
    
def with_store(use_thread_locks=True, use_file_locks=True):
    def with_store_dec(func):
        @functools.wraps(func)
        def inner():
            setup_store(use_thread_locks=use_thread_locks, use_file_locks=use_file_locks)
            try:
                for x in func():
                    yield x
            finally:
                teardown_store()
        return inner
    return with_store_dec

#
# Tests
#

PATH = ['foo', 'bar', '1234ae']

@with_store()
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

@with_store()
def test_errors():
    a = store_instance.get(PATH)
    yield assert_raises, IllegalOperationError, a.persist_output, (1, 2, 3)


def do_pessimistic_lock():
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
    

@with_store(use_file_locks=False)
def test_thread_locking_nowait():
    for x in do_pessimistic_lock():
        yield x

@with_store(use_file_locks=False)
def test_thread_locking_wait():
    # Test with blocking
    other_beforesleep = threading.Event()
    other_woke = threading.Event()
    first_thread_done = False
    class OtherThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.tests = []
        def run(self):
            b = store_instance.get(PATH)
            other_beforesleep.set()
            r = b.attempt_compute_lock(blocking=True)
            other_woke.set()
            self.tests.append((eq_, first_thread_done, True))
            self.tests.append((eq_, r, COMPUTED))
            self.tests.append((eq_, b.fetch_output(), (1, 2, 3)))

    a = store_instance.get(PATH)
    yield eq_, a.attempt_compute_lock(blocking=True), MUST_COMPUTE
    otherthread = OtherThread()
    otherthread.start()
    a.persist_output((1, 2, 3))
    first_thread_done = True
    assert other_beforesleep.wait(0.1)
    sleep(0.1)
    a.commit()
    yield eq_, other_woke.wait(0.1), True
    otherthread.join()
    for x in otherthread.tests:
        yield x

@with_store(use_thread_locks=False, use_file_locks=False)
def test_optimistic_locking():
    a = store_instance.get(PATH)
    b = store_instance.get(PATH)
    eq_(a.attempt_compute_lock(blocking=False), MUST_COMPUTE)
    eq_(b.attempt_compute_lock(blocking=False), MUST_COMPUTE)
    a.persist_output((1, 2, 3))
    b.persist_output((4, 5, 6))
    yield eq_, b.commit(), True
    yield eq_, a.commit(), False
    yield eq_, a.fetch_output(), (4, 5, 6)
    yield eq_, b.fetch_output(), (4, 5, 6)
