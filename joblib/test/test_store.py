# Author: Dag Sverre Selejbotn <d.s.seljebotn@astro.uio.no>
# Copyright (c) 2011 Dag Sverre Seljebotn
# License: BSD Style, 3 clauses.

import tempfile
import shutil
import os
from nose.tools import ok_, eq_, assert_raises
from nose import SkipTest
import threading
import multiprocessing
from time import sleep
import functools
from functools import partial
try:
    import queue
except ImportError:
    import Queue as queue

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

#
# Test cases reused for in-process and cross-process locking.
# The core tests are run twice, once with threading and once
# with multiprocessing.
#
def pessimistic_lock_otherthread(store_instance,
                                 beforesleep_event,
                                 aftersleep_event):
    b = store_instance.get(PATH)
    beforesleep_event.set()
    eq_(b.attempt_compute_lock(blocking=False), WAIT)
    eq_(b.is_computed(), False)
    r = b.attempt_compute_lock(blocking=True)
    aftersleep_event.set()
    eq_(r, COMPUTED)
    eq_(b.fetch_output(), (1, 2, 3))

def do_pessimistic_lock(spawn, Event):
    # Test with blocking
    beforesleep_event = Event()
    aftersleep_event = Event()

    a = store_instance.get(PATH)
    yield eq_, a.attempt_compute_lock(blocking=True), MUST_COMPUTE

    p = spawn(target=pessimistic_lock_otherthread,
              args=(store_instance,
                    beforesleep_event,
                    aftersleep_event))
    p.start()
    a.persist_output((1, 2, 3))
    assert beforesleep_event.wait(0.1)
    sleep(0.1)
    yield eq_, aftersleep_event.is_set(), False
    a.commit()
    yield eq_, aftersleep_event.wait(0.1), True

@with_store(use_file_locks=False)
def test_thread_locking():
    for x in do_pessimistic_lock(threading.Thread,
                                 threading.Event):
        yield x

@with_store(use_thread_locks=False)
def test_file_locking():
    try:
        import fcntl
    except ImportError:
        raise SkipTest("fcntl not available")
    for x in do_pessimistic_lock(multiprocessing.Process,
                                 multiprocessing.Event):
        yield x
