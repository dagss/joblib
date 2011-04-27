from nose.tools import eq_, ok_
from .. import versioned

def test_versioned():

    @versioned(4, ignore_deps=1, ignore=['y'])
    def f(x, y):
        return x**2 + y

    @versioned()
    def g(x):
        return x
    
    yield eq_, 2, f(1, 1)
    yield ok_, hasattr(f, 'version_info')
    if hasattr(f, 'version_info'):
        d = dict(f.version_info)
        yield eq_, len(d['digest']), 64
        del d['digest']
        yield eq_, d, dict(version=4, ignore_deps=True, ignore_args=('y',))
    

    yield ok_, hasattr(g, 'version_info')
    if hasattr(g, 'version_info'):
        d = dict(g.version_info)
        yield eq_, len(d['digest']), 64
        del d['digest']
        yield eq_, d, dict(version=None, ignore_deps=False, ignore_args=())
