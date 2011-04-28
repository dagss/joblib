# Functions that can be used from test script (and are in a package)
from joblib.hpc import versioned

@versioned(ignore_deps=True)
def func1(x, y):
    return x**2 + y

class MyException(Exception):
    pass

@versioned(ignore_deps=True)
def funcex(x, y):
    raise MyException()
