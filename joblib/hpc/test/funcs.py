# Functions that can be used from test script (and are in a package)
from joblib.hpc import versioned

@versioned(deps=False)
def func1(x, y):
    return x**2 + y

class MyException(Exception):
    pass

@versioned(deps=False)
def funcex(x, y):
    raise MyException()
