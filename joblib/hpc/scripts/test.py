from joblib.hpc import SlurmExecutor, versioned

@versioned(ignore_deps=True)
def func(x, y):
    return x**2 + y

e = SlurmExecutor(account='quiet')

fut = e.submit(func, 2, 3)
print fut
