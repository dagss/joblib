from joblib.hpc import versioned
from joblib.hpc.executor import TitanOsloExecutor, SlurmExecutor

from joblib.hpc.test.funcs import func1

e = TitanOsloExecutor(account='quiet')

fut = e.submit(func1, 2, 3)
print fut.result()

