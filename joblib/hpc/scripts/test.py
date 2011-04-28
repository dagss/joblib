from joblib.hpc import versioned
from joblib.hpc.executor import TitanOsloExecutor, SlurmExecutor
from joblib.hpc.test.funcs import func1, funcex
import logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
e = TitanOsloExecutor(account='quiet', logger=logger)

fut1 = e.submit(func1, 2, 3)
fut2 = e.submit(funcex, 2, 3)

print fut1.result()
print type(fut2.exception(100))
print fut2.result(100)
