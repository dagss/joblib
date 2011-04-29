from functools import wraps
from ..func_inspect import filter_args, get_func_code, get_func_name
from ..hashing import hash
import hashlib

__all__ = ['versioned']

def versioned_call(func, *args, **kwargs):
    """ Calls a function and tracks versioned functions being called.

    The intention is to use this for tracking changed dependencies.
    In addition to the output of the function, one gets a list of the
    functions called during the computation that are decorated with
    the @versioned decorator. Tracing is temporarily turned
    off when encountering functions flagged with "deps==False".

    Returns
    -------

    (func_output, list_of_called_functions)

    """
    if hasattr(func, 'version_info'):
        if func.version_info['ignore_deps'] == False:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    output = func(*args, **kwargs)
    return (output, [func])
    
def versioned(version=None, deps=True, ignore=()):
    passed_func = None
    if hasattr(version, '__call__'):
        # Used as decorator without ()
        if deps != True or ignore != ():
            raise TypeError('Invalid version argument')
        passed_func = version
        version = None
    def dec(func):
        # Make hash. The function hash does not consider dependencies.
        h = hashlib.sha1()
        module, name = get_func_name(func)
        h.update('.'.join(module + [name]))
        h.update('$')
        if version is None:
            # No manual version, so should hash by contents
            src, source_file, lineno = get_func_code(func)
            h.update(src)
        else:
            h.update(str(version))
        # Store information
        func.version_info = dict(version=version,
                                 ignore_deps=deps == False,
                                 ignore_args=tuple(ignore),
                                 digest=h.digest(),
                                 hexdigest=h.hexdigest())
        return func
    if passed_func is None:
        return dec
    else:
        return dec(passed_func)

