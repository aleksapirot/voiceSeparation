import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
def autocorrel(np.ndarray[DTYPE_t, ndim=1] x):
    cdef DTYPE_t sum = 0
    cdef int m = x.shape[0]
    cdef int j, k
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.empty(m, dtype=DTYPE)
    for j in range(m):
        sum = 0
        for k in range(m-j):
            sum += x[k]*x[k + j]
        res[j] = sum/(m-j)

    return res