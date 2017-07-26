import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t median(np.ndarray[DTYPE_t, ndim=1] x):
    cdef DTYPE_t temp
    cdef long int i, j
    cdef long int n = x.shape[0]

    '''for i in range(n):
        for j in range(i+1, n):
            if(x[j] < x[i]):
                temp = x[i]
                x[i] = x[j]
                x[j] = temp'''

    i = 1
    for i in range(n):
        temp = x[i]
        j = i - 1
        while j >= 0 and x[j] > temp:
            x[j+1] = x[j]
            j = j - 1
        x[j+1] = temp

    if(n%2==0):
        return((x[n//2] + x[n//2 - 1]) / 2.0)
    else:
        return x[n//2]


@cython.boundscheck(False)
@cython.wraparound(False)
def diagfilter(np.ndarray[DTYPE_t, ndim=2] spect, int steph, int stepw, int count):
    spect = np.copy(spect)
    cdef int m = spect.shape[0]
    cdef int n = spect.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] values = np.zeros(count, dtype=DTYPE) # vrednosti od kojih se uzima medijana
    cdef int i, j, k
    cdef int half = (count-1)//2
    cdef int w = half*stepw
    cdef int h = half*steph
    if h < 0:
        h = -h

    for i in range(h, m-h):
        for j in range(w, n-w):
            for k in range(-half, half+1):
                values[k+half]=spect[i+k*steph, j+k*stepw]
            spect[i, j] = median(values)

    return spect


@cython.boundscheck(False)
@cython.wraparound(False)
def vertfilter(np.ndarray[DTYPE_t, ndim=2] spect, int count):
    spect = np.copy(spect)
    cdef int m = spect.shape[0]
    cdef int n = spect.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] values = np.zeros(count, dtype=DTYPE) # vrednosti od kojih se uzima medijana
    cdef int i, j, k
    cdef int half = (count-1)//2
    for i in range(half, m-half):
        for j in range(n):
            for k in range(-half, half+1):
                values[k+half]=spect[i+k, j]
            spect[i, j] = median(values)

    return spect


@cython.boundscheck(False)
@cython.wraparound(False)
def horfilter(np.ndarray[DTYPE_t, ndim=2] spect, int count):
    spect = np.copy(spect)
    cdef int m = spect.shape[0]
    cdef int n = spect.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] values = np.zeros(count, dtype=DTYPE) # vrednosti od kojih se uzima medijana
    cdef int i, j, k
    cdef int half = (count-1)//2
    for i in range(0, m):
        for j in range(half, n-half):
            for k in range(-half, half+1):
                values[k+half]=spect[i, j+k]
            spect[i, j] = median(values)

    return spect
