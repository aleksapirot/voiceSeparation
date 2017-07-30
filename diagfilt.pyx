import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t median(np.ndarray[DTYPE_t, ndim=1] x, int start, int end):
    cdef DTYPE_t temp
    cdef int i, j
    cdef int n = end - start
    cdef int mid = start + n // 2

    for i in range(start + 1, end):
        temp = x[i]
        j = i - 1
        while j >= start and x[j] > temp:
            x[j + 1] = x[j]
            j = j - 1
        x[j + 1] = temp

    if (n % 2 == 0):
        return ((x[mid - 1] + x[mid]) / 2.0)
    else:
        return x[mid]

#@cython.boundscheck(False)
#@cython.wraparound(False)
def diagfilter(np.ndarray[DTYPE_t, ndim=2] spect, int steph, int stepw, int count):
    spect = np.copy(spect)
    cdef int m = spect.shape[0]
    cdef int n = spect.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] values = np.zeros(count, dtype=DTYPE)  # vrednosti od kojih se uzima medijana
    cdef int i, j, k
    cdef int half = (count - 1) // 2
    cdef int w = half * stepw
    cdef int h = half * steph
    if h < 0:
        h = -h

    for i in range(h, m - h):
        for j in range(0, w):
            for k in range(-(j // stepw), half + 1):
                values[k + half] = spect[i + k * steph, j + k * stepw]
            spect[i, j] = median(values, half - (j // stepw), count)
        for j in range(w, n - w):
            for k in range(-half, half + 1):
                values[k + half] = spect[i + k * steph, j + k * stepw]
            spect[i, j] = median(values, 0, count)
        for j in range(n - w, n):
            for k in range(-half, (n - j) // stepw):
                values[k + half] = spect[i + k * steph, j + k * stepw]
            spect[i, j] = median(values, 0, half + (n - j) // stepw)

    return spect

@cython.boundscheck(False)
@cython.wraparound(False)
def vertfilter(np.ndarray[DTYPE_t, ndim=2] spect, int count):
    spect = np.copy(spect)
    cdef int m = spect.shape[0]
    cdef int n = spect.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] values = np.zeros(count, dtype=DTYPE)  # vrednosti od kojih se uzima medijana
    cdef int i, j, k
    cdef int half = (count - 1) // 2

    for j in range(n):
        for i in range(0, half):
            for k in range(-i, half + 1):
                values[k + half] = spect[i + k, j]
            spect[i, j] = median(values, half - i, count)
        for i in range(half, m - half):
            for k in range(-half, half + 1):
                values[k + half] = spect[i + k, j]
            spect[i, j] = median(values, 0, count)
        for i in range(m - half, m):
            for k in range(-half, m - i):
                values[k + half] = spect[i + k, j]
            spect[i, j] = median(values, 0, half + m - i)

    return spect

@cython.boundscheck(False)
@cython.wraparound(False)
def horfilter(np.ndarray[DTYPE_t, ndim=2] spect, int count):
    spect = np.copy(spect)
    cdef int m = spect.shape[0]
    cdef int n = spect.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] values = np.zeros(count, dtype=DTYPE)  # vrednosti od kojih se uzima medijana
    cdef int i, j, k
    cdef int half = (count - 1) // 2
    for i in range(0, m):
        for j in range(0, half):
            for k in range(-j, half + 1):
                values[k + half] = spect[i, j + k]
            spect[i, j] = median(values, half - j, count)
        for j in range(half, n - half):
            for k in range(-half, half + 1):
                values[k + half] = spect[i, j + k]
            spect[i, j] = median(values, 0, count)
        for j in range(n - half, n):
            for k in range(-half, n - j):
                values[k + half] = spect[i, j + k]
            spect[i, j] = median(values, 0, half + n - j)
    return spect

@cython.boundscheck(False)
@cython.wraparound(False)
def matrixmax(np.ndarray[DTYPE_t, ndim=3] matrices):
    cdef int c = matrices.shape[2]
    cdef int n = matrices.shape[0]
    cdef int m = matrices.shape[1]
    cdef int i,j,k
    cdef DTYPE_t maxh
    cdef np.ndarray[DTYPE_t, ndim=2] values = np.zeros((n,m), dtype=DTYPE)
    for i in range(n):
        for j in range(m):
            maxh=matrices[i,j,0]
            for k in range(1,c):
                if matrices[i,j,k]>maxh:
                    maxh=matrices[i,j,k]
            values[i,j]=maxh
    return values



