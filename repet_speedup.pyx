import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def autocorrel(np.ndarray[DTYPE_t, ndim=1] x):
    cdef DTYPE_t sum = 0
    cdef int m = x.shape[0]
    cdef int j, k
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.empty(m, dtype=DTYPE)
    for j in range(m):
        sum = 0
        for k in range(m - j):
            sum += x[k] * x[k + j]
        res[j] = sum / (m - j)

    return res

cdef DTYPE_t cmin(DTYPE_t a, DTYPE_t b):
    if (a < b):
        return a
    else:
        return b

cdef int cminint(int a, int b):
    if (a < b):
        return a
    else:
        return b

cdef int cmaxint(int a, int b):
    if (a < b):
        return b
    else:
        return a

cdef DTYPE_t cmax(DTYPE_t a, DTYPE_t b):
    if (a < b):
        return b
    else:
        return a

@cython.boundscheck(False)
@cython.wraparound(False)
#vraca beat spectrogram
def beatspct(np.ndarray[DTYPE_t, ndim=2] spect):
    cdef int n = spect.shape[0]
    cdef int m = spect.shape[1]
    cdef int w = 300
    cdef int j, h, i
    cdef np.ndarray[DTYPE_t, ndim=2] V = np.empty((n, w), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] B = np.empty((w, m), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] A = np.empty((n, w), dtype=DTYPE)

    for j in range(m):
        for h in range(w):
            for i in range(n):
                V[i, h] = spect[i, cminint(h + j - w // 2, m - 1)] ** 2
        for i in range(n):
            A[i] = autocorrel(V[i])
        B[:, j] = np.mean(A, axis=0)

    return B

# racuna beat spectrum od beat spetrograma
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def bspectrum(np.ndarray[DTYPE_t, ndim=2] B):
    cdef int n = B.shape[0]
    cdef int m = B.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] b = np.ndarray(m, dtype=DTYPE)
    cdef DTYPE_t sum
    for j in range(m):
        sum = 0
        for i in range(n):
            sum += B[i][j]
        b[j] = sum / n
        if j > 0:
            b[j] /= b[0]
    b[0] = 1

    return b

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int argmax(np.ndarray[DTYPE_t, ndim=1] b, int start, int end):
    cdef DTYPE_t max = b[start]
    cdef int index = start
    cdef int i

    for i in range(start + 1, end):
        if (b[i] > max):
            max = b[i]
            index = i
    return index

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_t mean(np.ndarray[DTYPE_t, ndim=1] b, int start, int end):
    cdef DTYPE_t sum = 0
    cdef int i
    cdef DTYPE_t res
    for i in range(start, end):
        sum += b[i]
    res = sum / (end - start)
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int cperiod(np.ndarray[DTYPE_t, ndim=1] beat):
    cdef np.ndarray[DTYPE_t, ndim=1] b = beat[:3 * len(beat) // 4]
    cdef int l = len(b)
    cdef int d = 2
    cdef np.ndarray[DTYPE_t, ndim=1] J = np.zeros([l // 3])
    cdef int minper = 25
    cdef int i, j, D
    cdef double I
    cdef int p
    cdef int h
    for j in range(minper, l // 3):
        D = 3 * j // 4
        I = 0
        i = j
        while i < l:
            h = argmax(b, cmaxint(i - d, 0), cminint(i + d + 1, l))
            if (h + D - d) == argmax(b, i - D, cminint(i + D + 1, l)):
                I += b[h] - mean(b, i - D, cminint(i + D + 1, l))
            i+=j
        J[j] = I / (l // j)

    p = cmaxint(argmax(J, 0, l // 3), minper)

    return p

def period(np.ndarray[DTYPE_t, ndim=1] beat):
    cdef int per = cperiod(beat)
    return per

@cython.boundscheck(False)
@cython.wraparound(False)
def per(np.ndarray[DTYPE_t, ndim=2] bspect):
    cdef int n = bspect.shape[0]
    cdef int m = bspect.shape[1]
    cdef np.ndarray[int, ndim=1] p = np.empty(m, dtype=int)
    cdef int i, j
    cdef np.ndarray[DTYPE_t, ndim=1] b = np.empty(n, dtype=DTYPE)
    for j in range(m):
        for i in range(n):
            b[i] = bspect[i, j]
        p[j] = cperiod(b)

    return p
