from repet import *


# beat spectrogram
# previse sporo radi
def beatspect(spect):
    # spect = spect[:, 0:600]
    n = spect.shape[0]
    m = spect.shape[1]
    w = min(m // 3, 300)
    step = w // 2
    V = np.zeros((n, w))
    A = np.zeros((n, w))
    B = np.zeros((w, m))
    for j in range(0, m, step):
        for h in range(w):
            V[:, h] = spect[:, min(h + j - w // 2, m - 1)] ** 2
        for i in range(n):
            A[i] = autocor(V[i])
        B[:, j] = np.mean(A, axis=0)

    return B
    # B=beatspct(spect.astype(np.float64))
    # plt.plot(B)
    # return B


# racuna periode za svaki beat spectrum iz spectrograma
# previse sporo radi
def periods(bspect):
    p = per(bspect)
    # print(p)
    return p


# repeating spectrogram
def rspect(spect, ps):
    k = 2
    n = spect.shape[0]
    m = spect.shape[1]
    U = np.ndarray((n, m))
    for j in range(m):
        U[:, j] = np.median(spect[:, j + ps * np.arange(1, k) - ps * k // 2])
    return U


def adtrepet(audio, rate):
    return adaptiverepet(audio, rate, False)


def adaptiverepet(audio, rate, highpass):
    winlen = 1024
    f, t, cspect, spect = magspect(audio, rate, winlen=winlen)
    # plotspect((f, t, spect))

    bt = beatspect(spect)
    # plt.plot(bt)
    # plt.show()

    ps = periods(bt)
    # print(ps)

    rs = rspect(spect, ps)
    # plt.plot(rs)
    # plt.show()

    # nije gotovo
