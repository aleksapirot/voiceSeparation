from common import *
from repet_speedup import *


def autocor(spect):
    m = spect.shape[0]
    return signal.correlate(spect ** 2, spect ** 2)[m - 1:] / np.arange(m, 0, -1)

# racuna beat spectrum
# moglo bi se prebaciti u cython
def beat(spect):
    n = spect.shape[0] // 2 + 1
    m = spect.shape[1]
    B = np.ndarray((n, m))
    for i in range(n):
        B[i] = autocor(spect[i])

    return bspectrum(B)


# deli spektrogram na periode
def split(spect, period, last=True):
    split = np.split(spect, period * np.arange(1, spect.shape[1] // period + 1), axis=1)
    if last:
        return split
    else:
        return split[:-1]


# racuna ponavljajuci segment
def segment(spect, period):
    return np.median(split(spect, period, last=False), axis=0)


# racuna repeating spectrogram
def repspect(spect, seg):
    period = seg.shape[1]
    splt = split(spect, period)
    W = np.zeros(spect.shape)
    for i in range(0, len(splt) - 1):
        W[:, i * period:(i + 1) * period] = np.min([spect[:, i * period:(i + 1) * period], seg], axis=0)

    x = (len(splt) - 1) * period
    W[:, x:] = np.min([spect[:, x:], seg[:, :W.shape[1] - x]], axis=0)

    return W


def repet(audio, rate, highpass):
    winlen = 1024
    f, t, cspect, spect = magspect(audio, rate, winlen=winlen)
    #plotspect((f, t, spect))

    bt = beat(spect)
    #plt.plot(t, bt)
    #plt.show()

    per = period(bt)
    #print(t[per])
    #print(per)

    seg = segment(spect, per)
    # plotspect((f, t[0:per], seg))

    rep = repspect(spect, seg)
    # plotspect((f, t, rep))

    mask = rep / clip(spect)
    # plotspect((f, t, msk), maxcoef=1)

    return applymask(audio, cspect, 1-mask, winlen, None, highpass, rate)
