from common import *
from repet_speedup import *


# racuna beat spectrum
def beat(spect):
    n = spect.shape[0] // 2 + 1
    m = spect.shape[1]
    B = np.ndarray((n, m))
    for i in range(n):
        '''for j in range(m):
            sum = 0
            for k in range(m-j):
                sum += spect[i][k]*spect[i][k + j]
            B[i][j] = sum/(m-j)'''
        #result = np.correlate(spect[i], spect[i], mode='full')
        #result = result[result.size // 2:]
        #B[i] = result * np.exp(np.linspace(0, 1.7, num=result.size))
        B[i] = autocorrel(spect[i].astype(np.float64))

    b = np.ndarray([m])
    for j in range(m):
        sum = 0
        for i in range(n):
            sum += B[i][j]
        b[j] = sum / n
        if j > 0:
            b[j] /= b[0]
    b[0] = 1

    return b


# racuna period
def period(beat):
    b = beat[:3 * len(beat) // 4]
    l = len(b)
    d = 2
    J = np.zeros([l // 3])
    minper = 1
    for j in range(minper, l // 3):
        D = 3 * j // 4
        I = 0
        for i in range(j, l, j):
            h = np.argmax(b[np.max([i - d, 0]):np.min([i + d + 1, l])])
            if (h + D - d) == np.argmax(b[i - D:np.min([i + D + 1, l])]):
                I += b[h] - np.mean(b[i - D:np.min([i + D + 1, l])])
        J[j] = I / (l // j)

    p = np.max([np.argmax(J), minper])

    return p


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


# racuna masku
def mask(spect, repspect, rate, highpass):
    msk = repspect / spect
    if highpass:
        count = 2 * 100 * spect.shape[0] // rate
        msk[0:count, :] = np.ones((count, msk.shape[1]))
    return msk


def applymask(mask, spect):
    return (mask * spect, spect - mask * spect)


# repet sa high pass filterom
def repeth(audio, rate):
    return rpt(audio, rate, True)


def repet(audio, rate):
    return rpt(audio, rate, False)


def rpt(audio, rate, highpass):
    winlen = 1024
    f, t, cspect, spect = magspect(audio, rate, winlen=winlen)
    #plotspect((f, t, spect))

    bt = beat(spect)
    # plt.plot(t, bt)

    per = period(bt)
    # print(t[per])

    seg = segment(spect, per)
    # plotspect((f, t[0:per], seg))

    rep = repspect(spect, seg)
    # plotspect((f, t, rep))

    msk = mask(spect, rep, rate, highpass)
    # plotspect((f, t, msk), maxcoef=1)

    musicspect, voicespect = applymask(msk, cspect)
    # plotspect((f, t, musicspect))
    # plotspect((f, t, voicespect))


    music = inversestft(musicspect, winlen)[0:len(audio)]
    voice = inversestft(voicespect, winlen)[0:len(audio)]

    return voice, music
