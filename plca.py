from common import *
from sklearn import svm, preprocessing
from sklearn.externals import joblib
import argparse
import time
import os
import mfcc
import scipy.stats.mstats as ms
import numpy.random as rand

ncep = 25
nother = 3

def features(audio, rate, ncp=ncep):
    feats = np.empty(ncp + nother)

    coefs = mfcc.mfcc(audio, samplerate=rate, numcep=ncp, nfilt=2 * ncp, nfft=4096, winlen=4096 / rate,
                      winstep=4096 / 3 / rate)
    feats[0:ncp] = np.mean(coefs, axis=0)

    fft = np.abs(np.fft.rfft(audio))
    pwr = fft ** 2
    pwr.clip(min=np.finfo(pwr.dtype.type).eps)
    feats[ncp] = ms.gmean(pwr) / np.mean(pwr)

    freq = np.fft.rfftfreq(len(audio), 1 / rate)
    feats[ncp + 1] = np.sum(fft * freq) / np.sum(fft)

    feats[ncp + 2] = np.sqrt(np.mean(np.square(audio.astype(np.float64))))

    return feats


# fajl sa labelama za glas ima segmente duzine 20ms(seglen), posto je to dosta krakto vreme koristimo zajedno segnum segmenata (duzine biglen)
# TODO treba da se istrenira posebno za duge
segnums = 20
segnuml = 50
seglen = (20*16000)//1000
biglens = segnums*seglen
biglenl = segnuml*seglen

def train(longer=False):
    dir = '../base/MIR-1K/UndividedWavfile/' if longer else '../base/MIR-1K/Wavfile/'
    files = os.listdir(dir)
    rand.seed(0)
    rand.shuffle(files)
    l = len(files)//2

    segnum = segnuml if longer else segnums

    X = []
    y = []
    for i in range(l):
        # print(i)
        file = files[i][:-4]

        rate, audio = load(dir + file + '.wav', mono=True)
        lbl = labels(file, segnum, longer)

        for j in range(len(lbl)):
            start = segnum*j
            audio1 = audio[start*seglen:(start + segnum) * seglen]
            X.append(features(audio1, rate, ncep))

        y.extend(lbl)


    scaler = preprocessing.StandardScaler().fit(X)
    clf = svm.SVC(kernel='poly', degree=2, cache_size=500)
    clf.fit(scaler.transform(X), y)
    joblib.dump([clf, scaler], 'plca.pkl' if not longer else 'plca-long.pkl')


def label(audio, rate, biglen):
    clf, scaler = joblib.load('plca.pkl')[:2] # TODO treba proemniti u plca-long.pkl za duze
    X = np.zeros([len(audio) // biglen, ncep + nother])
    for i in range(len(audio) // biglen):
        X[i] = features(audio[i * biglen:(i + 1) * biglen], rate)

    return clf.predict(scaler.transform(X))


def norm(a, axis=None):
    return a / np.sum(a, axis=axis, keepdims=True)


def learn(S, nz, niter=100, Fmusic=None):
    S = norm(S)

    f = S.shape[0]
    t = S.shape[1]
    P = np.empty(S.shape)
    R = np.empty(S.shape)

    rand.seed(0)
    F = norm(rand.uniform(size=[f, nz], low=1e-10), axis=0)

    nzb = 0
    if Fmusic is not None:
        nzb = Fmusic.shape[1]
        F[:, :nzb] = np.copy(Fmusic)

    rand.seed(0)
    T = norm(rand.uniform(size=[nz, t], low=1e-10), axis=1)

    rand.seed(0)
    zs = norm(rand.uniform(size=nz, low=1e-10))
    Z = np.diagflat(zs)

    for i in range(niter):
        # print(i)
        P = np.dot(F, np.dot(Z, T))
        R = S / P

        F[:, nzb:] = norm(F[:, nzb:] * np.dot(R, T[nzb:, :].T), axis=0)
        T = norm(T * np.dot(F.T, R), axis=1)
        zs = np.sum(F, axis=0)
        Z = np.diagflat(zs)
    return F, Z, T


def plca(audio, rate, highpass=False, lbl=None, longer=False):
    segnum = segnuml if longer else segnums
    biglen = biglenl if longer else biglens
    if lbl is None:
        lbl = label(audio, rate, biglen)
    # print(lbl)
    mus = []
    for i in range(len(lbl)):
        if lbl[i] == 0:
            mus.extend(audio[i * biglen:(i + 1) * biglen])
    if len(mus) == 0: #SVM nije prepoznao delove bez muzike
        mus.extend(audio[:biglen]) #dodaje se bilo koji (prvi) segment kao muzika jer mora biti bar jedan

    wl = 2048
    ovl = 1024
    f, t, _, spectmus = magspect(mus, rate, wl, noverlap=ovl)
    # plotspect((f,t,spectmus))
    nzb = 150
    nzv = 50
    niter = 75
    Fmus, _, _ = learn(spectmus, nzb, niter)

    f, t, cspectmix, spectmix = magspect(audio, rate, wl, noverlap=ovl)
    # plotspect((f,t,spectmix))
    Fmix, Z, T = learn(spectmix, nzb + nzv, niter, Fmus)

    Pmus = np.dot(Fmus, np.dot(Z[:nzb, :nzb], T[:nzb, :]))
    Fvoc = Fmix[:, nzb:]
    Pvoc = np.dot(Fvoc, np.dot(Z[nzb:, nzb:], T[nzb:, :]))

    # cspectmus = cspectmix*Pmus/(Pmus+Pvoc)
    mask = Pvoc / (Pmus + Pvoc)

    return applymask(audio, cspectmix, mask, wl, ovl, highpass, rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('alg') # algoritam
    parser.add_argument('--cp', action='store_true')
    parser.add_argument('--long', action='store_true')
    args = parser.parse_args()

    alg = args.alg.upper()
    if alg == 'TRAIN':
        if args.cp:
            cp.run('train(args.long)')
        else:
            start = time.time()
            train(args.long)
            l = time.time() - start
            print('{:.1f}s za treniranje'.format(l))
    elif alg == 'SVM':
        if args.cp:
            cp.run('svmtest(args.long)')
        else:
            start = time.time()
            from svmtest import svmtest
            svmtest(args.long)
            l = time.time() - start
            print('{:.1f}s za SVM test'.format(l))