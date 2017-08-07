from common import *
from sklearn import svm, preprocessing
from sklearn.externals import joblib
import os
import mfcc
import scipy.stats.mstats as ms
import numpy.random as rand

ncep = 39
nother = 3


def features(audio, rate,ncep=ncep):
    feats = np.empty(ncep + nother)

    coefs = mfcc.mfcc(audio, samplerate=rate, numcep=ncep, nfilt=2 * ncep, nfft=4096, winlen=4096 / rate,
                      winstep=4096/3 / rate)
    feats[0:ncep] = np.mean(coefs, axis=0)

    fft = np.abs(np.fft.rfft(audio))
    pwr = fft ** 2
    feats[ncep] = ms.gmean(pwr) / np.mean(pwr)

    freq = np.fft.rfftfreq(len(audio), 1 / rate)
    feats[ncep + 1] = np.sum(fft * freq) / np.sum(fft)

    feats[ncep + 2] = np.sqrt(np.mean(np.square(audio.astype(np.int32))))

    return feats

seglen = 3200
def train():
    clf = svm.SVC(kernel='poly', degree=2, cache_size=500)
    files = os.listdir('../base/MIR-1K/Wavfile/')
    np.random.shuffle(files)
    l = 500
    start = 10 * seglen

    X = np.empty([l, ncep + nother])
    y = np.empty([l])
    for i in range(l):
        #print(i)
        file = files[i][:-4]
        lbl = '../base/MIR-1K/vocal-nonvocalLabel/' + file + '.vocal'
        lbl = open(lbl, 'r')
        lines = lbl.readlines()

        lbls = np.empty(9)
        for j in range(10, 19):
            lbls[j - 10] = int(lines[j])

        voice = np.median(lbls)

        rate, audio = load('../base/MIR-1K/Wavfile/' + file + '.wav', mono=True)
        audio = audio[start:start + 9*seglen]

        X[i] = features(audio, rate)
        y[i] = voice

    scaler = preprocessing.StandardScaler().fit(X)
    clf.fit(scaler.transform(X), y)
    joblib.dump(clf, 'clf.pkl')
    joblib.dump(scaler, 'scl.pkl')


def label(audio, rate):
    clf = joblib.load('clf.pkl')
    scaler = joblib.load('scl.pkl')
    X = np.zeros([len(audio) // seglen, ncep + nother])
    for i in range(len(audio) // seglen):
        X[i] = features(audio[i * seglen:(i + 1) * seglen], rate)

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
        #print(i)
        P = np.dot(F, np.dot(Z, T))
        R = S / P

        F[:, nzb:] = norm(F[:, nzb:] * np.dot(R, T[nzb:, :].T), axis=0)
        T = norm(T * np.dot(F.T, R), axis=1)
        zs = np.sum(F, axis=0)
        Z = np.diagflat(zs)

    return F, Z, T


def plca(audio, rate):
    lbl = label(audio, rate)
    #print(lbl)
    mus = []
    for i in range(len(lbl)):
        if lbl[i] == 0:
            mus.extend(audio[i * seglen:(i + 1) * seglen])

    wl = 2048
    ovl = 1024
    f, t, _, spectmus = magspect(mus, rate, wl, noverlap=ovl)
    #plotspect((f,t,spectmus))
    nzb = 200
    nzv = 200
    Fmus, _, _ = learn(spectmus, niter=150, nz=nzb)


    f, t, cspectmix, spectmix = magspect(audio, rate, wl, noverlap=ovl)
    #plotspect((f,t,spectmix))
    Fmix, Z, T = learn(spectmix, niter=150, nz=nzb + nzv, Fmusic=Fmus)

    Pmus = np.dot(Fmus, np.dot(Z[:nzb, :nzb], T[:nzb, :]))
    Fvoc = Fmix[:, nzb:]
    Pvoc = np.dot(Fvoc, np.dot(Z[nzb:, nzb:], T[nzb:, :]))

    #cspectmus = cspectmix*Pmus/(Pmus+Pvoc)
    cspectvoc = cspectmix * Pvoc / (Pmus + Pvoc)

    voice = inversestft(cspectvoc, wl, noverlap=ovl)[0:len(audio)]
    music = audio - voice

    return voice, music

