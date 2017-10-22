from plca import plca, features, load, ncep, nother, np, rand, svm, preprocessing, segnums, segnuml
from common import labels
import os
import sys
import cProfile as cp


def svmtest(longer=False):
    dir = '../base/MIR-1K/UndividedWavfile/' if longer else '../base/MIR-1K/Wavfile/'
    files = np.sort(os.listdir(dir))
    l = len(files)

    segnum = segnuml if longer else segnums
    seglen = (20*16000)//1000
    ntries = 10
    ltrain = 200
    ltest = 400

    audios = []
    rates = np.empty(l)
    voices = []
    segcounts = np.empty(l, dtype=int)

    # segnum segmenata (svaki duzine seglen samplova) se posmatra kao jedan veliki, u jednom fajlu je segcount velikih segmenata

    for i in range(l):
        file = files[i][:-4]
        
        rates[i], audio = load(dir + file + '.wav', mono=True)
        lbl = labels(file, segnum)

        voices1 = []
        audios1 = []
        segcount = len(lbl)
        segcounts[i] = segcount
        for j in range(segcount):
            start = segnum*j
            audios1.append(audio[start*seglen:(start + segnum) * seglen])

        audios.append(audios1)
        voices.append(lbl)

    clf = svm.SVC(kernel='poly', degree=2, cache_size=500)
    scaler = None
    for ncep in range(1, 100):
        print('{}:'.format(ncep))

        tp = 0; fp = 0; tn = 0; fn = 0
        np.random.seed(0)
        inds = np.arange(l)

        ftrs = []
        for i in range(l):
            ftrs1 = []
            for j in range(segcounts[i]):
                ftrs1.append(features(audios[i][j], rates[i], ncep))
            ftrs.append(ftrs1)

        for k in range(ntries):
            print(k + 1, end=' ')
            sys.stdout.flush()
            if k == ntries - 1:
                print()

            X = []
            y = []

            np.random.shuffle(inds)
            for i in range(ltrain + ltest):
                if (i == ltrain):
                    scaler = preprocessing.StandardScaler().fit(X)
                    clf.fit(scaler.transform(X), y)
                    X = []
                    y = []

                i = inds[i]
                for j in range(segcounts[i]):
                    X.append(ftrs[i][j])
                    y.append(voices[i][j])

            predict = clf.predict(scaler.transform(X))
            a = 0
            ap = 0; an = 0
            for i in range(ltrain, ltrain+ltest):
                i = inds[i]
                for j in range(segcounts[i]):
                    if voices[i][j]:
                        ap += 1
                        if predict[a]:
                            tp += 1
                        else:
                            fn += 1
                    else:
                        an += 1
                        if predict[a]:
                            fp += 1
                        else:
                            tn += 1
                    a += 1

        print(ap, an)
        print(tp, fp, tn, fn)
        perc = 100 * tn / (tn + fn)
        print('{:.2f}%'.format(perc))
        perc = 100 * tn / (tn + fp)
        print('{:.2f}%'.format(perc))
        print('\n')

        tp = 0
        fp = 0
        tn = 0
        fn = 0
