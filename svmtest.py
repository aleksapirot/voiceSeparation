from plca import plca, features, load, ncep, nother, np, rand, svm, preprocessing
import os
import sys
import cProfile as cp


def main():
    seglen = 9
    segstart = 10
    mul = 320

    files = np.sort(os.listdir('../base/MIR-1K/Wavfile'))
    rates = np.empty(len(files))
    audios = np.empty((len(files), seglen * mul))
    labels = np.empty((len(files), seglen))
    for i in range(len(files)):
        # print(i)
        file = files[i][:-4]
        lbl = '../base/MIR-1K/vocal-nonvocalLabel/' + file + '.vocal'
        lbl = open(lbl, 'r')
        lines = lbl.readlines()

        lbls = np.empty(seglen)
        for j in range(segstart, segstart + seglen):
            lbls[j - segstart] = int(lines[j])

        labels[i] = lbls

        rate, audio = load('../base/MIR-1K/Wavfile/' + file + '.wav', mono=True)
        audios[i] = audio[segstart * mul:segstart * mul + seglen * mul]
        rates[i] = rate


    a = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    clf = svm.SVC(kernel='poly', degree=2, cache_size=500)
    ltrain = 500
    ltest = 500
    # X = np.ndarray([ltrain, ncep + nother])
    y = np.empty(ltrain)
    scaler = None
    ntries = 20
    for ncep in range(1, 100):
        X = np.ndarray([ltrain, ncep + nother])
        print(ncep, end='')
        print(':')
        np.random.seed(0)
        inds = np.arange(0, 1000)
        for k in range(ntries):
            print(k + 1, end=' ')
            sys.stdout.flush()
            if k == ntries - 1:
                print()

            np.random.shuffle(inds)
            for i in range(len(files)):
                cnt = i
                i = inds[i]

                '''file = files[i][:-4]
                lbl = '../base/MIR-1K/vocal-nonvocalLabel/' + file + '.vocal'
                lbl = open(lbl, 'r')
                lines = lbl.readlines()

                lbls = np.empty(9)
                for j in range(10, 19):
                    lbls[j - 10] = int(lines[j])

                voice = np.median(lbls)

                rate, audio = load('../base/MIR-1K/Wavfile/' + file + '.wav', mono=True)
                audio = audio[10 * 320:19 * 320]'''
                rate = rates[i]
                audio = audios[i]
                voice = np.median(labels[i])

                if (cnt < ltrain):
                    X[cnt] = features(audio, rate, ncep)
                    y[cnt] = voice
                if (cnt > ltrain):
                    X[cnt - ltrain] = features(audio, rate, ncep)
                    y[cnt - ltrain] = voice
                if (cnt == ltrain):
                    scaler = preprocessing.StandardScaler().fit(X)
                    clf.fit(scaler.transform(X), y)
                    X = np.ndarray([ltest, ncep + nother])
                    y = np.empty(ltest)
                    X[0] = features(audio, rate, ncep)
                    y[0] = voice

            predict = clf.predict(scaler.transform(X))
            for i in range(ltest):
                if y[i]:
                    if predict[i]:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if predict[i]:
                        fn += 1
                    else:
                        tn += 1

        print(tp, fp, tn, fn)
        perc = 100 * tn / (tn + fn)
        print('{:.2f}%'.format(perc))
        print('\n')

        tp = 0
        fp = 0
        tn = 0
        fn = 0


# cp.run('main()')
main()
