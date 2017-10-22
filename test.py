import os
import time
import argparse
from pathlib import Path

from common import load, evaluate, np, labels
from separate import apply
from plca import plca, segnums, segnuml


def test(algorithm, resfolder='', longer=False):
    dir = '../base/MIR-1K/UndividedWavfile/' if longer else '../base/MIR-1K/Wavfile/'
    metrics = ''  # isto menjamo i ime fajla

    if longer:
        resfolder = '-long' + resfolder

    count = len(os.listdir(dir))
    results = np.empty((count, 6))
    songs = np.sort(os.listdir(dir))[0:count]
    i = 0

    algorithm = algorithm.upper()
    savedir = '../results{}/{}'.format(resfolder, algorithm)
    Path(savedir).mkdir(parents=True, exist_ok=True)
    for i in range(count):
        print('{}/{}'.format(i+1, count))

        song = songs[i] = songs[i][:-4]
        rate, audiol, audior = load('{}/{}.wav'.format(dir, song))
        audio = audiol // 2 + audior // 2

        if algorithm == 'PLCAL':
            voice, music = plca(audio, rate, lbl=labels(song, segnuml if longer else segnums, longer))
        else:
            voice, music = apply(algorithm, audio, rate)

        '''if i < 10:
            save(music, rate, '{}/{}-music.wav'.format(savedir, name))
            save(voice, rate, '{}/{}-voice.wav'.format(savedir, name))'''

        # print('gotov algoritam')

        sdr, sir, sar = evaluate(audior, audiol, voice, music)
        print("\033[1A\033[J", end='') # brise prosli red
        print('SDR: {0[0]:05.2f}(V)  {0[1]:05.2f}(M)  SIR: {1[0]:05.2f}(V)  {1[1]:05.2f}(M)  SAR: {2[0]:05.2f}(V)  {2[1]:05.2f}(M)'.format(sdr, sir, sar))
        results[i] = np.concatenate([sdr, sir, sar])


    mean = '\nMean:\nSDR: {0[0]:05.2f}(V)  {0[1]:05.2f}(M) \nSIR: {0[2]:05.2f}(V)  {0[3]:05.2f}(M) \nSAR: {0[4]:05.2f}(V)  {0[5]:05.2f}(M)'.format(np.mean(results, axis=0))
    median = ' \nMedian:\nSDR: {0[0]:05.2f}(V)  {0[1]:05.2f}(M) \nSIR: {0[2]:05.2f}(V)  {0[3]:05.2f}(M) \nSAR: {0[4]:05.2f}(V)  {0[5]:05.2f}(M)'.format(np.median(results, axis=0))
    maximum = ('\nMaximum:\nSDR: {0[0]:05.2f}[{1[0]:03}](V)  {0[1]:05.2f}[{1[1]:03}](M) \nSIR: {0[2]:05.2f}[{1[2]:03}](V)  {0[3]:05.2f}[{1[3]:03}](M)'
          '\nSAR: {0[4]:05.2f}[{1[4]:03}](V)  {0[5]:05.2f}[{1[5]:03}](M)').format(np.max(results, axis=0), np.argmax(results, axis=0)+1)
    minimum = ('\nMin:\nSDR: {0[0]:05.2f}[{1[0]:03}](V)  {0[1]:05.2f}[{1[1]:03}](M) \nSIR: {0[2]:05.2f}[{1[2]:03}](V)  {0[3]:05.2f}[{1[3]:03}](M)'
          '\nSAR: {0[4]:05.2f}[{1[4]:03}](V)  {0[5]:05.2f}[{1[5]:03}](M)').format(np.min(results, axis=0), np.argmin(results, axis=0)+1)

    print(mean)
    print(median)
    print(maximum)
    print(minimum)

    file = open('{}/metrics{}.txt'.format(savedir, metrics), 'w+')

    for i in range(count):
        print('{0:20}: SDR [{1[0]:05.2f}  {1[1]:05.2f}]  SIR [{1[2]:05.2f}  {1[3]:05.2f}]  SAR [{1[4]:05.2f}  {1[5]:05.2f}]'
              .format(songs[i], results[i]), file=file)

    print(mean, file=file)
    print(median, file=file)
    print(maximum, file=file)
    print(minimum, file=file)

    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('alg') # algoritam
    parser.add_argument('-r', dest='res', default='') # resfolder
    parser.add_argument('-l', '--long', dest='long', action='store_true') #ako ima -l ili --long radi na duzim klipovima (UndividedWavfile)
    args = parser.parse_args()

    now = time.localtime()
    print('Testiranje počelo u {0.tm_hour:02}:{0.tm_min:02}:{0.tm_sec:02}'.format(now))

    test(args.alg, args.res, args.long)

    now = time.localtime()
    print('Testiranje gotovo u {0.tm_hour:02}:{0.tm_min:02}:{0.tm_sec:02}'.format(now))
