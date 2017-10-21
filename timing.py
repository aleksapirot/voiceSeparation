import argparse
import time
import os
from pathlib import Path
from numpy import sort
from common import load
from plca import train
from separate import apply

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('alg') # algoritam
    parser.add_argument('-r', dest='res', default='') # resfolder
    parser.add_argument('--long', action='store_true')
    args = parser.parse_args()

    alg = args.alg.upper()
    if alg == 'TRAIN':
        start = time.time()
        train()
        l = time.time() - start
        print('{:.1f}s za treniranje'.format(l))
    else:
        length = 0
        audios = []
        rates = []

        if args.long:
            dir = '../base/MIR-1K/UndividedWavfile'
        else:
            dir = '../base/MIR-1K/Wavfile'

        count = len(os.listdir(dir))
        songs = sort(os.listdir(dir))[0:count]

        for i in range(count):
            song = songs[i] = songs[i][:-4]
            rate, audio = load('{}/{}.wav'.format(dir, song), True)
            length += len(audio) / rate
            rates.append(rate)
            audios.append(audio)

        l = 0

        for i in range(count):
            print('{}/{}'.format(i, count))
            start = time.time()
            voice, music = apply(alg, audios[i], rates[i])
            l += time.time() - start
            # TODO baseline
            print("\033[1A\033[J", end='') # brise prosli red

        print('{:.1f}s za {:.1f}s'.format(l, length))
        print('{:.1f}s po minutu'.format(l*60/length))

        savedir = '../results{}/{}'.format(args.res, alg)
        Path(savedir).mkdir(parents=True, exist_ok=True)
        timing=''
        file = open('{}/timing{}.txt'.format(savedir, timing), 'w+')
        print('{:.1f}s za {:.1f}s'.format(l, length), file=file)
        print('{:.1f}s po minutu'.format(l*60/length), file=file)
        file.close()
