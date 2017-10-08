import argparse
import time
import cProfile as cp
from pathlib import Path

from adaptive_repet import adtrepet, repet, repeth
from dmf import dmf, horver
from plca import plca, train
from common import load, save
from svmtest import svmtest


funcs = {'R': repet, 'RH': repeth, 'R_ADT': adtrepet, 'R_SIM': None,
         'HV': horver, 'DMF': dmf, 'PLCA': plca}


def apply(algorithm, audio, rate):
    dtype = audio.dtype
    algorithm = algorithm.upper()
    if '-' in algorithm:
        algs = algorithm.split('-')
        voice1, music1 = funcs[algs[0]](audio, rate)
        v, _ = funcs[algs[1]](voice1, rate)
        _, m = funcs[algs[1]](music1, rate)
    else:
        v, m = funcs[algorithm](audio, rate)

    return v.astype(dtype), m.astype(dtype)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('alg') # algoritam
    parser.add_argument('input', nargs='?') # input file
    parser.add_argument('output', nargs='?') # output folder (default je "../outputs")
    parser.add_argument('--wav', action='store_true')
    parser.add_argument('--cp', action='store_true')
    parser.add_argument('--long', action='store_true')
    args = parser.parse_args()

    alg = args.alg.upper()
    if alg == 'TRAIN':
        if args.cp:
            cp.run('train()')
        else:
            start = time.time()
            train()
            l = time.time() - start
            print('{:.1f}s za treniranje'.format(l))
    elif alg == 'SVM':
        if args.cp:
            cp.run('svmtest()')
        else:
            start = time.time()
            svmtest()
            l = time.time() - start
            print('{:.1f}s za SVM test'.format(l))
    else:
        rate, audio = load(args.input, True)
        if args.cp:
            cp.run('apply(args.alg, audio, rate)')
        else:
            start = time.time()
            voice, music = apply(args.alg, audio, rate)
            l = time.time() - start
            print('{:.1f}s za {:.1f}s'.format(l, len(audio)/rate))

            out = args.output
            if out is None:
                out = '../outputs'
            Path(out).mkdir(parents=True, exist_ok=True)
            name = Path(args.input).stem
            ext = 'wav' if args.wav else 'mp3'
            save(voice, rate, "{}/{}-{}-voice.{}".format(out, name, args.alg, ext), not args.wav)
            save(music, rate, "{}/{}-{}-music.{}".format(out, name, args.alg, ext), not args.wav)
