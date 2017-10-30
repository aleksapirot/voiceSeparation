import argparse
import time
import cProfile as cp
from pathlib import Path

from adaptive_repet import adtrepet, repet
from dmf import dmf, horver
from plca import plca, train
from common import load, save
from svmtest import svmtest


funcs = {'R': repet, 'R_ADT': adtrepet, 'R_SIM': None,
         'HV': horver, 'DMF': dmf, 'PLCA': plca}


def apply(algorithm, audio, rate):
    dtype = audio.dtype
    algorithm = algorithm.upper()

    if '-' in algorithm:
        algs = algorithm.split('-')
        voice1, music1 = apply(algs[0], audio, rate)
        v, _ = apply('-'.join(algs[1:]), voice1, rate)
        _, m = apply('-'.join(algs[1:]), music1, rate)
    else:
        if algorithm[-1] == 'H':
            v, m = funcs[algorithm[:-1]](audio, rate, True)
        else:
            v, m = funcs[algorithm](audio, rate, False)


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
