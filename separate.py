import argparse
import time
from pathlib import Path

from adaptive_repet import adtrepet, repet, repeth
from dmf import dmf, horver
from plca import plca
from common import load, save


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
    print('Počelo u {0.tm_hour:02}:{0.tm_min:02}:{0.tm_sec:02}'.format(time.localtime()))

    parser = argparse.ArgumentParser()
    parser.add_argument('alg') # algoritam
    parser.add_argument('input') # input file
    parser.add_argument('-o') # output folder (isti kao ulazni folder ako nije preciziran)
    parser.add_argument('--wav', action='store_true')
    args = parser.parse_args()

    rate, audio = load(args.input, True)

    start = time.time()
    voice, music = apply(args.alg, audio, rate)
    l = time.time() - start
    print('{:.1f}s za {:.1f}s'.format(l, len(audio)/rate))

    out = args.o
    if out is None:
        out = '../outputs'
    Path(out).mkdir(parents=True, exist_ok=True)
    name = Path(args.input).stem
    ext = 'wav' if args.wav else 'mp3'
    save(voice, rate, "{}/{}-{}-voice.{}".format(out, name, args.alg, ext), not args.wav)
    save(music, rate, "{}/{}-{}-music.{}".format(out, name, args.alg, ext), not args.wav)

    print('Gotovo u {0.tm_hour:02}:{0.tm_min:02}:{0.tm_sec:02}'.format(time.localtime()))
