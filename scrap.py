from adaptive_repet import *
from dmf import *
from plca import *
import test
import cProfile as cp


dir = '../base/MIR-1K/Wavfile/'#'../base/repet/'#'../base/MIR-1K/Wavfile/'
name = 'Hushabye_mus'#'Don\'t Talk (Put Your Head On My Shoulder)_mus'
#rate, om = load('{}{}.wav'.format(dir, name), mono=True)
name = 'Hushabye_voc'#'Don\'t Talk (Put Your Head On My Shoulder)_voc'
#rate, ov = load('{}{}.wav'.format(dir, name), mono=True)
#ov = np.concatenate([ov, [0]])
#om = np.concatenate([om, [0,0]])
'''name = 'Don\'t Talk (Put Your Head On My Shoulder)_mix-music'
rate, em = load('{}{}.wav'.format(dir, name), mono=True)
name = 'Don\'t Talk (Put Your Head On My Shoulder)_mix-voice'
rate, ev = load('{}{}.wav'.format(dir, name), mono=True) '''
name = 'Hushabye_mix'#'Don\'t Talk (Put Your Head On My Shoulder)_mix'
name = 'abjones_4_01'
#rate, audio = load('{}{}.wav'.format(dir, name), mono=True)
rate, om, ov = load('{}{}.wav'.format(dir, name), mono=False)
audio = om/2 + ov/2#audio = (om, ov)#audio = om/2 + ov/2

algorithm = 'plca'
funcs = {'repet': repet, 'repet-h': repeth, 'adaptive-repet': adtrepet, 'repet-sim': None, 'horver': horver, 'dmf': dmf, 'plca': plca}

#train()
t = np.arange(0, 2e5)

cp.run('evaluate(t, t, t, t)')

#import scipy.fftpack as fft
#cp.run('fft.ifft(t)')

cp.run('test.test()')

#cp.run('funcs[algorithm](audio, rate)')

'''voice, music = funcs[algorithm](audio, rate)#funcs[algorithm](audio, rate)

save(music, rate, '../results2/{}/{}-music.wav'.format(algorithm, name))
save(voice, rate, '../results2/{}/{}-voice.wav'.format(algorithm, name))
#print(len(ov), len(om), len(voice), len(music))
print(evaluate(ov, om, voice, music))#voice, music))
#print(evaluate(ov, om, ev[:7494070], em[:7494070]))'''