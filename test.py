from adaptive_repet import *
from dmf import *
import os

dir = '../base/MIR-1K/Wavfile/'
results = {}
i = 1
count = len(os.listdir(dir))
algorithm = 'dmf'
resfolder = '2'
funcs = {'repet': repet, 'repet-h': repeth, 'adaptive-repet': adtrepet, 'repet-sim': None, 'horver': horver, 'dmf': dmf, 'plca': None}
for name in os.listdir(dir)[0:count]:
    print('{}/{}'.format(i, count))
    i += 1

    name = name[:-4]

    rate, audiol, audior = load('{}{}.wav'.format(dir, name))
    audio = audiol // 2 + audior // 2

    voice, music = funcs[algorithm](audio, rate)

    #save(music, rate, '../results{}/{}/{}-music.wav'.format(resfolder, algorithm, name))
    #save(voice, rate, '../results{}/{}/{}-voice.wav'.format(resfolder, algorithm, name))

    sdr, sir, sar = evaluate(audior, audiol, voice, music)
    results[name] = {'SDR': sdr, 'SIR': sir, 'SAR': sar}

avgsdr = np.zeros(2)
avgsir = np.zeros(2)
avgsar = np.zeros(2)
maxsdrv = -100
maxsdrm = -100
maxsirv = -100
maxsirm = -100
maxsarv = -100
maxsarm = -100
minsdrv = 100
minsdrm = 100
minsirv = 100
minsirm = 100
minsarv = 100
minsarm = 100
file = open('../results{}/{}/metrics.txt'.format(resfolder, algorithm), 'w+')
for song in results:
    sdr = results[song]['SDR']
    sir = results[song]['SIR']
    sar = results[song]['SAR']

    if (sdr[0] > maxsdrv):
        maxsdrv = sdr[0]
    if (sdr[1] > maxsdrm):
        maxsdrm = sdr[1]
    if (sir[0] > maxsirv):
        maxsirv = sir[0]
    if (sir[1] > maxsirm):
        maxsirm = sir[1]
    if (sar[0] > maxsarv):
        maxsarv = sar[0]
    if (sar[1] > maxsarm):
        maxsarm = sar[1]

    if (sdr[0] < minsdrv):
        minsdrv = sdr[0]
    if (sdr[1] < minsdrm):
        minsdrm = sdr[1]
    if (sir[0] < minsirv):
        minsirv = sir[0]
    if (sir[1] < minsirm):
        minsirm = sir[1]
    if (sar[0] < minsarv):
        minsarv = sar[0]
    if (sar[1] < minsarm):
        minsarm = sar[1]

    print(
        '{0:20}: SDR [{1[0]:05.2f}  {1[1]:05.2f}]  SIR [{2[0]:05.2f}  {2[1]:05.2f}]  SAR [{3[0]:05.2f}  {3[1]:05.2f}]'.format(
            song, sdr, sir, sar), file=file)

    avgsdr += sdr
    avgsir += sir
    avgsar += sar
avgsdr /= len(results)
avgsir /= len(results)
avgsar /= len(results)

# print(results, file=file)
print('\nAverage:\nSDR [{0[0]:05.2f}  {0[1]:05.2f}]\nSIR [{1[0]:05.2f}  {1[1]:05.2f}]\nSAR [{2[0]:05.2f}  {2[1]:05.2f}]'
    .format(avgsdr, avgsir, avgsar), file=file)

print('\nMaximums:\nSDR [{0:05.2f}  {1:05.2f}]\nSIR [{2:05.2f}  {3:05.2f}]\nSAR [{4:05.2f}  {5:05.2f}]'.format(
    maxsdrv, maxsdrm, maxsirv, maxsirm, maxsarv, maxsarm), file=file)

print('\nMinimums:\nSDR [{0:05.2f}  {1:05.2f}]\nSIR [{2:05.2f}  {3:05.2f}]\nSAR [{4:05.2f}  {5:05.2f}]'.format(
    minsdrv, minsdrm, minsirv, minsirm, minsarv, minsarm), file=file)
file.close()
