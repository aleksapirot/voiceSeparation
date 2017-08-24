import numpy as np

alg = 'dmf'
l = 20
sdrv = np.ndarray([l])
sdrm = np.ndarray([l])
sirv = np.ndarray([l])
sirm = np.ndarray([l])
sarv = np.ndarray([l])
sarm = np.ndarray([l])
file = open('../results/{}/metrics-fix.txt'.format(alg), 'r')
i = 0
for line in file.readlines()[:l]:  # 1000]:
    sdrv[i] = float(line[27:32])
    sdrm[i] = float(line[34:39])
    sirv[i] = float(line[47:52])
    sirm[i] = float(line[54:59])
    sarv[i] = float(line[67:72])
    sarm[i] = float(line[74:79])
    i += 1

avgsdr = [np.median(sdrv), np.median(sdrm)]
avgsir = [np.median(sirv), np.median(sirm)]
avgsar = [np.median(sarv), np.median(sarm)]
print('\nMedian:\nSDR [{0[0]:05.2f}  {0[1]:05.2f}]\nSIR [{1[0]:05.2f}  {1[1]:05.2f}]\nSAR [{2[0]:05.2f}  {2[1]:05.2f}]'
      .format(avgsdr, avgsir, avgsar))
