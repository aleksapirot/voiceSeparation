from plca import *

#train()
rate, audio = load('../base/MIR-1K/Wavfile/amy_5_04.wav', mono=True)#load('../base/MIR-1K/Wavfile/amy_8_01.wav', mono=True)#load('../base/exyu/exyu_13.wav', mono=True)
print(label(audio, rate))