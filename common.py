import scipy.io.wavfile as wf
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import mir_eval as ev

# ucitava audio fajl
def load(path, mono = False):
    (rate, audio) = wf.read(path)
    if (audio.ndim == 2):
        audiol = audio[:, 0]
        audior = audio[:, 1]
        if mono:
            audio = audiol / 2 + audior / 2
            return (rate, np.trim_zeros(audio))
        else:
            return (rate, audiol, audior)
    else:
        return (rate, np.trim_zeros(audio))


# cuva audio fajl
def save(audio, rate, path):
    # print(np.max(audio), np.min(audio))
    wf.write(open(path, 'wb+'), rate, audio.astype(np.int16))#audio/np.max(np.abs(audio)))


# spectrogram
def magspect(audio, rate, winlen, overlap=0.5, square=False):
    f, t, spect = signal.stft(audio, fs=rate, noverlap=round(winlen * overlap), window=signal.hamming(winlen, False), nperseg=winlen)

    if (square):
        spect = np.square(np.abs(spect))
    cspect = spect # kompleksan spektrogram
    spect = np.abs(spect) # absolutna vrednost spektrograma
    return (f, t, cspect, spect)


# inverse stft
def inversestft(spect, winlen, overlap=0.5):
    return signal.istft(spect, nperseg=winlen, noverlap=round(winlen*overlap), window=signal.hamming(winlen, False))[1]


#plotuje spectrogram
def plotspect(stft, maxcoef=0.8):
    f, t, spect = stft
    plt.pcolormesh(t, f, np.abs(spect), vmax=np.max(np.abs(spect))*maxcoef)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.ylim((0, 10000))
    plt.xlabel('Time [sec]')
    plt.show()


# vraca SDR, SIR i SAR
def evaluate(origv, origm, algv, algm):
    return ev.separation.bss_eval_sources(np.vstack((origv, origm)), np.vstack((algv, algm)))[0:3]


