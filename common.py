import scipy.io.wavfile as wf
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import separation


# ucitava audio fajl
def load(path, mono=False):
    if path[-3:] == 'wav':
        (rate, audio) = wf.read(path)
    elif path[-3:] == 'mp3':
        import subprocess as sp
        rate = 44100
        command = ["ffmpeg",
                '-i', path,
                '-f', 'f32le',
                '-acodec', 'pcm_f32le',
                '-ar', str(rate),
                '-ac', '1',
                '-']
        pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.DEVNULL, bufsize=10 ** 8)
        audio = np.frombuffer(pipe.communicate()[0], dtype=np.float32)

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
def save(audio, rate, path, dtype=np.int16):
    # print(np.max(audio), np.min(audio))
    wf.write(open(path, 'wb+'), rate, audio.astype((dtype)))  # audio/np.max(np.abs(audio)))


# spectrogram
def magspect(audio, rate, winlen, noverlap=None):
    if noverlap is None:
        noverlap = winlen // 2
    f, t, spect = signal.stft(audio, fs=rate, noverlap=noverlap, window=signal.hamming(winlen, False), nperseg=winlen)

    cspect = spect  # kompleksan spektrogram
    spect = np.abs(spect)  # absolutna vrednost spektrograma
    return (f, t, cspect, spect)


# inverse stft
def inversestft(spect, winlen, noverlap=None):
    if noverlap is None:
        noverlap = winlen // 2
    return signal.istft(spect, nperseg=winlen, noverlap=noverlap, window=signal.hamming(winlen, False))[1]


# plotuje spectrogram
def plotspect(stft, maxcoef=0.8):
    f, t, spect = stft
    plt.pcolormesh(t, f, np.abs(spect), vmax=np.max(np.abs(spect)) * maxcoef)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.ylim((0, 10000))
    plt.xlabel('Time [sec]')
    plt.show()


# vraca SDR, SIR i SAR
def evaluate(origv, origm, algv, algm):
    return separation.bss_eval_sources(np.vstack((origv, origm)), np.vstack((algv, algm)), False)[0:3]
