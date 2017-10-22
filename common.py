import scipy.io.wavfile as wf
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
from bss_eval import bss_eval_sources


# ucitava audio fajl
def load(path, mono=False):
    if path[-3:] == 'wav':
        (rate, audio) = wf.read(path)
    elif path[-3:] == 'mp3':
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
            dtype = audio.dtype
            audio = audiol / 2 + audior / 2
            return (rate, np.trim_zeros(audio.astype(dtype)))
        else:
            return (rate, audiol, audior)
    else:
        return (rate, np.trim_zeros(audio))


# cuva audio fajl
def save(audio, rate, path, mp3=True):
    if mp3:
        typetoformat = {np.int16:'s16le', np.int32:'s32le', np.float32:'f32le', np.float64:'f64le'}
        f = typetoformat[audio.dtype.type]
        command = ['ffmpeg',
                   '-y', # (optional) means overwrite the output file if it already exists.
                   "-f", f, # input format
                   "-acodec", "pcm_"+ f, # means raw  input
                   '-ar', str(rate),
                   '-ac','1', # 1 channel
                   '-i', '-', # means that the input will arrive from the pipe
                   '-vn', # means "don't expect any video input"
                   '-codec:a', "libmp3lame", # output audio codec
                   '-q:a', '3', # quality
                   path]
        pipe = sp.Popen(command, stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT)
        pipe.stdin.write(audio.tobytes())
        pipe.stdin.close()
    else:
        wf.write(open(path, 'wb+'), rate, audio)


def labels(file, segnum):
    lbl = '../base/MIR-1K/vocal-nonvocalLabel/' + file + '.vocal'
    lbl = open(lbl, 'r')
    lines = lbl.readlines()
    lbl.close()

    label = []
    for i in range(len(lines)//segnum):
        sum = 0
        for j in range(segnum):
            sum += int(lines[segnum*i+j])

        voice = 1 if sum > segnum/3 else 0
        label.append(voice)

    return label


# spectrogram
def magspect(audio, rate, winlen, noverlap=None):
    if noverlap is None:
        noverlap = winlen // 2
    f, t, spect = signal.stft(audio, fs=rate, noverlap=noverlap, window=signal.hamming(winlen, False), nperseg=winlen)

    cspect = spect  # kompleksan spektrogram
    spect = np.abs(spect)  # absolutna vrednost spektrograma
    return (f, t, cspect, spect)


def applymask(audio, spect, mask, winlen, noverlap=None, highpass=False, rate=0):
    if highpass:
        cutoff = 100 #Hz
        count = (2 * cutoff * mask.shape[0]) // rate
        mask[0:count, :] = np.zeros((count, mask.shape[1]))
    voice = inversestft(spect*mask, winlen, noverlap)[:len(audio)]
    music = audio-voice
    return voice, music


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
    return bss_eval_sources(np.vstack((origv, origm)), np.vstack((algv, algm)), False)[0:3]
