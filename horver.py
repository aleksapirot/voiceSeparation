from common import *


# pretvara herce u broj binova stft-a
def hztobins(hz, l, rate):
    return round((hz * l / rate) / 2) * 2 + 1


# pretvara sekunde u broj binova stft-a
def sectobins(sec, step, rate):
    return round((sec * rate / step) / 2) * 2 + 1


# vertikalni medijanski filter
def vertfilter(spect, l):
    return signal.medfilt(spect, (l, 1))


# horizontalni medijanski filter
def horfilter(spect, l):
    return signal.medfilt(spect, (1, l))


# sufiks h -za spektrogram visoke frenkvecijske rezolucije
# l -... niske rezolucije
hzh = 20
hzl = 250
sech = 2
secl = 0.15

winlenh = 16384
steph = 2048
winlenl = 1024
stepl = 256


# ne koriste se dijagonalni filteri, samo horizontalni i vertikalni
def horver(audio, rate):
    fh, th, cspecth, specth = magspect(audio, rate, winlenh, (winlenh - steph) / winlenh)

    verbinh = hztobins(hzh, winlenh, rate)
    horbinh = sectobins(sech, steph, rate)
    vfh = vertfilter(specth, verbinh)
    hfh = horfilter(specth, horbinh)

    mph = vfh ** 2 / (hfh ** 2 + vfh ** 2) # maska za perkusije (i glas)

    vocper = inversestft(cspecth * mph, winlenh, (winlenh - steph) / winlenh) # vokali sa perkusijama


    fl, tl, cspectl, spectl = magspect(vocper, rate, winlenl, (winlenl - stepl) / winlenl)

    verbinl = hztobins(hzl, winlenl, rate)
    horbinl = sectobins(secl, stepl, rate)

    vfl = vertfilter(spectl, verbinl)
    hfl = horfilter(spectl, horbinl)
    mhl = hfl ** 2 / (vfl ** 2 + hfl ** 2) # maska za glas

    voc = inversestft(cspectl * mhl, winlenl, (winlenl - stepl) / winlenl)[:len(audio)] # samo vokali
    mus = audio-voc # samo muzika

    return voc, mus
