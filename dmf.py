from common import *
from horver import *


# primenjuje dijag. filter
def diagfilter(spect, h, w):
    # TODO
    pass


# ne koriste se jos uvek dijagonalni filteri, samo horizontalni i vertiklani
def dmf(audio, rate):
    # sufiks h -za spektrogram visoke frenkvecijske rezolucije
    # l -... niske rezolucije
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

    #TODO dodati dijag. filtere i maske za njih

    voc = inversestft(cspectl * mhl, winlenl, (winlenl - stepl) / winlenl)[:len(audio)] # samo vokali
    mus = audio-voc # samo muzika

    return voc, mus
