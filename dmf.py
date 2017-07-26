from horver import *
from diagfilt import *


def dmf(audio, rate):
    # sufiks h -za spektrogram visoke frenkvecijske rezolucije
    # l -... niske rezolucije
    fh, th, cspecth, specth = magspect(audio, rate, winlenh, (winlenh - steph) / winlenh)
    specth = specth.astype(np.float64)

    verbinh = hztobins(hzh, winlenh, rate)
    horbinh = sectobins(sech, steph, rate)
    vfh = vertfilter(specth, verbinh)
    hfh = horfilter(specth, horbinh)

    mph = vfh ** 2 / (hfh ** 2 + vfh ** 2) # maska za perkusije (i glas)

    vocper = inversestft(cspecth * mph, winlenh, (winlenh - steph) / winlenh) # vokali sa perkusijama

    fl, tl, cspectl, spectl = magspect(vocper, rate, winlenl, (winlenl - stepl) / winlenl)
    spectl = spectl.astype(np.float64)

    verbinl = hztobins(hzl, winlenl, rate)
    horbinl = sectobins(secl, stepl, rate)

    vfl = vertfilter(spectl, verbinl)
    hfl = horfilter(spectl, horbinl)

    d1 = diagfilter(spectl, 2, 1, horbinl)
    d2 = diagfilter(spectl, 1, 1, horbinl)
    d3 = diagfilter(spectl, 1, 2, horbinl)
    d4 = diagfilter(spectl, -2, 1, horbinl)
    d5 = diagfilter(spectl, -1, 1, horbinl)
    d6 = diagfilter(spectl, -1, 2, horbinl)

    hfl = np.max([hfl, d1, d2, d3, d4, d5, d6], axis=(0,1))
    mhl = hfl ** 2 / (vfl ** 2 + hfl ** 2)  # maska za glas

    voc = inversestft(cspectl * mhl, winlenl, (winlenl - stepl) / winlenl)[:len(audio)] # samo vokali
    mus = audio-voc # samo muzika

    return voc, mus
