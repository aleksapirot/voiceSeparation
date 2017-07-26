from repet import *


# beat spectrogram
# previse sporo radi
def beatspect(spect):
    spect = spect[:, -1500:-900]
    n=spect.shape[0]
    m=spect.shape[1]
    print(m)
    w=300
    V = np.ndarray((n,w))
    B = np.ndarray((w,m))
    A = np.ndarray((n,w))
    for j in range(m):
        for h in range(w):
            V[:,h]=spect[:, min(h+j-w//2, m-1)]**2
        for i in range(n):
            '''result = np.correlate(V[i], V[i], mode='full')
            result = result[result.size // 2:]
            A[i] = result * np.exp(np.linspace(0, 1.7, num=result.size))'''
            A[i] = autocorrel(V[i])
        B[:, j] = np.mean(A, axis=0)

        print('Gotov frame {}'.format(j))

    return B


# racuna periode za svaki beat spectrum iz spectrograma
# previse sporo radi
def periods(bspect):
    m = bspect.shape[1]
    p = np.empty(m)
    for j in range(m):
        p[j] = period(bspect[:, j])
    return p


# repeating spectrogram
def rspect(spect, ps):
    k=2
    n=spect.shape[0]
    m=spect.shape[1]
    U = np.ndarray((n,m))
    for j in range(m):
        U[:,j]=np.median(spect[:,j+ps*np.arange(1,k)-ps*k//2])
    return U



def adtrepet(audio,rate):
    return adaptiverepet(audio,rate,False)


def adaptiverepet(audio,rate, highpass):
    winlen = 1024
    f, t, spect = magspect(audio, rate, winlen=winlen)
    cspect = spect
    spect = np.abs(spect)
    # plotspect((f, t, spect))

    bt = beatspect(spect)
    #plt.plot(bt)
    #plt.show()

    ps = periods(bt)
    print(ps)

    rs = rspect(spect,ps)
    plt.plot(rs)
    plt.show()

    # nije gotovo
