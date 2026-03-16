# -*- coding: UTF-8 -*-
"""
@Project : Algorithms 
@File    : fast_kurtogram.py
@IDE     : PyCharm 
@Author  : Zhuofu
@Date    : 2024/6/13 13:55 
"""


import numpy as np
import scipy.signal as si
import matplotlib.pyplot as plt
import logging


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
eps = np.finfo(float).eps


def nextpow2(n):
    """
    Calculates next power of 2.

    :param n: a number
    :type n: integer

    :rtype: integer
    :returns: The next power of 2
    """

    m_f = np.log2(n)
    m_i = int(np.ceil(m_f))
    return 2**m_i


def get_h_parameters(NFIR, fcut):
    """
    Calculates h-parameters used in Antoni (2005)

    :param NFIR: length of FIR filter
    :param fcut: fraction of Nyquist for filter

    :type NFIR: integer
    :type fcut: float

    :rtype: numpy array
    :returns: h-parameters: h, g, h1, h2, h3

    """

    h = si.firwin(NFIR+1, fcut) * np.exp(2*1j*np.pi*np.arange(NFIR+1) * 0.125)
    n = np.arange(2, NFIR+2)
    g = h[(1-n) % NFIR]*(-1.)**(1-n)
    NFIR = int(np.fix((3./2.*NFIR)))
    h1 = si.firwin(NFIR+1, 2./3*fcut)*np.exp(2j*np.pi*np.arange(NFIR+1) *
                                             0.25/3.)
    h2 = h1*np.exp(2j*np.pi*np.arange(NFIR+1)/6.)
    h3 = h1*np.exp(2j*np.pi*np.arange(NFIR+1)/3.)
    return (h, g, h1, h2, h3)


def plotKurtogram(Kwav, freq_w, nlevel, Level_w, Fs, fi, I):
    """
    Plots the kurtogram.

    :param Kwav: kurtogram
    :param freq_w: frequency vector
    :param nlevel: number of decomposition levels
    :param level_w: vector of levels
    :param Fs: sampling frequency of the signal
    :param fi:
    :param I: level index

    :type Kwav: numpy array
    :type freq_w: numpy array
    :type nlevel: integer
    :type level_w: numpy array
    :type Fs: integer
    :type fi:
    :type I: integer
    """

    plt.imshow(Kwav, aspect='auto', extent=(0, freq_w[-1], range(2*nlevel)[-1],
                                            range(2*nlevel)[0]),
               interpolation='none', cmap=plt.cm.hot_r)
    #imgplot.set_cmap('gray')
    # xx = np.arange(0, int(freq_w[len(freq_w)-1]), step=5)
    # plt.xticks(xx)
    plt.yticks(range(2*nlevel), np.round(Level_w*10)/10)
    plt.plot(Fs*fi, I, 'yo')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Level k")
    #plt.figtext(0.075, 0.90, "(a)", fontsize=15)
    plt.title("Level %.1f, Bw=%.2f Hz, fc=%.2f Hz" % (np.round(10*Level_w[I])/10, Fs*2**(-(Level_w[I]+1)), Fs*fi))
    plt.colorbar()
    plt.show()


def getBandwidthAndFrequency(nlevel, Fs, level_w, freq_w, level_index,
                             freq_index):
    """
    Gets bandwidth bw and frequency parameters knowing the
    level and the frequency indexes.

    :param nlevel: number of decomposition levels
    :param Fs: sampling frequency of the signal
    :param level_w: vector of decomposition levels
    :param freq_w: vector of frequencies
    :param level_index: index of the level
    :param freq_index: index of the frequency

    :type nlevel: integer
    :type Fs: integer
    :type level_w: numpy array
    :type freq_w: numpy array
    :type level_index: integer
    :type freq_index: integer

    :returns: bw, fc, fi, l1
        * bw: bandwidth
        * fc: central frequency
        * fi: index of the frequency sequence within the level l1
        * l1: level
    """

    #f1 = freq_w[freq_index]
    l1 = level_w[level_index]
    fi = (freq_index)/3./2**(nlevel+1)
    fi += 2.**(-2-l1)
    bw = Fs*2**-(l1)/2
    fc = Fs * fi

    return bw, fc, fi, l1


def get_GridMax(grid):
    """
    Gets maximum of a nD grid and its unraveled index

    :param grid: an nD-grid
    :type param: numpy array

    :returns:
    * M : grid maximum
    * index : index of maximum in unraveled grid
    """

    index = np.argmax(grid)
    M = np.amax(grid)
    index = np.unravel_index(index, grid.shape)

    return M, index


def Fast_Kurtogram(x, nlevel, verbose=False, Fs=1, NFIR=16, fcut=0.4,
                   opt1=1, opt2=1):
    """
    Computes the fast kurtogram Kwav of signal x up to level 'nlevel'
    Maximum number of decomposition levels is log2(length(x)), but it is
    recommended to stay by a factor 1/8 below this.
    Also returns the vector of k-levels Level_w, the frequency vector
    freq_w, the complex envelope of the signal c and the extreme
    frequencies of the "best" bandpass f_lower and f_upper.

    J. Antoni : 02/2005
    Translation to Python: T. Lecocq 02/2012

    :param x: signal to analyse
    :param nlevel: number of decomposition levels
    :param verbose: If ``True`` outputs debugging information
    :param Fs: Sampling frequency of signal x
    :param NFIR: Length of FIR filter
    :param fcut: Fraction of Nyquist for filter
    :param opt1: [1 | 2]:
        * opt1 = 1: classical kurtosis based on 4th order statistics
        * opt1 = 2: robust kurtosis based on 2nd order statistics of the
        envelope (if there is any difference in the kurtogram between the
        two measures, this is due to the presence of impulsive additive
        noise)
    :param opt2: [1 | 2]:
        * opt2=1: the kurtogram is computed via a fast decimated filterbank
        * opt2=2: the kurtogram is computed via the short-time Fourier
        transform (option 1 is faster and has more flexibility than option
        2 in the design of the analysis filter: a short filter in option 1
        gives virtually the same results as option 2)

    :type x: numpy array
    :type nlevel: integer
    :type Fs: integer
    :type NFIR: integer
    :type fcut: float

    :returns: Kwav, Level_w, freq_w, c, f_lower, f_upper
        * Kwav: kurtogram
        * Level_w: vector of levels
        * freq_w: frequency vector
        * c: complex envelope of the signal filtered in the frequency band that maximizes the kurtogram
        * f_lower: lower frequency of the band pass
        * f_upper: upper frequency of the band pass
    """

    N = len(x)
    N2 = np.log2(N) - 7
    if nlevel > N2:
        logging.error('Please enter a smaller number of decomposition levels')

    # Fast computation of the kurtogram
    ####################################

    if opt1 == 1:
        # 1) Filterbank-based kurtogram
        ############################
        # Analytic generating filters

        h, g, h1, h2, h3 = get_h_parameters(NFIR, fcut)

        if opt2 == 1:
            # kurtosis of the complex envelope
            Kwav = K_wpQ(x, h, g, h1, h2, h3, nlevel, verbose, 'kurt2')
        else:
            # variance of the envelope magnitude
            Kwav = K_wpQ(x, h, g, h1, h2, h3, nlevel, verbose, 'kurt1')

        # keep positive values only!
        Kwav[Kwav <= 0] = 0
        Level_w = np.arange(1, nlevel+1)
        Level_w = np.array([Level_w, Level_w + np.log2(3.)-1])
        Level_w = sorted(Level_w.ravel())
        Level_w = np.append(0, Level_w[0:2*nlevel-1])
        freq_w = Fs*(np.arange(0, 3*2.0**nlevel)/(3*2**(nlevel+1)) +
                     1.0/(3.*2.**(2+nlevel)))

        M, index = get_GridMax(Kwav)

        level_index = index[0]
        freq_index = index[1]
        bw, fc, fi, l1 = getBandwidthAndFrequency(nlevel, Fs, Level_w, freq_w,
                                                  level_index, freq_index)

        if verbose:
            logging.info("max kur:{}".format(M))
            plotKurtogram(Kwav, freq_w, nlevel, Level_w, Fs, fi, level_index)

    else:
        logging.error('stft-based is not implemented')

    # Signal filtering !
    c = []
    test = 1
    lev = l1
    while test == 1:
        test = 0
        c, s, threshold, Bw, fc = Find_wav_kurt(x, h, g, h1, h2, h3,
                                                nlevel, lev, fi, Fs=Fs,
                                                verbose=verbose)

    # Determine the lowest and the uppest frequencies of the bandpass
    f_lower = Fs*np.round((fc-Bw/2.)*10**3)/10**3
    f_upper = Fs*np.round((fc+Bw/2.)*10**3)/10**3

    return Kwav, Level_w, freq_w, c, f_lower, f_upper


def K_wpQ(x, h, g, h1, h2, h3, nlevel, verbose, opt, level=0):
    """
    Calculates the kurtosis K (2-D matrix) of the complete quinte wavelet packet
    transform w of signal x, up to nlevel, using the lowpass and highpass filters
    h and g, respectively. The WP coefficients are sorted according to the frequency
    decomposition. This version handles both real and analytical filters, but
    does not yield WP coefficients suitable for signal synthesis.

    J. Antoni : 12/2004
    Translation to Python: T. Lecocq 02/2012

    :param x: signal
    :param h: lowpass filter
    :param g: higpass filter
    :param h1: filter parameter returned by get_h_parameters
    :param h2: filter parameter returned by get_h_parameters
    :param h3: filter parameter returned by get_h_parameters
    :param nlevel: number of decomposition levels
    :param verbose: If ``True`` outputs debugging information
    :param opt: ['kurt1' | 'kurt2']
        * 'kurt1' = variance of the envelope magnitude
        * 'kurt2' = kurtosis of the complex envelope
    :param level: decomposition level for this call

    :type x: numpy array
    :type h: numpy array
    :type g: numpy array
    :type h1: numpy array
    :type h2: numpy array
    :type h3: numpy array
    :type nlevel: integer
    :type opt: string

    :returns: kurtosis

    """

    L = np.floor(np.log2(len(x)))
    if level == 0:
        if nlevel >= L:
            logging.error('nlevel must be smaller')
        level = nlevel
    x = x.ravel()
    KD, KQ = K_wpQ_local(x, h, g, h1, h2, h3, nlevel, verbose, opt, level)
    K = np.zeros((2*nlevel, 3*2**nlevel))

    K[0, :] = KD[0, :]
    for i in range(1, nlevel):
        K[2*i-1, :] = KD[i, :]
        K[2*i, :] = KQ[i-1, :]

    K[2*nlevel-1, :] = KD[nlevel, :]
    return K


def K_wpQ_local(x, h, g, h1, h2, h3, nlevel, verbose, opt, level):
    """
    Is a recursive funtion.
    Computes and returns the 2-D vector K, which contains the kurtosis value of the signal as
    well as the 2 kurtosis values corresponding to the signal filtered into 2 different
    band-passes.
    Also returns and computes the 2-D vector KQ which contains the 3 kurtosis values corresponding
    to the signal filtered into 3 different band-passes.

    :param x: signal
    :param h: lowpass filter
    :param g: highpass filter
    :param h1: filter parameter returned by get_h_parameters
    :param h2: filter parameter returned by get_h_parameters
    :param h3: filter parameter returned by get_h_parameters
    :param nlevel: number of decomposition levels
    :param verbose: If ``True`` outputs debugging information
    :param opt: ['kurt1' | 'kurt2']
        * 'kurt1' = variance of the envelope magnitude
        * 'kurt2' = kurtosis of the complex envelope
    :param level: decomposition level for this call

    :type x: numpy array
    :type h: numpy array
    :type g: numpy array
    :type h1: numpy array
    :type h2: numpy array
    :type h3: numpy array
    :type nlevel: integer
    :type opt: string
    :type level: integer

    :returns: K, KQ

    """

    a, d = DBFB(x, h, g)

    N = len(a)
    d = d*np.power(-1., np.arange(1, N+1)) # indices pairs multipliés par -1
    K1 = kurt(a[len(h)-1:], opt)
    K2 = kurt(d[len(g)-1:], opt)

    if level > 2:
        a1, a2, a3 = TBFB(a, h1, h2, h3)
        d1, d2, d3 = TBFB(d, h1, h2, h3)
        Ka1 = kurt(a1[len(h)-1:], opt)
        Ka2 = kurt(a2[len(h)-1:], opt)
        Ka3 = kurt(a3[len(h)-1:], opt)
        Kd1 = kurt(d1[len(h)-1:], opt)
        Kd2 = kurt(d2[len(h)-1:], opt)
        Kd3 = kurt(d3[len(h)-1:], opt)
    else:
        Ka1 = 0
        Ka2 = 0
        Ka3 = 0
        Kd1 = 0
        Kd2 = 0
        Kd3 = 0

    if level == 1:
        K = np.array([K1*np.ones(3), K2*np.ones(3)]).flatten()
        KQ = np.array([Ka1, Ka2, Ka3, Kd1, Kd2, Kd3])
    if level > 1:
        Ka, KaQ = K_wpQ_local(a, h, g, h1, h2, h3, nlevel, verbose, opt,
                              level-1)

        Kd, KdQ = K_wpQ_local(d, h, g, h1, h2, h3, nlevel, verbose, opt,
                              level-1)

        K1 = K1*np.ones(np.max(Ka.shape))
        K2 = K2*np.ones(np.max(Kd.shape))
        K12 = np.append(K1, K2)
        Kad = np.hstack((Ka, Kd))
        K = np.vstack((K12, Kad))

        Long = int(2./6*np.max(KaQ.shape))
        Ka1 = Ka1*np.ones(Long)
        Ka2 = Ka2*np.ones(Long)
        Ka3 = Ka3*np.ones(Long)
        Kd1 = Kd1*np.ones(Long)
        Kd2 = Kd2*np.ones(Long)
        Kd3 = Kd3*np.ones(Long)
        tmp = np.hstack((KaQ, KdQ))

        KQ = np.concatenate((Ka1, Ka2, Ka3, Kd1, Kd2, Kd3))
        KQ = np.vstack((KQ, tmp))

    if level == nlevel:
        K1 = kurt(x, opt)
        K = np.vstack((K1*np.ones(np.max(K.shape)), K))

        a1, a2, a3 = TBFB(x, h1, h2, h3)
        Ka1 = kurt(a1[len(h)-1:], opt)
        Ka2 = kurt(a2[len(h)-1:], opt)
        Ka3 = kurt(a3[len(h)-1:], opt)
        Long = int(1./3*np.max(KQ.shape))
        Ka1 = Ka1*np.ones(Long)
        Ka2 = Ka2*np.ones(Long)
        Ka3 = Ka3*np.ones(Long)
        tmp = np.array(KQ[0:-2])

        KQ = np.concatenate((Ka1, Ka2, Ka3))
        KQ = np.vstack((KQ, tmp))

    return K, KQ


def kurt(x, opt):
    """
    Calculates kurtosis of a signal according to the option chosen

    :param x: signal
    :param opt: ['kurt1' | 'kurt2']
        * 'kurt1' = variance of the envelope magnitude
        * 'kurt2' = kurtosis of the complex envelope

    :type x: numpy array
    :type opt: string

    :rtype: float
    :returns: Kurtosis
    """
    if opt == 'kurt2':
        if np.all(x == 0):
            K = 0
            E = 0
            return K
        x = x - np.mean(x)
        E = np.mean(np.abs(x)**2)
        if E < eps:
            K = 0
            return K

        K = np.mean(np.abs(x)**4)/E**2

        if np.all(np.isreal(x)):
            K = K - 3
        else:
            K = K - 2

    if opt == 'kurt1':
        if np.all(x == 0):
            K = 0
            E = 0
            return K
        x = x - np.mean(x)
        E = np.mean(np.abs(x))
        if E < eps:
            K = 0
            return K

        K = np.mean(np.abs(x)**2)/E**2

        if np.all(np.isreal(x)):
            K = K-1.57
        else:
            K = K-1.27

    return K


def DBFB(x, h, g):
    """
    Double-band filter-bank.
    [a,d] = DBFB(x,h,g) computes the approximation
    coefficients vector a and detail coefficients vector d,
    obtained by passing signal x though a two-band analysis filter-bank.

    :param x: signal
    :param h: The decomposition low-pass filter and
    :param g: The decomposition high-pass filter.

    :type x: numpy array
    :type h: numpy array
    :type g: numpy array

    :rtype: numpy array
    :returns: a, d

    """

    # lowpass filter
    a = si.lfilter(h, 1, x)
    a = a[1::2]
    a = a.ravel()

    # highpass filter
    d = si.lfilter(g, 1, x)
    d = d[1::2]
    d = d.ravel()

    return (a, d)


def TBFB(x, h1, h2, h3):
    """
    Triple-band filter-bank.
    [a1,a2,a3] = TBFB(x,h1,h2,h3)

    :param x: signal
    :param h1: filter parameter
    :param h2: filter parameter
    :param h3: filter parameter

    :type x: numpy array
    :type h1: numpy array
    :type h2: numpy array
    :type h3: numpy array

    :rtype: numpy array
    :returns: a1, a2, a3

    """

    # lowpass filter
    a1 = si.lfilter(h1, 1, x)
    a1 = a1[2::3]
    a1 = a1.ravel()

    # passband filter
    a2 = si.lfilter(h2, 1, x)
    a2 = a2[2::3]
    a2 = a2.ravel()

    # highpass filter
    a3 = si.lfilter(h3, 1, x)
    a3 = a3[2::3]
    a3 = a3.ravel()

    return (a1, a2, a3)


def Find_wav_kurt(x, h, g, h1, h2, h3, nlevel, Sc, Fr, Fs=1, verbose=False):
    """
    TODO flesh out this doc-string

    J. Antoni : 12/2004
    Translation to Python: T. Lecocq 02/2012

    :param x: signal
    :param h: lowpass filter
    :param g: highpass filter
    :param h1: filter parameter returned by get_h_parameters
    :param h2: filter parameter returned by get_h_parameters
    :param h3: filter parameter returned by get_h_parameters
    :param nlevel: number of decomposition levels
    :param Sc: Sc = -log2(Bw)-1 with Bw the bandwidth of the filter
    :param Fr: in the range [0, 0.5]
    :param Fs: Sampling frequency of signal x
    :param verbose: If ``True`` outputs debugging information

    :type x: numpy array
    :type h: numpy array
    :type g: numpy array
    :type h1: numpy array
    :type h2: numpy array
    :type h3: numpy array
    :type nlevel: integer
    :type Fr: float
    :type: Fs: integer

    :returns: c, s, threshold, Bw, fc
    """
    level = np.fix((Sc))+((Sc % 1) >= 0.5)*(np.log2(3)-1)
    Bw = 2**(-level-1)
    freq_w = np.arange(0, 2**level) / 2**(level+1) + Bw/2.
    J = np.argmin(np.abs(freq_w-Fr))
    fc = freq_w[J]
    i = int(np.round(fc/Bw-1./2))
    if level % 1 == 0:
        acoeff = binary(i, int(level))
        bcoeff = []
        temp_level = level
    else:
        i2 = int(np.fix((i/3.)))
        temp_level = np.fix((level))-1
        acoeff = binary(i2, int(temp_level))
        bcoeff = i-i2*3
    acoeff = acoeff[::-1]
    c = K_wpQ_filt(x, h, g, h1, h2, h3, acoeff, bcoeff, temp_level)

    t = np.arange(len(x))/float(Fs)
    tc = np.linspace(t[0], t[-1], len(c))
    s = np.real(c*np.exp(2j*np.pi*fc*Fs*tc))

    sig = np.median(np.abs(c))/np.sqrt(np.pi/2.)
    threshold = sig*raylinv(np.array([.999, ]), np.array([1, ]))

    return c, s, threshold, Bw, fc


def getFTSquaredEnvelope(c):
    """
    Calculates the Fourier transform of the squared envelope

    :param c: signal

    :returns S: FT of squared envelope

    """

    nfft = int(nextpow2(len(c)))
    env = np.abs(c)**2
    S = np.abs(np.fft.fft((env.ravel()-np.mean(env)) *
               np.hanning(len(env))/len(env), nfft))
    return S


def plot_envelope(x, Fs, c, fc, level, spec=False):
    """
    Plots envelope (with or without its spectrum)

    :param x: signal
    :param Fs: sampling frequency of signal
    :param c: complex envelope of signal
    :param fc: central frequency of the bandpass in Hz
    :param level: index of the decomposition level
    :param spec: If ``True`` also plots the envelope spectrum

    :type x: numpy array
    :type Fs: float
    :type c: numpy array
    :type fc: float
    :type level: float
    """

    fig = plt.figure()
    fig.set_facecolor('white')
    t = np.arange(len(x))/Fs
    tc = np.linspace(t[0], t[-1], len(c))
    plt.subplot(2+spec, 1, 1)
    plt.plot(t, x/np.max(x), 'k')
    plt.plot(tc, np.abs(c)/np.max(np.abs(c)), 'r')
    plt.title('Original Signal (4-10 Hz)')

    if spec:
        plt.subplot(3, 1, 2)
    else:
        plt.subplot(2, 1, 2)
    plt.plot(tc, np.abs(c), 'k')
    plt.title("Envelope of the filtered signal, Bw=Fs/2^%.1f, fc=%.2f Hz" %
              (np.round(level*10)/10, Fs*fc))
    plt.xlabel('time [s]')
    if spec == 1:
        nfft = int(nextpow2(len(c)))
        env = np.abs(c)**2
        S = np.abs(np.fft.fft(env.ravel()-np.mean(env) *
                   np.hanning(len(env))/len(env), nfft))
        f = np.linspace(0, 0.5*Fs/2**level, nfft/2)
        plt.subplot(313)
        plt.plot(f, S[:nfft/2], 'k')
        plt.title('Fourier transform magnitude of the squared envelope')
        plt.xlabel('frequency [Hz]')

    plt.show()


def binary(i, k):
    """
    Computes the coefficients of the binary expansion of i:
    i = a(1)*2^(k-1) + a(2)*2^(k-2) + ... + a(k)

    :param i: integer to expand
    :param k: nummber of coefficients

    :returns: coefficients a
    """

    if i >= 2**k:
        logging.error('i must be such that i < 2^k !!')

    a = np.zeros(k)
    temp = i
    for l in np.arange(k-1, -1, -1):
        a[k-l-1] = int(np.fix(temp/2**l))
        temp = temp - int(np.fix(a[k-l-1]*2**l))

    return a


def K_wpQ_filt(x, h, g, h1, h2, h3, acoeff, bcoeff, level=0):
    """
    Calculates the kurtosis K of the complete quinte wavelet packet transform w
    of signal x, up to nlevel, using the lowpass and highpass filters h and g,
    respectively. The WP coefficients are sorted according to the frequency
    decomposition.
    This version handles both real and analytical filters, but does not yield
    WP coefficients suitable for signal synthesis.

    J. Antoni : 12/2004
    Translation to Python: T. Lecocq 02/2012

    :param x: signal
    :param h: lowpass filter
    :param g: highpass filter
    :param h1: filter parameter returned by get_h_parameters
    :param h2: filter parameter returned by get_h_parameters
    :param h3: filter parameter returned by get_h_parameters
    :param acoeff:
    :param acoeff:
    :param level:

    """

    nlevel = len(acoeff)
    L = np.floor(np.log2(len(x)))
    if level == 0:
        if nlevel >= L:
            logging.error('nlevel must be smaller !!')
        level = nlevel
    x = x.ravel()
    if nlevel == 0:
        if bcoeff == []:
            c = x
        else:
            c1, c2, c3 = TBFB(x, h1, h2, h3)
            if bcoeff == 0:
                c = c1[len(h1)-1:]
            elif bcoeff == 1:
                c = c2[len(h2)-1:]
            elif bcoeff == 2:
                c = c3[len(h3)-1:]
    else:
        c = K_wpQ_filt_local(x, h, g, h1, h2, h3, acoeff, bcoeff, level)
    return c


def K_wpQ_filt_local(x, h, g, h1, h2, h3, acoeff, bcoeff, level):
    """
    Performs one analysis level into the analysis tree
    TODO : flesh out this doc-string

    :param x: signal
    :param h: lowpass filter
    :param g: higpass filter
    :param h1: filter parameter returned by get_h_parameters
    :param h2: filter parameter returned by get_h_parameters
    :param h3: filter parameter returned by get_h_parameters
    :param acoeff:
    :param acoeff:
    :param level:

    """

    a, d = DBFB(x, h, g)
    N = len(a)
    d = d*np.power(-1., np.arange(1, N+1))
    level = int(level)
    if level == 1:
        if bcoeff == []:
            if acoeff[level-1] == 0:
                c = a[len(h)-1:]
            else:
                c = d[len(g)-1:]
        else:
            if acoeff[level-1] == 0:
                c1, c2, c3 = TBFB(a, h1, h2, h3)
            else:
                c1, c2, c3 = TBFB(d, h1, h2, h3)
            if bcoeff == 0:
                c = c1[len(h1)-1:]
            elif bcoeff == 1:
                c = c2[len(h2)-1:]
            elif bcoeff == 2:
                c = c3[len(h3)-1:]
    if level > 1:
        if acoeff[level-1] == 0:
            c = K_wpQ_filt_local(a, h, g, h1, h2, h3, acoeff, bcoeff, level-1)
        else:
            c = K_wpQ_filt_local(d, h, g, h1, h2, h3, acoeff, bcoeff, level-1)

    return c


def raylinv(p, b):
    """
    Inverse of the Rayleigh cumulative distribution function (cdf).
    X = RAYLINV(P,B) returns the Rayleigh cumulative distribution function
    with parameter B at the probabilities in P.

    :param p: probabilities
    :param b:

    """

    # Initialize x to zero.
    x = np.zeros(len(p))
    # Return NaN if the arguments are outside their respective limits.
    k = np.where(((b <= 0) | (p < 0) | (p > 1)))[0]

    if len(k) != 0:
        tmp = np.NaN
        x[k] = tmp(len(k))

    # Put in the correct values when P is 1.
    k = np.where(p == 1)[0]
    if len(k) != 0:
        tmp = np.Inf
        x[k] = tmp(len(k))

    k = np.where(((b > 0) & (p > 0) & (p < 1)))[0]

    if len(k) != 0:
        pk = p[k]
        bk = b[k]
        x[k] = np.sqrt((-2*bk ** 2) * np.log(1 - pk))

    return x
