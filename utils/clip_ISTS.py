import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

from scipy.io import wavfile

if __name__ == '__main__':

    # reading the original ISTS signal
    fs, signal = wavfile.read('/Users/gustavocidornelas/Downloads/ISTS_V2 2/ISTS-V1.0_60s_16bit.wav')

    # segmenting the ISTS (mixed) signal
    seg_eng_1 = signal[:95937]
    seg_eng_2 = signal[95937:375318]
    seg_eng_3 = signal[375318:515722]
    seg_eng_4 = signal[515722:746299]
    seg_eng_5 = signal[746299:827213]
    seg_eng_6 = signal[827213:940187]
    seg_eng_7 = signal[940187:1036370]
    seg_eng_8 = signal[1036370:1158500]
    seg_eng_9 = signal[1158500:1247050]
    seg_eng_10 = signal[1247050:1460780]
    seg_eng_11 = signal[1460780:1659250]
    seg_eng_12 = signal[1659250:1773750]
    seg_eng_13 = signal[1773750:1836340]
    seg_eng_14 = signal[1836340:1915730]
    seg_eng_15 = signal[1915730:1976800]
    seg_eng_16 = signal[1976800:2060760]
    seg_eng_17 = signal[2060760:2187480]
    seg_eng_18 = signal[2187480:2254650]
    seg_eng_19 = signal[2254650:2303500]
    seg_eng_20 = signal[2303500:2358460]
    seg_eng_21 = signal[2358460:2457700]
    seg_eng_22 = signal[2457700:]

    np.save('ists_s1.npy', seg_eng_1)
    np.save('ists_s2.npy', seg_eng_2)
    np.save('ists_s3.npy', seg_eng_3)
    np.save('ists_s4.npy', seg_eng_4)
    np.save('ists_s5.npy', seg_eng_5)
    np.save('ists_s6.npy', seg_eng_6)
    np.save('ists_s7.npy', seg_eng_7)
    np.save('ists_s8.npy', seg_eng_8)
    np.save('ists_s9.npy', seg_eng_9)
    np.save('ists_s10.npy', seg_eng_10)
    np.save('ists_s11.npy', seg_eng_11)
    np.save('ists_s12.npy', seg_eng_12)
    np.save('ists_s13.npy', seg_eng_13)
    np.save('ists_s14.npy', seg_eng_14)
    np.save('ists_s15.npy', seg_eng_15)
    np.save('ists_s16.npy', seg_eng_16)
    np.save('ists_s17.npy', seg_eng_17)
    np.save('ists_s18.npy', seg_eng_18)
    np.save('ists_s19.npy', seg_eng_19)
    np.save('ists_s20.npy', seg_eng_20)
    np.save('ists_s21.npy', seg_eng_21)
    np.save('ists_s22.npy', seg_eng_22)



    # plt.figure()
    # plt.plot(range(seg_eng_1.shape[0]), seg_eng_1)
    # plt.figure()
    # plt.plot(range(seg_eng_2.shape[0]), seg_eng_2)
    # plt.figure()
    # plt.plot(range(seg_eng_3.shape[0]), seg_eng_3)
    # plt.figure()
    # plt.plot(range(seg_eng_4.shape[0]), seg_eng_4)
    # plt.figure()
    # plt.plot(range(seg_eng_5.shape[0]), seg_eng_5)
    # plt.figure()
    # plt.plot(range(seg_eng_6.shape[0]), seg_eng_6)
    # plt.figure()
    # plt.plot(range(seg_eng_7.shape[0]), seg_eng_7)
    # plt.figure()
    # plt.plot(range(seg_eng_8.shape[0]), seg_eng_8)
    # plt.figure()
    # plt.plot(range(seg_eng_9.shape[0]), seg_eng_9)
    # plt.figure()
    # plt.plot(range(seg_eng_10.shape[0]), seg_eng_10)
    # plt.figure()
    # plt.plot(range(seg_eng_11.shape[0]), seg_eng_11)
    # plt.figure()
    # plt.plot(range(seg_eng_12.shape[0]), seg_eng_12)
    # plt.figure()
    # plt.plot(range(seg_eng_13.shape[0]), seg_eng_13)
    # plt.figure()
    # plt.plot(range(seg_eng_14.shape[0]), seg_eng_14)
    # plt.figure()
    # plt.plot(range(seg_eng_15.shape[0]), seg_eng_15)
    # plt.figure()
    # plt.plot(range(seg_eng_16.shape[0]), seg_eng_16)
    # plt.figure()
    # plt.plot(range(seg_eng_17.shape[0]), seg_eng_17)
    # plt.figure()
    # plt.plot(range(seg_eng_18.shape[0]), seg_eng_18)
    # plt.figure()
    # plt.plot(range(seg_eng_19.shape[0]), seg_eng_19)
    # plt.figure()
    # plt.plot(range(seg_eng_20.shape[0]), seg_eng_20)
    # plt.figure()
    # plt.plot(range(seg_eng_21.shape[0]), seg_eng_21)
    # plt.figure()
    # plt.plot(range(seg_eng_22.shape[0]), seg_eng_22)
    # plt.show()



    # individual language ISTS signals
    fs, signal = wavfile.read('/Users/gustavocidornelas/Downloads/FR and NFIM 2/FR/FRenglish.wav')
    #sd.play(signal[0:241691])
    #sd.play(signal[241691:496260])
    #sd.play(signal[496260:756424])
    #sd.play(signal[756424:866924])
    #sd.play(signal[866924:947351])
    #sd.play(signal[947351:1041070])
    #sd.play(signal[1041070:1148770])
    #sd.play(signal[1148770:])

    # segmenting the full ISTS signal
    seg_eng_1 = signal[:241691]
    seg_eng_2 = signal[241691:496260]
    seg_eng_3 = signal[496260:756424]
    seg_eng_4 = signal[756424:866924]
    seg_eng_5 = signal[866924:947351]
    seg_eng_6 = signal[947351:1041070]
    seg_eng_7 = signal[1041070:1148770]
    seg_eng_8 = signal[1148770:]
    #
    # plt.figure()
    # plt.plot(range(seg_eng_1.shape[0]), seg_eng_1)
    # plt.figure()
    # plt.plot(range(seg_eng_2.shape[0]), seg_eng_2)
    # plt.figure()
    # plt.plot(range(seg_eng_3.shape[0]), seg_eng_3)
    # plt.figure()
    # plt.plot(range(seg_eng_4.shape[0]), seg_eng_4)
    # plt.figure()
    # plt.plot(range(seg_eng_5.shape[0]), seg_eng_5)
    # plt.figure()
    # plt.plot(range(seg_eng_6.shape[0]), seg_eng_6)
    # plt.figure()
    # plt.plot(range(seg_eng_7.shape[0]), seg_eng_7)
    # plt.figure()
    # plt.plot(range(seg_eng_8.shape[0]), seg_eng_8)
    # plt.show()