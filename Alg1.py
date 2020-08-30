#=================================================================
#==                    ALGORITHM 1 LIBRARY                      ==
#=================================================================



# IMPORTING LIBRARIES ============================================

import sys, importlib, os, time, datetime, math
from numba import njit, jit, prange

from  McsPy.McsData import RawData
import h5py

import matplotlib.style as pltstyle
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import numpy as np
import pandas as pd

import scipy
from scipy.signal import ellip, cheby1, bessel, butter, lfilter, filtfilt, iirfilter
import pywt
from tqdm import tqdm




# FILTERING ======================================================

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    INPUT:
     - lowcut = low cutting frequency (Hz)
     - highcut = high cutting frequency (Hz)
     - fs = sampling frequency (Hz)
     - order = order of the filter. The default is 5
    
    OUTPUT:
     - b, a = coefficients of the filter
    
    """
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a




def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Perfom the filtering of the data using a zero-phase Butterworth filter.
    
    INPUT:
     - data = The signal as a 1-dimensional numpy array
     - lowcut = low cutting frequency (Hz)
     - highcut = high cutting frequency (Hz)
     - fs = sampling frequency (Hz)
     - order = order of the filter. The dafault is 5
     
    OUTPUT:
     - y = signal filtered
    
    """
    
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y



def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n




def detect_ref_noise(ref, m1, m2, multiple_first_peak, multiple_second_peak, current_dir, fs, detect_second_peak= True, n=100, save=False):

    """
    INPUT:
    ref = reference channel that is needed by the function to detect the noise frequencies
    m1 = percentage increase of the first noise band for safety
    m2 =  percentage increase of the second noise band for safety (it is only used if  detect_second_peak is True)
    multiple_first_peak = when the threshold is obtained with the standard deviation, the function increments it by this factor
    multiple_second_peak = when the threshold is obtained with the standard deviation, the function increments it by this factor (it is only used if  detect_second_peak is True)
    Detect_second_peak = if True, the second peak is otained individually. If it is false, the function calculates its width as the double of the first peak.
    fs = samplig frequency
    n = number of samples of the moving average used to smooth the signal
    current_dir = folder in which save the file
    save = if True, saves the pictures in the current_dir folder


    OUTPUT:
    b_1 = numerator polynomial of the IIR filter for the first peak
    a_1 = denominator polynomial of the IIR filter for the first peak
    a_2 = numerator polynomial of the IIR filter for the second peak
    b_2 = denominator polynomial of the IIR filter for the second peak
    """

    fourier = np.fft.fft(ref) #FFT sul Ref
    freq = np.fft.fftfreq(ref.shape[0])*fs #indici delle frequenze (asse x --> campionamento asse)
    average = moving_average(abs(fourier), n=n) #Fa la media mobile sul Fourier


    #PRIMO PICCO
    top_1 = (np.argmax(average[:(int(len(average)/2))]))*fs/len(average) #Calcola la frequenza del primo picco
    index_top_1 = (np.argmax(average[:(int(len(average)/2))])) #Calcola l'indice del primo picco (= posizione all'interno dell'array)
    print("\nThe maximum peak is located at: % .2f"% ((np.argmax(average[:(int(len(average)/2))])+n)*fs/len(average)), " Hz" ) #stampa valore in hz del primo picco, considerando la traslazione della fft dovuta alla media mobile di n campioni
    threshold_1 = multiple_first_peak*np.mean(abs(average)) #Calcola la soglia che individua il picco
    threshold_2 = multiple_second_peak*np.mean(abs(average))
    print("\nThe threshold for the first peak is :", threshold_1)
    print("\nThe threshold for the second peak is :", threshold_2)
    check = 0
    #Li setto come valore iniziale pari allo stesso valore e poi nel ciclo while le modifico
    #continuo a traslarli a dx e sx di un incremento arbitrario finchè smettono di superare la soglia
    freq_min_1 = top_1
    freq_max_1 = top_1
    while check==0:
        if (average[int(freq_min_1*(len(average))/fs)] >= threshold_1):
            freq_min_1 = freq_min_1 - 0.1
        else:
            check = 1
    freq_min_corr = freq_min_1 + (fs*n/len(average)) #Correggo per la media mobile
    print("Minimum frequency: % .2f" %freq_min_corr, "Hz")

    check = 0
    while check==0:
        if (average[int(freq_max_1*(len(average))/fs)] >= threshold_1):
            freq_max_1 = freq_max_1 + 0.1
        else:
            check = 1
    freq_max_corr = freq_max_1 + (fs*n/len(average))
    print("Maximum frequency: % .2f" %freq_max_corr, "Hz")

    band_1 = freq_max_corr - freq_min_corr #Trovo la banda da arrestare

    #Incremento la banda del primo picco per il numero percentuale che ho dato in ingresso all'inizio (in modo simmetrico)
    freq_max_corr = freq_max_corr + (band_1/2) * m1
    freq_min_corr = freq_min_corr - (band_1/2) * m1

    print("Increase chosen for the first peak (m1) :", m1*100, " %")
    print("Increase for each side: % .2f" %((band_1/2)*m1), "Hz" )
    print("Stopped band: % .2f"% (freq_max_corr-freq_min_corr), "Hz, centred on: % .2f" %((freq_max_corr+freq_min_corr)/2), "Hz" )

    #Filtro il primo picco e poi uso questo segnale appena calcolato in ingresso ad una funzione analoga che mi filtra l'altro picco
    b_1,a_1 = scipy.signal.butter(4, [freq_min_corr, freq_max_corr], btype='bandstop', fs=fs)
    plotted_filtered_signal = scipy.signal.filtfilt(b=b_1, a=a_1, x=ref)


    # SECONDO PICCO
    if (detect_second_peak == True):
        top_2=(np.argmax(average[int(index_top_1*2 - (band_1*len(average))/fs) : int(index_top_1*2 + (band_1*len(average))/fs)])+
                       int(index_top_1*2 - (band_1*len(average))/fs))*fs/len(average)

        print("\nThe second peak is located at: % .2f" %((np.argmax(average[int(index_top_1*2 - (band_1*len(average))/fs) :
                        int(index_top_1*2 + (band_1*len(average))/fs)])+n+int(index_top_1*2 - (band_1*len(average))/fs))*fs/len(average)), " Hz" )

        freq_min_2 = top_2
        freq_max_2 = top_2
        check=0
        while check==0:
            if (average[int(freq_min_2*(len(average))/fs)] >= threshold_2):
                freq_min_2 = freq_min_2 - 0.1
            else:
                check = 1
        freq_min_corr_2 = freq_min_2 + (fs*n/len(average))

        check = 0
        while check==0:
            if (average[int(freq_max_2*(len(average))/fs)] >= threshold_2):
                freq_max_2 = freq_max_2 + 0.1
            else:
                check = 1
        freq_max_corr_2 = freq_max_2 + (fs*n/len(average))

        if (freq_max_2 == freq_min_2) & (freq_min_2 == top_2):
            print("No second peak has been detected because no points reach the secon threshold")
            graph_2 = False
            second_peak = False
        else:
            second_peak = True
            graph_2 = True
            print("Minimum frequency of the second peak: % .2f" %freq_min_corr_2, "Hz")
            print("Maximum frequency of the second peak: % .2f" %freq_max_corr_2, "Hz")
            band_2 = freq_max_corr_2 - freq_min_corr_2

            freq_max_corr_2 = freq_max_corr_2 + (band_2/2) * m2
            freq_min_corr_2 = freq_min_corr_2 - (band_2/2) * m2

            print("Chosen margin for the second peak (m2): ", m2*100, " %")
            print("Increase for each side: % .2f"% ((band_2/2)*m2), "Hz" )
            print("Stopped band: % .2f" %(freq_max_corr_2-freq_min_corr_2), "Hz, centrata su: % .2f"% ((freq_max_corr_2+freq_min_corr_2)/2), "Hz" )

            b_2,a_2 = scipy.signal.butter(4, [freq_min_corr_2, freq_max_corr_2], btype='bandstop', fs=fs)
            plotted_filtered_signal = scipy.signal.filtfilt(b=b_2, a=a_2, x=plotted_filtered_signal)



    else:
        print("\n\nDetect_second peak is false, so the second peak has been considered the double of the first peak in band and in frequencies")
        second_peak = True
        graph_2 = True
        top_2 = top_1*2
        freq_max_corr_2 = freq_max_corr * 2
        freq_min_corr_2 = freq_min_corr * 2
        band_2 = freq_max_corr_2 - freq_min_corr_2
        freq_min_2 = freq_min_1 * 2
        freq_max_2 = freq_max_1 * 2
        print("Stopped band : % .2f" %(freq_max_corr*2-freq_min_corr*2), "Hz, centred on: % .2f"% ((freq_max_corr*2 + freq_min_corr*2)/2), "Hz\n" )
        b_2,a_2 = scipy.signal.butter(4, [freq_min_corr*2, freq_max_corr*2], btype='bandstop', fs=fs)
        plotted_filtered_signal = scipy.signal.filtfilt(b=b_2, a=a_2, x=plotted_filtered_signal)


    final_fourier = np.fft.fft(plotted_filtered_signal)
    final_frequency = np.fft.fftfreq(plotted_filtered_signal.shape[0])*fs
    final_average = moving_average(abs(final_fourier), n=n)


    #GRAFICI

    dim = 100000        #dimensione della finestra della PSD (più larga implica più risoluzione ma anche più varianza)
    overlap= dim/2      #overlap al 50%, ma è arbitrario, si può porre a zero (overlap permette di ridurre la varianza)

    fig, ax = plt.subplots(2,2, figsize=(16,10))
    ax[0,0].plot(np.fft.fftshift(freq)[n-1:], abs(np.fft.fftshift(average)))
    ax[0,0].axis([freq_min_1-2*band_1,freq_max_1+2*band_1,0,200000])
    ax[0,0].axhline(threshold_1, 0,1, color="#FF0000")
    ax[0,0].set_title("FFT del primo picco individuato")
    ax[0,0].grid()

    ax[1,0].plot(np.fft.fftshift(final_frequency)[n-1:], abs(np.fft.fftshift((final_average))))
    ax[1,0].axis([freq_min_1-2*band_1,freq_max_1+2*band_1,0,200000])
    ax[1,0].set_title("FFT del primo picco filtrato")
    ax[1,0].grid()

    if graph_2 == True:
        ax[0,1].plot(np.fft.fftshift(freq)[n-1:], abs(np.fft.fftshift(average)))
        ax[0,1].axis([freq_min_2 - 3*band_2, freq_max_2 + 3*band_2, 0, 200000])
        if detect_second_peak == True:
            ax[0,1].axhline(threshold_2, 0,1, color="#FF0000")
        ax[0,1].set_title("FFT del secondo picco individuato")
        ax[0,1].grid()

        ax[1,1].plot(np.fft.fftshift(final_frequency)[n-1 :], abs(np.fft.fftshift(final_average)))
        ax[1,1].axis([freq_min_2 - 3*band_2, freq_max_2 + 3*band_2, 0, 200000])
        ax[1,1].set_title("FFT del secondo picco filtrato")
        ax[1,1].grid()
    if (save == True):
        file_name =current_dir + "/fft_noise_peaks.png"
        plt.savefig(file_name, dpi=300)

    plt.show()
    print("\n")

    fig, ax = plt.subplots(2,2, figsize=(16,12))
    ax[0,0].psd(ref,dim,Fs=10000,window=np.hamming(dim),noverlap=overlap)
    ax[0,0].axis([freq_min_corr-2.5*band_1,freq_max_corr+2.5*band_1,-50,10])
    ax[0,0].set_title("PSD del primo picco individuato")

    ax[1,0].psd(plotted_filtered_signal,dim,Fs=10000,window=np.hamming(dim),noverlap=overlap)
    ax[1,0].axis([freq_min_corr-2.5*band_1,freq_max_corr+2.5*band_1,-50,10])
    ax[1,0].set_title("PSD del primo picco filtrato")

    if graph_2 == True:
        ax[0,1].psd(ref,dim,Fs=10000,window=np.hamming(dim),noverlap=overlap)
        ax[0,1].axis([freq_min_corr_2-2.5*band_2,freq_max_corr_2+2.5*band_2,-60,15])
        ax[0,1].set_title("PSD del secondo picco individuato")

        ax[1,1].psd(plotted_filtered_signal,dim,Fs=10000,window=np.hamming(dim),noverlap=overlap)
        ax[1,1].axis([freq_min_corr_2-2.5*band_2,freq_max_corr_2+2.5*band_2,-60,15])
        ax[1,1].set_title("PSD del secondo picco filtrato")

    if (save == True):
        file_name =current_dir + "/psd_noise_peaks.png"
        plt.savefig(file_name, dpi=300)

    plt.show()

    if graph_2 == True:
        plt.figure(figsize=(16,8))
        plt.plot(np.fft.fftshift(freq)[n-1:], abs(np.fft.fftshift(average)), color='blue', label ='Originale', linewidth=1)
        plt.plot(np.fft.fftshift(final_frequency)[n-1:], abs(np.fft.fftshift(final_average)), color='red', label = 'Filtrato', linewidth=1)
        plt.axis([top_1-100, top_2+100 , 0, 200000])
        plt.grid()
        plt.legend(loc='upper right')
        plt.title("FFT sovrapposta")

    else:
        plt.figure(figsize=(16,8))
        plt.plot(np.fft.fftshift(freq)[n-1:], abs(np.fft.fftshift(average)), color='blue', label ='Originale', linewidth=1)
        plt.plot(np.fft.fftshift(final_frequency)[n-1:], abs(np.fft.fftshift(final_average)), color='red', label = 'Filtrato', linewidth=1)
        plt.grid()
        plt.legend(loc='upper right')
        plt.title("FFT sovrapposta")
        plt.axis([top_1-100, top_1+100 , 0, 200000])

    if (save == True):
        file_name =current_dir + "/fft_noise_complete_peaks.png"
        plt.savefig(file_name, dpi=300)

    plt.show()

    return a_1, b_1, a_2, b_2, second_peak







# THRESHOLD ======================================================

def ThresholdVariation(filtered_readings, save, current_dir, aligned_indexes, deviation_list, fs): 
    """
    See the effect of changing the multiple of the threshold
    """
    numero_spike = pd.DataFrame(data = 0, columns=filtered_readings.columns, index=['spikes'], dtype = "int32")
    i=0
    for electrode in (filtered_readings.columns):
        numero_spike[electrode] = len(aligned_indexes[i])
        i+=1
    canali = []
    numero_spike=numero_spike.sort_values(by='spikes', axis=1,  ascending=False)
    canali.append(numero_spike.columns[0])
    canali.append(numero_spike.columns[len(filtered_readings.columns)-1])
    canali.append(numero_spike.columns[int((len(filtered_readings.columns)-1)/2)])


    spike_plot = np.empty((3,82))
    i=0



    for j in tqdm(np.arange(0,8.2,0.1)):
        riga=0
        for electrode in (canali):
            s = (deviation_list[electrode])*j
            thr = float(s.values[0])
            sgn = filtered_readings[electrode].values
            spike_plot[riga, i] = len(DetectCrossings(signal = sgn, threshold = thr, fs = fs))
            riga += 1
        i+=1
    fig = plt.figure(figsize=(14,10))
    plt.rcParams.update({'font.size': 15})


    ax = fig.add_subplot(111)
    testo = ['max detected spikes', 'min detected spike', 'intermediate detected spike']
    for j in range(3):
        legenda = str(canali[j]) + ' - ' + testo[j]
        ax.plot(np.arange(0,8.2,0.1), spike_plot[j,:], label=legenda )
        ax.scatter([3,4,5], spike_plot[j,[30,40,50]])

    plt.legend(loc='upper right')
    ax.set_xticks(np.arange(0,8.5,0.5))
    plt.title("Spikes detected as a function of the multiple of the threshold")
    plt.ylabel('Detected spikes')
    plt.xlabel('Threshold multiple')
    plt.axis([0,8,0,np.amax(spike_plot)*(1.03)])
    plt.grid()
    print('\n')
    if (save == True):
        file_name = current_dir + "/changing_MAD_multiple.png"
        plt.savefig(file_name, dpi=300)

    plt.show()


    



# DETECT CROSSINGS, ALIGN AND EXTRACT SPIKES =====================

@njit
def DetectCrossings(signal, threshold, fs, dead_time = 0.003):   
    """
    INPUT:
    - signal = signal in which to search the peaks 
    - threshold = threshold to apply
    - fs = sample frequency
    - dead_time [optional] = time interval in which the function neglects other crossings after having found one. The default is 0.003
    
    OUTPUT:
    - spike_crossings[:k] = list containing the indexes of the signal to have exceeded the threshold
    """
    
    sample_num = len(signal)
    
    #To have both a positive and a negative threshold
    threshold = abs(threshold)
    
    spike_crossings = np.zeros_like(signal, )
    dead_samples = int(dead_time*fs)
    
    i, k = 0, 0
    while (i<sample_num):
        value = abs(signal[i])
        if (value>threshold):
            spike_crossings[k] = i
            k += 1
            i += dead_samples
        else:
            i += 1
    
    return spike_crossings[:k]




@njit
def ExtractWaveform(signal, indexes, fs, pre = 0.0017, post = 0.0018):
    """
    INPUT:
    - signal = signal from which to extract the cutouts
    - indexes = indexes that tells the function where to cut the signal
    - fs = sample frequency
    - pre [optional] = how many samples to cut before the given index. The default is 0.0017
    - post [optional] = how many samples to cut after the given index. The default is 0.0018
    
    OUTPUT:
    - cutout = array containing the cut signal
    - indexes = array containing the indexes the of the waveform 
                the function was able to cut entirely. 
                If a waveform exceeds the signal it is not cut.
    """
    
    #The extracted waveform will be inside this window
    pre = int(pre*fs)
    post = int(post*fs)
    
    cutout = np.empty((len(indexes), (pre+post)), dtype = np.int32)
    indexes_to_delete = np.empty(len(indexes), dtype = np.int32)
    
    #cycle to extract the spikes
    dim = signal.shape[0]
    
    #from 0 to the length of the array "indexes"
    k = 0 
    cut_beginning, cut_end = 0, 0
    
    for i in indexes:
        #verifing the selected window does not exit the given signal
        if (i-pre >= 0) and (i+post <= dim):
            cutout[k] = signal[(int(i)-pre):(int(i)+post)]         
        elif (i-pre <= 0):
            indexes_to_delete[cut_beginning+cut_end] = k
            cut_beginning += 1
        elif (i+post >= dim):
            indexes_to_delete[cut_beginning+cut_end] = k
            cut_end += 1
        k += 1
       
    if (cut_beginning + cut_end) != 0:
        indexes = np.delete(indexes, indexes_to_delete[:cut_beginning+cut_end])
   
    if cut_end != 0:
        return cutout[cut_beginning:-cut_end], indexes
    else:
        return cutout[cut_beginning:], indexes


    
    
@njit
def FlipArray(input_array):
    """
    INPUT: 
    - input_array = the array to flip
    
    OUTPUT:
    - new_array = the flipped array
    """
    
    new_array = np.empty(input_array.shape, dtype=input_array.dtype)
    arr_len = input_array.shape[0]
    for i in prange(arr_len):
        new_array[arr_len - i - 1] = input_array[i]

    return new_array




@njit(parallel=True)
def AlignToCrosscorrelation(signal, given_indexes, fs, pre = 0.000, post = 0.0015):
    """
    INPUT:
    - signal = the signal in which spikes are present
    - given_indexes = indexes to correct in order to have a better alignment (they can be the crossings with a given threshold)
    - fs = sample frequency
    - pre = time before the given_indexes to extract the waveforms. From these extracted waveforms crosscorrelation will be computed.
    - post = time after the given_indexes to extract the waveforms. From these extracted waveforms crosscorrelation will be computed.
    - align_minimum = the sample in which the minimum has to be in the final extraction
    
    OUTPUT:
    - aligned_indexes = indexes aligned to best crosscorrelation value
    """

    M, misaligned_indexes = ExtractWaveform(signal=signal, indexes=given_indexes, fs=fs, pre=pre, post=post)
    length = len(M)
    
    #A matrix that contains the maximum value of crosscorrelation betwwen two waveforms
    crosscorr_matrix_max = np.zeros((length,length))
    
    #A matrix that contains the value (indexes) needed to move the spike so that maximum crosscorrelation occures
    crosscorr_matrix_index = np.zeros((length,length))

    #Correction due to the fact the signals are moved in full mode
    correction = len(M[0]) - 1
    
    for i in range(length):
        for j in prange(0,i):
            cross_correlation = np.convolve(M[i],FlipArray(M[j]))
            crosscorr_matrix_max[i][j] = np.max(cross_correlation)
            crosscorr_matrix_index[i][j] = np.argmax(cross_correlation) - correction
    
    #This is a vector to keep track of the translations to maximise correlation
    diff_indexes = np.zeros(M.shape[0], dtype = np.int32)
        
    #Starting from the second element (using the first one as a reference for the others) and then moving in the lower triangular matrix
    i=1
    while i < length:
        diff_indexes[i] = crosscorr_matrix_index[i][np.argmax(crosscorr_matrix_max[i][:i])]
        crosscorr_matrix_index[:,i] = crosscorr_matrix_index[:,i] + diff_indexes[i]
        i+=1
        
    pre_aligned_indexes_1 = (misaligned_indexes + diff_indexes)
    
    pre_cut = 0.0015
    post_cut = 0.0025
    M_2, pre_aligned_indexes_2 = ExtractWaveform(signal=signal, indexes=pre_aligned_indexes_1, fs=fs, pre=pre_cut, post=post_cut)
    
    #Searching for the maximum activity
    c = np.zeros(int(pre_cut*fs+post_cut*fs))
    for i in range(M_2.shape[1]):
        c[i] = np.mean(np.abs(M_2[:,i]))
    max_activity = np.argmax(c) - pre_cut*fs
    
    aligned_indexes = pre_aligned_indexes_2 + max_activity 
    
    return aligned_indexes




def AlignToMinimum(signal, indexes, threshold, fs, research_time = 0.002):  
    """
    INPUT:
    - signal = the signal in which to search the peaks
    - indexes = threshold indexes obtained with DetectSpike
    - fs = Sampling frequency
    - research_time [optional] = time (in seconds) in which to search the minimum (default: 0.002)
    
    OUTPUT:
    - indexes_spike[:k] = indexes aligned to minimum if lower threshold is overcome (otherwise to maximum)
    """
    num_of_samples = len(signal)
    
    research_campioni = int(research_time*fs)
    aligned_indexes = [None] * num_of_samples
    threshold = abs(threshold)
    
    # m = indexes of aligned_indexes
    # i = to cycle through the given threshold indexes
    # k = to cycle through the selected window
    
    m=0
    for i in indexes:
        #initialization of k before starting to scan the window
        k = 0
        negative_peak = False  
        
        #if the window is contained in the signal
        if (i + research_campioni) <= num_of_samples:
            #Searching if there is at least one sample that overcomes the lower threshold in the window
            while (k<research_campioni):   
                if signal[i+k] < -threshold:
                    negative_peak = True
                    aligned_indexes[m] = i+k #saving the position of the first threshold overcoming
                    k+=1
                    break
                k+=1
                
            #If there is no negative peak that overcomes the lower threshold
            if negative_peak == False:
                aligned_indexes[m] = i #initializing at the first overcoming of the positive threshold
                k=0 #Reset k
                while (k<research_campioni):
                    if signal[i+k] > signal[aligned_indexes[m]]:
                        aligned_indexes[m] = i+k
                    k += 1     
            #If there's a negative peak overcoming the threshold
            else:
                while (k<research_campioni):
                    if signal[i+k] < signal[aligned_indexes[m]]:
                        aligned_indexes[m] = i+k
                    k += 1
            m +=1 
            
        else:
            break
            
    return aligned_indexes[:m]



def AlignToMinimumMaximum(signal, indexes, thr, fs, research_time = 0.002):  
    """
    INPUT:
    - signal = the signal in which there are the spikes
    - indexes = the detected threshold indexes (using DetectCrossings)
    - thr = the thresholds that detected the spikes
    - fs = sampling frequency
    - research_time [opzionale] = time (in seconds) in which to align the spikes (default: 0.002)
    
    OUTPUT:
    - spike_idx[:k] = returned list of aligned spike indexes 
    """
    
    #(Align to maximum if the spike is detected for +thr or to minimum for -thr)
    
    #counts the number of samples in the signal
    j = 0 
    sample_number = 0
    for j in signal:
        sample_number += 1
     
    #create some useful variables
    research_samples = int(research_time*fs)
    spike_idx = [None] * sample_number
    thr = abs(thr)
  
    m=0
    for i in indexes:
        #initialization of k for each window
        k = 0
        if (i + research_samples) < sample_number:
            spike_idx[m] = i
            if(signal[i] > thr):
                #cycle to compare values in the window
                while (k<research_samples):
                    if(signal[spike_idx[m]]<signal[i+k]):
                        spike_idx[m] = i+k
                    k += 1
            elif(signal[i] < -thr):
                #cycle to compare values in the window
                while (k<research_samples):
                    if(signal[spike_idx[m]]>signal[i+k]):
                        spike_idx[m] = i+k
                    k += 1
            
            m += 1
        else:
            break
    
    return spike_idx[:m]



@njit
def AlignToMidPoint(signal, indexes, fs, research_time = 0.002):  
    """
    INPUT:
    - signal = the signal in which there are the spikes
    - indexes = the detected threshold indexes (using DetectCrossings)
    - fs = sampling frequency
    - research_time [opzionale] = time (in seconds) in which to align the spikes (default: 0.002)
    
    OUTPUT:
    - spike_idx[:k] = returned list of aligned spike indexes 
    """
    
    #(Align to Mid point between the maximum and minimum of the spike detected in the research_time)
    
    #counting the number of samples in the signal
    sample_number = len(signal)
    
    #Creation of some useful variables
    research_samples = int(research_time*fs)
    spike_idx = np.zeros_like(signal, dtype=np.int32)
    
    m=0
    for i in indexes:
        #initialization of k for each window
        k = 0
        minimum_idx = i+k
        maximum_idx = i+k
        if (i + research_samples) <= sample_number:
            while (k<research_samples):   
                #Detecting the minimum in the window
                if signal[i+k] < signal[minimum_idx]:
                    minimum_idx = i+k
                #Detecting the maximum in the window
                if signal[i+k] > signal[maximum_idx]:
                    maximum_idx = i+k
                k +=1
                    
            #Mid point
            mid_point = np.round((maximum_idx+minimum_idx)/2)

            spike_idx[m] = mid_point
            m+=1
        
        else:
            break
             
    return spike_idx[:m]



@njit
def AlignToBarycenter(signal, indexes, fs, research_time = 0.002):  
    """
    INPUT:
    - signal = the signal in which there are the spikes
    - indexes = the detected threshold indexes (using DetectCrossings)
    - fs = sampling frequency
    - research_time [opzionale] = time (in seconds) in which to align the spikes (default: 0.002)
    
    OUTPUT:
    - spike_idx[:k] = returned list of aligned spike indexes 
    """
    
    #(Align to barycenter between the maximum and minimum detected in the research time)
    
    #counting the number of samples in the signal
    sample_number = len(signal)
    
    #creation of useful variables
    research_samples = int(research_time*fs)
    spike_idx = np.zeros_like(signal, dtype=np.int32)

    
    m=0
    for i in indexes:
        #initialization of k for each window
        k = 0
        minimum_idx = i+k
        maximum_idx = i+k
        if (i + research_samples) <= sample_number:
            while (k<research_samples):   
                #finding minimum in the window
                if signal[i+k] < signal[minimum_idx]:
                    minimum_idx = i+k
                #finding maximum in the window
                if signal[i+k] > signal[maximum_idx]:
                    maximum_idx = i+k
                k +=1
                    
            #Calc the barycenter between the found min and max using the abs of the waveform voltage as a weight
            
            #max before min
            if maximum_idx < minimum_idx:
                in_between_idx = maximum_idx
                numerator = 0
                denominator = 0
                while in_between_idx <= minimum_idx:
                    weight = abs(signal[in_between_idx])
                    numerator += in_between_idx*weight
                    denominator += weight
                    in_between_idx += 1
                barycenter = np.round(numerator/denominator)

            #Min before max
            else:
                in_between_idx = minimum_idx
                numerator = 0
                denominator = 0
                while in_between_idx <= maximum_idx:
                    weight = abs(signal[in_between_idx])
                    numerator += in_between_idx*weight
                    denominator += weight
                    in_between_idx += 1
                barycenter = np.round(numerator/denominator)

            spike_idx[m] = barycenter
            m+=1
        
        else:
            break
             
    return spike_idx[:m]




def ExtractMorphology(extracted_spikes, indexes, electrode):
    
    """
    INPUT:
    - spike = Spikes detected by the electrode and extracted by previous functions
    - indexes = Aligned Indexes
    OUTPUT:
    - morphology_df = Pandas Dataframe that contains the following properties: 
                    1. Top-Peak Amplitude
                    2. Bottom-Peak Amplitude
                    3. Top to bottom peak total amplitude
                    4. Max first derivative
                    5. Min first derivative
                    6. Max second derivative
                    7. Min second derivative
    """
    
    top_amp, bottom_amp, total_amp, max_der, min_der, max_value_der2, min_value_der2 = 0, 0, 0, 0, 0, 0, 0
    
    total_morphology = []
    
    for spike in extracted_spikes:
        
        spike_morphology = []
        
        #Top-Peak Amplitude
        spike_morphology.append(max(spike))
        
        #Bottom-Peak Amplitude
        spike_morphology.append(min(spike))
        
        #Top to bottom peak total amplitude
        spike_morphology.append(abs(spike_morphology[0]) + abs(spike_morphology[1]))
        
        #Derivatives
        der = np.diff(spike, 1)
        max_value_der = max(der)
        min_value_der = min(der)
        spike_morphology.append(max_value_der)
        spike_morphology.append(min_value_der)
        
        der2 =np.diff(der,1)
        max_value_der2 = max(der2)
        min_value_der2 = min(der2)
        spike_morphology.append(max_value_der2)
        spike_morphology.append(min_value_der2)
        
        #Width 1
        #spike_lenght = range(len(spike)-1)
        #crossings = [] 
        #for i in spike_lenght:
        #    if (spike[i] < 0 and spike[i+1] > 0) or (spike[i] > 0 and spike[i+1] < 0):
        #        crossings.append(i)
        #spike_morphology.append(max(crossings) - min(crossings))
        
        
        #Width 2 
        #(From min to max derivative) --> Non penso sia una buona idea (?)
        
        #max_index = np.where(der == max_value)
        #min_index = np.where(der == min_value)
        #width = Int(abs(max_index[0][0] - min_index[0][0]))
        #spike_morphology.append(width)
        
        
        total_morphology.append(spike_morphology)
      
    
    features = ["Top Amplitude", "Bottom Amplitude", "Total Amplitude", "Max 1 derivative", "Min 1 derivative",  "Max 2 derivative", "Min 2 derivative"]
    morphology = pd.DataFrame(data = total_morphology, columns=features, dtype = "float32")
    morphology.name = str(electrode)
    
    return morphology




# PRINCIPAL COMPONENT ANALYSIS ===================================

def PerformPCA(x, current_dir, electrode_name, n=3, show=True, spike_count = 300, save=False):
    """
    INPUT:
    - x = feature to process using Principal Component Analysis
    - current_dir = Path to save the images
    - electrode_name = Name of the electrode
    - n = number of dimensions to consider (it can either be 2 or 3)
    - show = boolean variable to show the images or not
    - save = boolean variable to save the images or not
    
    OUTPUT:
    - principal_dataframe = Pandas dataframe that contains the reduced characteristics
    """
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    #Standardizing data x
    standardized = StandardScaler().fit_transform(x)
    print("\nSignal standardized\nMean: ", np.mean(standardized), "\nVariance: ", np.std(standardized)**2, "\n")
    
    #Count the number of spikes present
    if len(x) < spike_count:
        spike_count = len(x)
    
    #3D
    if n==3:
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(standardized)
        principal_DataFrame = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2','PC3'])

        if show == True:
            fig = plt.figure(figsize=(20,10))
            
            #PCA 3D
            plt.rcParams.update(plt.rcParamsDefault)
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(principal_components[:,0], principal_components[:,1], principal_components[:,2], color='#000000', depthshade=True, lw=0)
            #WAVEFORMS
            pltstyle.use('seaborn-dark')
            plt.rcParams.update({'font.size': 20})
            ax2 = fig.add_subplot(122)
            ax2.grid()
            ax2.set_title("Primi " +  str(spike_count) + " spike dell'elettrodo " + electrode_name.decode(encoding='UTF-8'))
            ax2.set_xlabel("Campioni")
            ax2.set_ylabel("Tensione")
            i=0
            while i < spike_count:
                ax2.plot(x[i], color='#000000', alpha=0.2)
                i+=1
                
            if (save == True):
                file_name = current_dir + "/PCA_3D_" + str(electrode_name) + ".png"
                plt.savefig(file_name, dpi=300)
            plt.show()
        
    #2D    
    elif n==2:
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(standardized)
        principal_DataFrame = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])

        if show == True:
            fig = plt.figure(figsize = (20,10))
            
            #PCA 2D
            plt.rcParams.update(plt.rcParamsDefault)
            ax1 = fig.add_subplot(121) 
            ax1.set_xlabel('Principal Component 1', fontsize = 30)
            ax1.set_ylabel('Principal Component 2', fontsize = 30)
            x = principal_components[:,0]
            y = principal_components[:,1]
            ax1.scatter(x, y, color ="#000000")
            ax1.grid()
            
            #WAVEFORMS
            pltstyle.use('seaborn-dark')
            plt.rcParams.update({'font.size': 20})
            ax2 = fig.add_subplot(122)
            ax2.grid()
            ax2.set_title("Primi " +  str(spike_count) + " spike dell'elettrodo " + electrode_name.decode(encoding='UTF-8'))
            ax2.set_xlabel("Campioni")
            ax2.set_ylabel("Tensione")
            i=0
            while i < spike_count:
                ax2.plot(x[i], color='#000000', alpha=0.2)
                i+=1
            
            if (save == True):
                file_name = current_dir + "/PCA_2D_" + str(electrode_name) + ".png"
                plt.savefig(file_name, dpi=300) 
            plt.show()
    
    else:
        raise Exception("PCA funciton only work with 2 or 3 dimensions! n=2, n=3")
    
    return principal_DataFrame






# CLUSTERING =====================================================

def plot_dendrogram(clustering_model, **kwargs):
    
    from scipy.cluster.hierarchy import dendrogram
    
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(clustering_model.children_.shape[0])
    n_samples = len(clustering_model.labels_)
    for i, merge in enumerate(clustering_model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([clustering_model.children_, clustering_model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    


def perform_pca_HIERARCHICAL(cutouts, spike_list, fs, n_comp, electrode, current_dir, points, min_silhouette_score=0.3, ignorare=500, centroids=False, save=False):
    """

    1) Perform feature estraction using Principal Component Analysis (PCA) on cutouts (normalized).
    2) Perform Agglomerative clustering on the PCs

    Params:
     - cutouts:  spikes x samples numpy array
     - spike_list: indexes of the spikes
     - fs: The sampling frequency in Hz
     - show: if True plot the average shape of the spikes of the same cluster

    Return:
     - final_list: a list that contains an array of spikes for each cluster (neuron)

    """
    import sys,os
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from mpl_toolkits.mplot3d import Axes3D



    #Normalizzo i dati
    scale = StandardScaler()
    estratti_norm = scale.fit_transform(cutouts)
    print('Total spikes', estratti_norm.shape[0])
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)

    if len(spike_list)!=transformed.shape[0]:
        dif = len(spike_list)-transformed.shape[0]

    #Lista che contiene i silhouette score
    list_score = []

    #Plotta tutti i grafici compreso quello con numero di cluster = 1 (in quel caso senza silhouette score)
    iterazione = 1
    cluster_labels = []
    fig = plt.figure(figsize=(30,15))
    plt.rcParams.update({'font.size': 10})
    for n in range (2,6):

        model = AgglomerativeClustering(n_clusters=n, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None)
        cluster_labels.append(model.fit_predict(transformed))
        if (n != 1):
            silhouette_avg = silhouette_score(transformed, cluster_labels[n-2])
            print('\n______________________________________________________________________________________________________________')
            print("For", n,"clusters, the silhouette score is:", silhouette_avg)
            print('\n')
            list_score.append(silhouette_avg)

        #Plot PCA
        plt.rcParams.update(plt.rcParamsDefault)
        ax = fig.add_subplot(2, 4, iterazione, projection='3d')

        color = []
        for i in cluster_labels[n-2]:
            color.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][i])

        ax.scatter(transformed[:,0], transformed[:,1], transformed[:,2], c=color, alpha=0.8, s=10, marker='.')
        if centroids == True:
            print(model.cluster_centers_)
            ax.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1],model.cluster_centers_[:,2],s=300,  c='black', depthshade=False)



        #Plot average waveforms
        pltstyle.use('seaborn-dark')
        ax2 = fig.add_subplot(2, 4, iterazione+4)
        for i in range(n):
            idx = cluster_labels[n-2] == i
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
            mean_wave = np.mean(cutouts[idx,:],axis = 0)
            std_wave = np.std(cutouts[idx,:],axis = 0)
            ax2.errorbar(range(cutouts[idx,:].shape[1]),mean_wave,yerr = std_wave)
            ax2.set_title("Silhouette Score = " + str(list_score[n-2]))
            ax2.grid(b=True, which='both', color='white', linestyle='-')

        plt.xlabel('Time [0.1ms]')
        plt.ylabel('Voltage [\u03BCV]')
        iterazione += 1

    if (save == True):
        file_name = current_dir + "/" + str(electrode) + "_clustering_attempts" + ".png"
        plt.savefig(file_name, dpi=300)

    plt.show()

    top_clusters = list_score.index(max(list_score))+2

    print("\n\n\033[1;31;47mBest cluster in the range 2 to 5: ",top_clusters,", with a silhouette score of: ",max(list_score), "\u001b[0m  \n\n")
    print('Spike list: ',len(spike_list))


    transformed_dendrogram = AgglomerativeClustering(n_clusters=None, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=0)
    model_dendrogram = transformed_dendrogram.fit(transformed)
    plt.figure(figsize=[8,6])
    plt.grid(color="white")
    plot_dendrogram(model_dendrogram, truncate_mode='level', p=3)
    
    if (save == True):
        file_name = current_dir + "/" + str(electrode) + "_dendrogram" + ".png"
        plt.savefig(file_name, dpi=200)
    
    plt.show()
    final_list = []

    #If the silhouette score is less than ideal
    if (max(list_score)<min_silhouette_score):

        print("Maximum silhouette score is less then the masximum thresold chosen")

        plt.figure(figsize=(16,9))
        plt.rcParams.update({'font.size': 15})
        plt.title("Forme d'onda medie interpolate")
        plt.grid(color="white")
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        mean_wave = np.mean(cutouts,axis = 0)
        interp = scipy.interpolate.CubicSpline(np.arange(points), mean_wave)
        dominio = np.arange(0,points,0.05)
        appross = interp(dominio)
        plt.plot(appross)
        plt.scatter(np.arange(points)*20, mean_wave, marker='.', s=70)
        plt.show()

        plt.figure(figsize=(16,9))
        plt.grid(color="white")
        plt.rcParams.update({'font.size': 15})
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        j=0
        while j<300 and j<len(cutouts[idx,:]):
            plt.plot(cutouts[j], color = color, alpha=0.3)
            j+=1

        plt.title("Primi 300 spike di ogni gruppo estratti e classificati")
        #plt.grid()
        if (save == True):
            file_name = current_dir + "/" + str(electrode) + "_first_300_spikes" + ".png"
            plt.savefig(file_name, dpi=200)

        plt.show()
        
        
        final_list.append(spike_list)
        return final_list




    list_idx = list(np.unique(cluster_labels[top_clusters-2]))

    #Forme d'onda interpolate
    plt.figure(figsize=(16,9))
    plt.rcParams.update({'font.size': 15})
    plt.title("Forme d'onda medie interpolate")
    plt.grid(color="white")
    for i in range(top_clusters):
        #plt.grid(b=True)
        idx = cluster_labels[top_clusters-2] == i
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        mean_wave = np.mean(cutouts[idx,:],axis = 0)
        interp = scipy.interpolate.CubicSpline(np.arange(points), mean_wave)
        dominio = np.arange(0,points,0.05)
        appross = interp(dominio)
        plt.plot(appross)
        plt.scatter(np.arange(points)*20, mean_wave, marker='.', s=70)

    if (save == True):
        file_name = current_dir + "/" + str(electrode) + "_best_attempt" + ".png"
        plt.savefig(file_name, dpi=300)

    plt.show()

    if len(spike_list)!=transformed.shape[0]:
        spike_list = spike_list[:-dif]

    for i in list_idx:
        if (len(cluster_labels[top_clusters-2]==i) > ignorare):
            final_list.append(spike_list[cluster_labels[top_clusters-2]==i])
        else:
            print("Cluster escluso per numero ridotto di spike")

    plt.figure(figsize=(16,9))
    plt.grid(color="white")
    plt.rcParams.update({'font.size': 15})
    for i in range(top_clusters):
        idx = cluster_labels[top_clusters-2] == i
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        j=0

        cluster_spikes = cutouts[idx,:]
        while j<300 and j<len(cutouts[idx,:]):
            plt.plot(cluster_spikes[j,:], color = color, alpha=0.3)
            j+=1

    plt.title("Primi 300 spike di ogni gruppo estratti e classificati")
    #plt.grid()
    if (save == True):
        file_name = current_dir + "/" + str(electrode) + "_first_300_spikes" + ".png"
        plt.savefig(file_name, dpi=200)

    plt.show()

    return final_list






def perform_pca_KMEANS(cutouts, spike_list, fs, n_comp, electrode, current_dir, points, min_silhouette_score=0.3, ignorare=500, centroids=False, save=False):
    """
    1) Perform feature estraction using Principal Component Analysis (PCA) on cutouts (normalized).
    2) Perform Agglomerative clustering on the PCs
    
    INPUT:
     - cutouts:  spikes x samples numpy array
     - spike_list: indexes of the spikes
     - fs: The sampling frequency in Hz
     - show: if True plot the average shape of the spikes of the same cluster

    OUTPUT:
     - final_list: a list that contains an array of spikes for each cluster (neuron)

    """
    import sys,os
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from mpl_toolkits.mplot3d import Axes3D

    #Normalizzo i dati
    scale = StandardScaler()
    estratti_norm = scale.fit_transform(cutouts)
    print('Total spikes', estratti_norm.shape[0])
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)

    if len(spike_list)!=transformed.shape[0]:
        dif = len(spike_list)-transformed.shape[0]

    #Lista che contiene i silhouette score
    list_score = []

    #Plotta tutti i grafici compreso quello con numero di cluster = 1 (in quel caso senza silhouette score)
    iterazione = 1
    cluster_labels = []
    fig = plt.figure(figsize=(26,13))
    plt.rcParams.update({'font.size': 10})
    for n in range (2,6):
        model = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=400, tol=0.00005, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto')
        cluster_labels.append(model.fit_predict(transformed))
        if (n != 1):
            silhouette_avg = silhouette_score(transformed, cluster_labels[n-2])
            print('\n______________________________________________________________________________________________________________')
            print("For", n,"clusters, the silhouette score is:", silhouette_avg)
            print('\n')
            list_score.append(silhouette_avg)

        #Plot PCA
        plt.rcParams.update(plt.rcParamsDefault)
        ax = fig.add_subplot(2, 4, iterazione, projection='3d')

        color = []
        for i in cluster_labels[n-2]:
            color.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][i])

        ax.scatter(transformed[:,0], transformed[:,1], transformed[:,2], c=color, alpha=0.8, s=10, marker='.')
        if centroids == True:
            print(model.cluster_centers_)
            ax.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1],model.cluster_centers_[:,2],s=300,  c='black', depthshade=False)



        #Plot average waveforms
        pltstyle.use('seaborn-dark')
        ax2 = fig.add_subplot(2, 4, iterazione+4)
        for i in range(n):
            idx = cluster_labels[n-2] == i
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
            mean_wave = np.mean(cutouts[idx,:],axis = 0)
            std_wave = np.std(cutouts[idx,:],axis = 0)
            ax2.errorbar(range(cutouts[idx,:].shape[1]),mean_wave,yerr = std_wave)
            ax2.set_title("Silhouette Score = " + str(list_score[n-2]))
            ax2.grid(b=True, which='both', color='white', linestyle='-')

        plt.xlabel('Time [0.1ms]')
        plt.ylabel('Voltage [\u03BCV]')
        iterazione += 1


    if (save == True):
        file_name = current_dir + "/" + str(electrode) + "_clustering_attempts" + ".png"
        plt.savefig(file_name, dpi=300)

    plt.show()


    top_clusters = list_score.index(max(list_score))+2

    print("\n\n\033[1;31;47mBest cluster in the range 2 to 5: ",top_clusters,", with a silhouette score of: ",max(list_score), "\u001b[0m  \n\n")

    print('Spike list: ',len(spike_list))

    final_list = []

    #If the silhouette score is less than ideal
    if (max(list_score)<min_silhouette_score):

        print("Maximum silhouette score is less then the masximum thresold chosen")

        plt.figure(figsize=(16,9))
        plt.rcParams.update({'font.size': 15})
        plt.title("Forme d'onda medie interpolate")
        plt.grid(color="white")
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        mean_wave = np.mean(cutouts,axis = 0)
        interp = scipy.interpolate.CubicSpline(np.arange(points), mean_wave)
        dominio = np.arange(0,points,0.05)
        appross = interp(dominio)
        plt.plot(appross)
        plt.scatter(np.arange(points)*20, mean_wave, marker='.', s=70)
        plt.show()

        plt.figure(figsize=(16,9))
        plt.grid(color="white")
        plt.rcParams.update({'font.size': 15})
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        j=0
        while j<300 and j<len(cutouts[idx,:]):
            plt.plot(cutouts[j], color = color, alpha=0.3)
            j+=1

        plt.title("Primi 300 spike di ogni gruppo estratti e classificati")
        plt.grid()
        if (save == True):
            file_name = current_dir + "/" + str(electrode) + "_first_300_spikes" + ".png"
            plt.savefig(file_name, dpi=200)

        plt.show()
        
        final_list.append(spike_list)
        return final_list



    list_idx = list(np.unique(cluster_labels[top_clusters-2]))

    #Forme d'onda interpolate
    plt.figure(figsize=(16,9))
    plt.rcParams.update({'font.size': 15})
    plt.title("Forme d'onda medie interpolate")
    plt.grid()
    for i in range(top_clusters):
        plt.grid(b=True)
        idx = cluster_labels[top_clusters-2] == i
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        mean_wave = np.mean(cutouts[idx,:],axis = 0)
        interp = scipy.interpolate.CubicSpline(np.arange(points), mean_wave)
        dominio = np.arange(0,points,0.05)
        appross = interp(dominio)
        plt.plot(appross, color=color)
        plt.scatter(np.arange(points)*20, mean_wave, marker='.', s=70)

    if (save == True):
        file_name = current_dir + "/" + str(electrode) + "_best_attempt" + ".png"
        plt.savefig(file_name, dpi=300)

    plt.show()

    if len(spike_list)!=transformed.shape[0]:
        spike_list = spike_list[:-dif]

    for i in list_idx:
        if (len(cluster_labels[top_clusters-2]==i) > ignorare):
            final_list.append(spike_list[cluster_labels[top_clusters-2]==i])
        else:
            print("Cluster escluso per numero ridotto di spike")

    plt.figure(figsize=(16,9))
    plt.rcParams.update({'font.size': 15})
    for i in range(top_clusters):
        idx = cluster_labels[top_clusters-2] == i
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        j=0

        cluster_spikes = cutouts[idx,:]
        while j<300 and j<len(cutouts[idx,:]):
            plt.plot(cluster_spikes[j,:], color = color, alpha=0.3)
            j+=1

    plt.title("Primi 300 spike di ogni gruppo estratti e classificati")
    plt.grid()
    if (save == True):
        file_name = current_dir + "/" + str(electrode) + "_first_300_spikes" + ".png"
        plt.savefig(file_name, dpi=200)

    plt.show()

    return final_list





def perform_pca_DBSCAN(cutouts, spike_list, fs, n_comp, electrode, current_dir, points, save=False, distanza = 1, punti_min = 15, ignorare = 500):
    """
    1) Perform feature estraction using Principal Component Analysis (PCA) on cutouts (normalized).
    2) Perform Agglomerative clustering on the PCs
    
    INPUT:
     - cutouts:  spikes x samples numpy array
     - spike_list: indexes of the spikes
     - fs: The sampling frequency in Hz
     - show: if True plot the average shape of the spikes of the same cluster
     
    OUTPUT:
     - final_list: a list that contains an array of spikes for each cluster (neuron) 
    """
    
    import sys,os
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    from mpl_toolkits.mplot3d import Axes3D
    import sklearn.preprocessing as ps

    scale = ps.StandardScaler()
    estratti_norm = scale.fit_transform(cutouts)
    
    print('Total spikes', estratti_norm.shape[0])
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)
    
    if len(spike_list)!=transformed.shape[0]:
        dif = len(spike_list)-transformed.shape[0]
    
    list_score = []   # List that contains the results of the silhouette score for each cluster
    
    model = DBSCAN(eps=distanza, min_samples=punti_min, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
    cluster_labels = model.fit_predict(transformed)
    
    out=0
    a=0
    b=0
    c=0
    d=0
    e=0
    f=0
    
    for i in range(cluster_labels.shape[0]):
        if (cluster_labels[i] == -1):
            out+=1
        elif (cluster_labels[i] == 0):
            a +=1
        elif (cluster_labels[i] == 1):
            b+=1        
        elif (cluster_labels[i] == 2):
            c+=1
        elif (cluster_labels[i] == 3):
            d+=1
        elif (cluster_labels[i] == 4):
            e+=1
        elif (cluster_labels[i] == 5):
            f+=1

    if(out!=0):
        print('\nSpike detected as noise', out)
    else:
        print('\nNo spike detected as noise')
        
    check=0
    
    color = []
    for i in cluster_labels:
        color.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
        if i==-1:
            color[i]='k'

    indici=cluster_labels
    coordinate=transformed  
    if(b!=0):
        for i in reversed(range(indici.shape[0])):
            if (indici[i] == -1):
                #print(coordinate[i])
                coordinate = np.delete(coordinate, i, axis=0)
                indici = np.delete(indici, i)
        
        print("Residual noise in silhouette score (if this is not zero there is a bug): ", np.count_nonzero(indici == -1))
        silhouette_avg = silhouette_score(coordinate, indici)
        print("\nNumber of clusters: ", len(set(cluster_labels))-1,"\nThe silhouette score is:", silhouette_avg)
        list_score.append(silhouette_avg)
        check=1

    if check==0:
        print('\nOnly one cluster detected')
    if(a!=0):
        print('\nBlue spikes:', a)
    if(b!=0):
        print('\nOrange spikes:', b)
    if(c!=0):
        print('\nGreen spikes:', c)
    if(d!=0):
        print('\nRed spikes:', d)
    if(e!=0):
        print('\nPurple spikes:', e)
    if(f!=0):
        print('\nBrown spikes:', f)

    #Plot PCA
    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure(figsize=(18,8))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(transformed[:,0], transformed[:,1], transformed[:,2], c=color, alpha=0.8, s=10, marker='.')
    
    #Plot average waveform
    pltstyle.use('seaborn-dark')
    ax = fig.add_subplot(1, 2, 2)
    for i in range(len(set(cluster_labels))-1):
        idx = cluster_labels == i
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        mean_wave = np.mean(cutouts[idx,:],axis = 0)
        std_wave = np.std(cutouts[idx,:],axis = 0)
        ax.grid(b=True, which='both', color='white', linestyle='-')
        ax.errorbar(range(cutouts[idx,:].shape[1]),mean_wave,yerr = std_wave)

    if (save == True):
        file_name = current_dir + "/" + str(electrode) + "_clustering_resuts" + ".png"
        plt.savefig(file_name, dpi=300)
        
    plt.xlabel('Time [0.1ms]')
    plt.ylabel('Voltage [\u03BCV]')
    plt.show()    
        
    list_idx = list(np.unique(cluster_labels))
    final_list = []
    
    if len(spike_list)!=transformed.shape[0]:
        spike_list = spike_list[:-dif]
    
    for i in list_idx:
        if i != -1:
            if len(spike_list[cluster_labels==i]) > ignorare:
                final_list.append(spike_list[cluster_labels==i])
        
    
    
    plt.figure(figsize=(16,9))
    plt.rcParams.update({'font.size': 15})
    plt.title("Forme d'onda medie interpolate")    
    plt.grid()
    for i in range(len(set(cluster_labels))-1):
        plt.grid(b=True)
        idx = cluster_labels == i
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        mean_wave = np.mean(cutouts[idx,:],axis = 0)
        interp = scipy.interpolate.CubicSpline(np.arange(points), mean_wave)
        dominio = np.arange(0,points,0.05)
        appross = interp(dominio)
        plt.plot(appross)
        plt.scatter(np.arange(points)*20, mean_wave, marker='.', s=70)
    
    if (save == True):
        file_name = current_dir + "/" + str(electrode) + "_best_attempt" + ".png"
        plt.savefig(file_name, dpi=300)
    
    plt.show()
    
    plt.figure(figsize=(16,9))
    plt.rcParams.update({'font.size': 15})
    for i in range(len(set(cluster_labels))-1):
        idx = cluster_labels == i
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        j=0
        
        cluster_spikes = cutouts[idx,:]
        while j<300 and j<len(cutouts[idx,:]):
            plt.plot(cluster_spikes[j,:], color = color, alpha=0.3)
            j+=1
        
    plt.title("Primi 300 spike di ogni gruppo estratti e classificati")
    plt.grid()
    if (save == True):
        file_name = current_dir + "/" + str(electrode) + "_first_300_spikes" + ".png"
        plt.savefig(file_name, dpi=200)
    
    plt.show()
    

    return final_list
