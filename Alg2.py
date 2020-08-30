#=================================================================
#==                    ALGORITHM 2 LIBRARY                      ==
#=================================================================


# IMPORTING LIBRARIES ============================================

import sys, importlib, os, time
from  McsPy.McsData import RawData
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.style as pltstyle
import seaborn as sns
import h5py
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import pywt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from numba import njit, jit
import math





# MOVING AVARAGE =================================================

def moving_average_analysis(a, n) :
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return np.array(ret[n - 1:] / n, dtype=np.float32)
    



# ACTIVITY BARPLOT ===============================================

def SpikeNumberInWindow(max_n, window, duration, cluster, electrodes_name, legend, fs, save, current_dir):
    """
    INPUT:
    - max_n = maximum number of pictures to plot in a square - it must be a perfect square (to not exceed your computer limitations)
    - window = number of seconds in a single window
    - duration = duration of the total signal
    - cluster = clustered neurons
    - electrodes_name = name of electrodes that measured the signal
    - legend = pandas dataframe containing a link between the electrode name and the electrode position in the cluster array
    - fs = sampling frequency
    - save = if True, images will be saved in current_dir
    - current_dir = directory in which to save images

    OUTPUT:
    - max_indexes = indexes of maximum activity
    - min_indexes = indexes of minimum activity
    """



    print("\nSPIKE NUMBER IN WINDOW\n")



    num_of_neurons = 0
    for electrode in electrodes_name:
        if cluster[legend[electrode].values[0]] != None:
            num_of_neurons = num_of_neurons + len(cluster[legend[electrode].values[0]])
    print('Con questo metodo rilevo', num_of_neurons, 'neuroni')

    num_of_windows = int(round(duration/window, 0))  #Calcolo il numero di finestre da 30secondi che possono esserci nel mio segnale
    count = 1
    check = False

    max_indexes = []
    min_indexes = []
    total_indexes = np.zeros((num_of_neurons, num_of_windows), dtype=np.uint32)

    fig = plt.figure(figsize = (50,50))
    plt.rcParams.update({'font.size': 9})
    if num_of_neurons <= max_n:
        cells = math.ceil(math.sqrt(num_of_neurons))
    else:
        cells = int(math.sqrt(max_n))

    iteration = 0

    for electrode in electrodes_name:
        if cluster[legend[electrode].values[0]] != None:

            for neuron in cluster[legend[electrode].values[0]]:

                tot_detected_spikes = len(neuron)
                n_spike = []

                for a in range(1, num_of_windows+1): #vario le finestre

                    a_inf = (a - 1)*fs*window #estremo inf della finestra
                    a_sup = a*fs*window  #estremo sup della finestra

                    spike_neuron_indexes_in_window = []
                    for f in range(0, tot_detected_spikes):
                        if (neuron[f] > a_inf) and (neuron[f] <= a_sup):
                                spike_neuron_indexes_in_window.append(neuron[f])

                    n_spike.append(len(spike_neuron_indexes_in_window))

                total_indexes[iteration,:]=np.array(n_spike)
                iteration +=1

                #Numero di neuroni fino a 36
                if num_of_neurons <= max_n:
                    ax = fig.add_subplot(cells, cells, count)
                    ax.set_title("Neuron " + str(count)+"/" +str(num_of_neurons)+ " Number of spikes in " + str(window) + "s" )
                    ax.bar(np.arange(num_of_windows),height= n_spike)
                    if count == num_of_neurons:
                        if save == True:
                            file_name = current_dir + "/SpikeInNumberWindow.png"
                            plt.savefig(file_name, dpi=300)
                        plt.show()
                    count += 1

                #Numero di neuroni maggiore di 36 (per liberare RAM)
                else:

                    if check == False:
                        p=1

                    #Caso base
                    if count <= max_n:
                        ax = fig.add_subplot(cells, cells, count)
                        ax.set_title("Neuron " + str(count)+"/" +str(num_of_neurons)+ " Number of spikes in " + str(window) + "s" )
                        ax.bar(np.arange(num_of_windows),height= n_spike)
                        if count == max_n:
                            if save == True:
                                file_name = current_dir + "/SpikeInNumberWindow" + str(p) + ".png"
                                plt.savefig(file_name, dpi=300)
                            plt.show() #Mostro il grafico da 0 a 36
                            if num_of_neurons > count: #Creo il nuovo plot se ci sono altri neuroni
                                check = True
                                p+=1
                                fig = plt.figure(figsize = (50,50))
                                plt.rcParams.update({'font.size': 9})
                        count += 1

                    #Altri casi
                    elif count > max_n*(p-1) and count <= max_n*p:
                        ax = fig.add_subplot(cells, cells, count - max_n*(p-1))
                        ax.set_title("Neuron " + str(count)+"/" +str(num_of_neurons)+ " Number of spikes in " + str(window) + "s" )
                        ax.bar(np.arange(num_of_windows),height= n_spike)
                        if (count == max_n*p) or (count ==num_of_neurons):
                            if save == True:
                                file_name = current_dir + "/SpikeInNumberWindow" + str(p) + ".png"
                                plt.savefig(file_name, dpi=300)
                            plt.show() #Mostro il grafico
                            if num_of_neurons > count: #Creo il nuovo plot se ci sono altri neuroni
                                p+=1
                                fig = plt.figure(figsize = (50,50))
                                plt.rcParams.update({'font.size': 9})
                        count += 1

            #Per cercare la finestra del massimo
            M = 0


    normalize=np.zeros((num_of_neurons, num_of_windows), dtype=np.float32)
    for i in range(num_of_neurons):
        somma=np.sum(total_indexes[i, :])
        normalize[i,:]=total_indexes[i,:]/somma

    #GRAFICO NORMALIZZATO
    print("Grafico delle medie normalizzate")
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize = (12,6))
    plt.bar(np.arange(num_of_windows), np.mean(normalize, axis=0),
        color = '#779ecb', yerr=np.std(normalize, axis=0),
        ecolor='#333333', capsize=1.5)
    plt.grid()
    if save == True:
        file_name = current_dir + "/GraficoNormalizzato.png"
        plt.savefig(file_name, dpi=300)
    plt.show()

    
    #GRAFICO NON NORMALIZZATO
    print("Grafico delle medie non normalizzate")
    plt.figure(figsize = (12,6))
    plt.bar(np.arange(num_of_windows), np.mean(total_indexes, axis=0),
        color = '#779ecb')
    plt.grid()
    plt.axis([-1,num_of_windows,0.8*np.min(np.mean(total_indexes, axis=0)),1.05*np.max(np.mean(total_indexes, axis=0))])
    if save == True:
        file_name = current_dir + "/GraficoNonNormalizzato.png"
        plt.savefig(file_name, dpi=300)
    plt.show()
    
    
    #VARI PARAMETRI CALCOLATI
    print("Il massimo dell'attività neuronale è mediamente nella seguente finestra:", np.argmax(np.mean(total_indexes, axis=0))*window,
          " s -", (np.argmax(np.mean(total_indexes, axis=0))+1)*window, "s, di valore: ", int(np.max(np.mean(total_indexes, axis=0))),
          " \ne di deviazione standard nella finestra di massimo di: ", np.std(total_indexes[:, np.argmax(np.mean(total_indexes, axis=0))]))

    print("Il minimo dell'attività neuronale è mediamente nella seguente finestra:", np.argmin(np.mean(total_indexes, axis=0))*window,
          "s -", (np.argmin(np.mean(total_indexes, axis=0))+1)*window, "s, di valore: ", int(np.min(np.mean(total_indexes, axis=0))),
          " \ne di deviazione standard nella finestra di minimo di: ", np.std(total_indexes[:, np.argmin(np.mean(total_indexes, axis=0))]))


    return max_indexes, min_indexes


    
    

    
    

# ISI HISTOGRAM ==================================================

def ISI(max_n, cluster, electrodes_name, fs, save, current_dir, legend, resolution = 50):
    """
    INPUT:
    - max_n = maximum number of pictures to plot in a square - it must be a perfect square (to not exceed your computer limitations)
    - cluster = clustered_neurons
    - electrodes_name = name of electrodes that detected the signal
    - fs = sampling frequency
    - legend = pandas dataframe containing a link between the electrode name and the electrode position in the cluster array
    - save = if True, images will be saved in current_dir
    - current_dir = directory in which to save images
    - resolution = resolution in which to plot the histogram. The lower the more fine it will be.
    
    
    OUTPUT:
    - diff_indexes = array containing the differences between indexes of the neurons detected
    """
    
    print("\nISI HISTOGRAM\n")
    
    #Calculate time difference between spikes of each neuron
    diff_indexes = []
    for electrode in electrodes_name:
        if cluster[legend[electrode].values[0]] != None:
            for neuron in cluster[legend[electrode].values[0]]:
                    diff_indexes.append(np.diff(neuron, n=1))

    print("Numero di neuroni identificati con questo metodo: " + str(len(diff_indexes)))

    #ISI Histogram
    #On the y-axis we see the density of probability of spikes
    #On the x-axis we see the "samples of time" (whoose unit measure is 1/10000Hz)

    #Doing this method we loose the time information of WHEN a spike occured as we sum up in the histogram the number of intervals

    #Searching for the maximum InterSpike Interval in all neurons (used in the PLOT to set the axis and pace)
    neurons_maximum_interval = []
    for neuron in range(len(diff_indexes)):
        for i in range(len(diff_indexes[neuron])):
            neurons_maximum_interval.append(diff_indexes[neuron][i])
    maximum_interval = max(neurons_maximum_interval)

    print("Il massimo InterSpike Interval ISI su tutti i neuroni è: " + str(maximum_interval))
    
    #ISI Histogram
    #Set to "density = True" in order to see a distribution of probability and to make a comparison between different electrodes
    
    #PLOT
    check = False
    count = 1
    num_of_neurons = len(diff_indexes)
    pace = [x*resolution for x in range(int(fs/resolution))]
    
    fig = plt.figure(figsize = (50,50))
    plt.rcParams.update({'font.size': 9})
    
    if num_of_neurons <= max_n:
        cells = math.ceil(math.sqrt(num_of_neurons))
    else:
        cells = int(math.sqrt(max_n))
    
    
    while count <= (num_of_neurons):
        #Numero di neuroni fino a 36
        if num_of_neurons <= max_n: 
            ax = fig.add_subplot(cells, cells, count)
            ax.set_title("Neuron " + str(count)+"/" +str(num_of_neurons)+ " ISI (Probability Distribution)")
            ax.hist(diff_indexes[count-1], histtype='bar', density=True, alpha = 1, color='k', bins=pace)
            if count == num_of_neurons:
                if save == True:
                    file_name = current_dir + "/ISI.png"
                    plt.savefig(file_name, dpi=300)
                plt.show()
            count += 1
                
        #Numero di neuroni maggiore di 36 (per liberare RAM)
        else:          
            if check == False:
                p=1          
            #Caso base
            if count <= max_n:
                ax = fig.add_subplot(cells, cells, count)
                ax.set_title("Neuron " + str(count)+"/" +str(num_of_neurons)+ " ISI (Probability Distribution)")
                ax.hist(diff_indexes[count-1], histtype='bar', density=True, alpha = 1, color='k', bins=pace)
                if count == max_n:
                    if save == True:
                        file_name = current_dir + "/ISI" + str(p) + ".png"
                        plt.savefig(file_name, dpi=300)
                    plt.show() #Mostro il grafico da 0 a 36
                    if num_of_neurons > count: #Creo il nuovo plot se ci sono altri neuroni
                        check = True
                        p+=1
                        fig = plt.figure(figsize = (50,50))
                        plt.rcParams.update({'font.size': 9})
                count += 1   
            #Altri casi
            elif count > max_n*(p-1) and count <= max_n*p:
                ax = fig.add_subplot(cells, cells, count - max_n*(p-1))
                ax.set_title("Neuron " + str(count)+"/" +str(num_of_neurons)+ " ISI (Probability Distribution)")
                ax.hist(diff_indexes[count-1], histtype='bar', density=True, alpha = 1, color='k', bins=pace)
                if (count == max_n*p) or (count == num_of_neurons):
                    if save == True:
                        file_name = current_dir + "/ISI" + str(p) + ".png"
                        plt.savefig(file_name, dpi=300)
                    plt.show() #Mostro il grafico
                    if num_of_neurons > count: #Creo il nuovo plot se ci sono altri neuroni
                        p+=1
                        fig = plt.figure(figsize = (50,50))
                        plt.rcParams.update({'font.size': 9})
                count += 1
        
    
    return diff_indexes

    

    
    
    

# FREQUENCY VARIATION ============================================

    
def FrequencyVariation(max_n, cluster, electrodes_name, num_of_samples, fs, legend, save, current_dir, secondi=5):
    """
    INPUT:
    - max_n = maximum number of pictures to plot in a square - it must be a perfect square (to not exceed your computer limitations)
    - cluster = clustered_neurons
    - electrodes_name = name of electrodes that detected the signal
    - num_of_samples = number of samples in the total signal
    - fs = sampling frequency
    - legend = pandas dataframe containing a link between the electrode name and the electrode position in the cluster array
    - save = if True, images will be saved in current_dir
    - current_dir = directory in which to save images
    - secondi = seconds to do the moving avarage
    
    
    OUTPUT:
    - None
    """
    
    
    print("\nFREQUENCY VARIATION\n")
    
    data_neurons = pd.DataFrame(columns=['Neurone', 'Media', 'Deviazione_standard', 'Num_spikes'])
    
    n=int(secondi*fs)
    

    num_of_neurons = 0
    for electrode in electrodes_name:
        if cluster[legend[electrode].values[0]] != None:    
            num_of_neurons = num_of_neurons + len(cluster[legend[electrode].values[0]]) 
    print('Con questo metodo rilevo', num_of_neurons, 'neuroni')
    
    #PLOT
    check = False
    count = 1
    fig = plt.figure(figsize = (50,50))
    plt.rcParams.update({'font.size': 15})
    
    if num_of_neurons <= max_n:
        cells = math.ceil(math.sqrt(num_of_neurons))
    else:
        cells = int(math.sqrt(max_n))
 
    for electrode in electrodes_name:
        if cluster[legend[electrode].values[0]] != None:
            for neuron in cluster[legend[electrode].values[0]]:
                
                #Numero di neuroni fino a 36
                if num_of_neurons <= max_n: 
                    
                    diff_indexes = np.diff(neuron, n=1)
                    freq = (1/diff_indexes)*fs
                    interp = scipy.interpolate.interp1d(neuron[:-1], freq)
                    dominio = np.arange(min(neuron),max(neuron[:-1]),1)
                    appross = np.array(interp(dominio), dtype=np.float32)
                    moving_appross = moving_average_analysis(appross, n=n)
                    
                    new_row = pd.DataFrame([[count, np.mean(freq, dtype=np.float32),
                                            np.std(freq, dtype=np.float32), len(diff_indexes)+1]],
                                            columns=['Neurone', 'Media', 'Deviazione_standard', 'Num_spikes'])
                    #print(new_row)
                    data_neurons = data_neurons.append(new_row)
                
                    ax = fig.add_subplot(cells, cells, count)
                    ax.set_title("Neuron " + str(count)+"/" +str(num_of_neurons)+ " Frequency" )
                    ax.plot(dominio/fs, np.array(appross, dtype=np.float16), color='#BBBBBB')
                    ax.plot(dominio[:-(n-1)]/fs, np.array(moving_appross, dtype=np.float16), color='#CC0000', linewidth=1.5)
                    ax.axis([0,num_of_samples/fs,0,max(appross)*(1.1)])
                    if count == num_of_neurons:
                        if save == True:   
                            file_name = current_dir + "/Frequency.png"
                            plt.savefig(file_name, dpi=300)
                        plt.show()
                    count += 1    
                
                #Numero di neuroni maggiore di 36 (per liberare RAM)
                else:
                    
                    if check == False:
                        p=1
                    
                    #Caso base
                    if count <= max_n:
                        
                        diff_indexes = np.diff(neuron, n=1)
                        freq = (1/diff_indexes)*fs
                        interp = scipy.interpolate.interp1d(neuron[:-1], freq)
                        dominio = np.arange(min(neuron),max(neuron[:-1]),1)
                        appross = np.array(interp(dominio), dtype=np.float32)
                        moving_appross = moving_average_analysis(appross, n=n)
                        
                        new_row = pd.DataFrame([[count, np.mean(freq, dtype=np.float32),
                                            np.std(freq, dtype=np.float32), len(diff_indexes)+1]],
                                            columns=['Neurone', 'Media', 'Deviazione_standard', 'Num_spikes'])
                        
                        data_neurons = data_neurons.append(new_row)
                
                        ax = fig.add_subplot(cells, cells, count)
                        ax.set_title("Neuron " + str(count)+"/" +str(num_of_neurons)+ " Frequency" )
                        ax.plot(dominio/fs, np.array(appross, dtype=np.float16), color='#BBBBBB')
                        ax.plot(dominio[:-(n-1)]/fs, np.array(moving_appross, dtype=np.float16), color='#CC0000', linewidth=1.5)
                        ax.axis([0,num_of_samples/fs,0,max(appross)*(1.1)])
                        
                        if count == max_n:
                            if save == True:   
                                file_name = current_dir + "/Frequency" + str(p) + ".png"
                                plt.savefig(file_name, dpi=300)
                            plt.show() #Mostro il grafico da 0 a 36
                            if num_of_neurons > count: #Creo il nuovo plot se ci sono altri neuroni
                                check = True
                                p+=1
                                fig = plt.figure(figsize = (50,50))
                                plt.rcParams.update({'font.size': 15})
                        count += 1                      
                    
                    #Altri casi
                    elif count > max_n*(p-1) and count <= max_n*p:
                        
                        
                        diff_indexes = np.diff(neuron, n=1)
                        freq = (1/diff_indexes)*fs
                        interp = scipy.interpolate.interp1d(neuron[:-1], freq)
                        dominio = np.arange(min(neuron),max(neuron[:-1]),1)
                        appross = np.array(interp(dominio), dtype=np.float32)
                        moving_appross = moving_average_analysis(appross, n=n)
                        
                        new_row = pd.DataFrame([[count, np.mean(freq, dtype=np.float32),
                                            np.std(freq, dtype=np.float32), len(diff_indexes)+1]],
                                            columns=['Neurone', 'Media', 'Deviazione_standard', 'Num_spikes'])
                        #print(new_row)
                        data_neurons = data_neurons.append(new_row)
                
                        ax = fig.add_subplot(cells, cells, count - max_n*(p-1))
                        ax.set_title("Neuron " + str(count)+"/" +str(num_of_neurons)+ " Frequency" )
                        ax.plot(dominio/fs, np.array(appross, dtype=np.float16), color='#BBBBBB')
                        ax.plot(dominio[:-(n-1)]/fs, np.array(moving_appross, dtype=np.float16), color='#CC0000', linewidth=1.5)
                        ax.axis([0,num_of_samples/fs,0,max(appross)*(1.1)])
                        
                        if (count == max_n*p) or (count ==num_of_neurons):
                            if save == True:   
                                file_name = current_dir + "/Frequency" + str(p) + ".png"
                                plt.savefig(file_name, dpi=300)
                            plt.show() #Mostro il grafico
                            if num_of_neurons > count: #Creo il nuovo plot se ci sono altri neuroni
                                p+=1
                                fig = plt.figure(figsize = (50,50))
                                plt.rcParams.update({'font.size': 15})
                        count += 1
                        
    
    return data_neurons


# BURSTS =========================================================


def FindML(cluster, electrodes_name, legend, fs, maximum_time_interval = 0.06):
    """
    INPUT:
    - electrodes_name = name of electrodes that detected the signal
    - legend = pandas dataframe containing a link between the electrode name and the electrode position in the cluster array
    - fs = sampling frequency
    - maximum_time_interval = the maximum time interval allowed to consider a series of spikes a burst (0.06s)
    
    OUTPUT:
    - ML = array containing the interval for each neuron in which spikes are considered a burst
    """
    
    #Calculate time difference between spikes of each neuron
    diff_indexes = []
    for electrode in electrodes_name:
        if cluster[legend[electrode].values[0]] != None:
            for neuron in cluster[legend[electrode].values[0]]:
                    diff_indexes.append(np.diff(neuron, n=1))
    
    #To limit the bursting interval
    maximum_time_samples = maximum_time_interval*fs
    
    #Creating useful variables
    L = [None for i in range(len(diff_indexes))]
    ISI_mean = []
    ML = []
    
    #Cycle to calculate the bursting intervals
    for neuron in range(len(diff_indexes)):
        less_then = []
        ISI_mean.append(np.mean(diff_indexes[neuron]))
        for spike_index in diff_indexes[neuron]:
            if spike_index < ISI_mean[neuron]:
                less_then.append(spike_index)
        L[neuron] = less_then
        if (np.mean(L[neuron])<maximum_time_samples):
            ML.append(np.mean(L[neuron]))
        else:
            ML.append(maximum_time_samples)
            
    return ML




def FindBursts(neuron_indexes, thr, electrode_name, neuron_number, fs):
    """
    INPUT:
    - neuron_indexes = indexes of the clustered neuron
    - thr = ML threshold
    - electrode_name = name of the electrode that detected the signal
    - neuron_number = number associated with the detected neuron
    - fs = sampling frequency
    
    OUTPUT:
    - data = Pandas dataframe containing information regarding the detected bursts
    """

    #Creation of the dataframe
    data = pd.DataFrame(columns=['Canale', 'Neurone', 'Indice_in', 'Indice_fin', 'Durata', 'Num_spikes', 'Tempo_minimo(soglia)'])
    isi = np.diff(neuron_indexes, n=1)
    time=thr/fs
    
    #Cycle to find the bursts
    i=0
    while (i < len(isi)):
        
        k=1
        check=False
        while (thr >= np.mean(isi[i:i+k])) & ((i+k <= len(isi))):
            k+=1            
            check=True

        if (check == True):
            new_row = pd.DataFrame([[electrode_name, neuron_number, int(neuron_indexes[i]), int(neuron_indexes[i+k-1]), np.sum(isi[i:i+k-1])/fs, int(k), time]], columns=['Canale', 'Neurone', 'Indice_in', 'Indice_fin', 'Durata', 'Num_spikes', 'Tempo_minimo(soglia)'])
            data = data.append(new_row)
            i = i + k
        else:
            i += 1

    return data