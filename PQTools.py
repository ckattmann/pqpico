# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 18:02:43 2015

@author: Malte Gerber
"""
##########---------------------------Module--------------------------##########

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import sys, os
import json
import scipy.fftpack as fftpack

##########------------------------Konstanten-------------------------##########

V_max = 32768/50
R1 = 993000 # Ohm
R2 = 82400*1000000/(82400+1000000) # Ohm
Resolution = (R1+R2)/R2
f_line = 50 # Hz
Class  = 0

##########-------------Initialisierung von globalen Listen-----------##########

y = np.array(np.zeros(1500))

##########------------------------Funktionen-------------------------##########

# Filters
# =======

def moving_average3(a,n=25):
    ret = np.cumsum(a,dtype=float)
    ret_begin = ret[:n:2]/np.arange(1,n+1,2)
    ret_end = np.cumsum(a[-n/2:], dtype=float)
    ret_end = (ret_end[-1]+ret_end[0]-ret_end)/np.arange(n,0,-2)    
    ret[n/2+1:-n/2+1] = (ret[n:] - ret[:-n])/n
    ret[:n/2+1] = ret_begin
    ret[-(n/2+1):] = ret_end
    return ret

def moving_average(a,n=25):
    ret = np.cumsum(a,dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return np.append(np.zeros(n/2),ret[n-1:]/n)

def moving_average2(values,window=25):
    weigths = np.repeat(1.0, window)/window
    #including valid will REQUIRE there to be enough datapoints.
    #for example, if you take out valid, it will start @ point one,
    #not having any prior points, so itll be 1+0+0 = 1 /3 = .3333
    smas = np.convolve(values, weigths, 'same')
    return smas # as a numpy array

def Lowpass_Filter(data, SAMPLING_RATE):
    show_filtered_measurement = 1    
    roundoff_freq = 2000.0
    b_hp, a_hp = signal.butter(1, round(roundoff_freq / SAMPLING_RATE / 2,5))
    #print('WP: '+str(round(roundoff_freq/SAMPLING_RATE/2)))
    data_filtered = signal.lfilter(b_hp, a_hp, data)
    
    if (show_filtered_measurement):
        plt.plot(data, 'b') 
        plt.plot(data_filtered, 'r')
        plt.xlim(0, 100000)
        plt.grid(True)
        plt.show()
    return data_filtered
        
# Frequency Calculation
# =====================

def detect_zero_crossings(data,PLOTTING=False,filter_func='moving_average2'):
    #data_filtered = Lowpass_Filter(data, SAMPLING_RATE)    
    #data -= np.mean(data)
    #data_filtered = moving_average(data)
    #data_filtered2 = moving_average3(data)
    modulename = sys.modules[__name__]
    func = getattr(modulename,filter_func)
    data_filtered = func(data)
    pos = data_filtered > 0
    npos = ~pos
    zero_crossings_raw = ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:]))

    pos = data_filtered >= 0
    npos = ~pos
    zero_crossings_raw2 = ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:]))

    zero_crossings_combined = (zero_crossings_raw | zero_crossings_raw2).nonzero()[0]

    if PLOTTING:
        plt.plot(data, 'b') 
        plt.plot(data_filtered, 'r')
        #plt.plot(data_filtered2, 'g')
        #plt.xlim(0, zero_crossings_combined[20])
        plt.grid(True)
        plt.plot(zero_crossings_combined,np.zeros(zero_crossings_combined.size),'o')
        plt.show()
    return zero_crossings_combined

def compare_filter_for_zero_crossings(data, SAMPLING_RATE):
    freq1 = calculate_Frequency(SAMPLING_RATE, data, filter_func = 'moving_average')
    freq2 = calculate_Frequency(SAMPLING_RATE, data, filter_func = 'moving_average2')
    freq3 = calculate_Frequency(SAMPLING_RATE, data, filter_func = 'moving_average3')

    print('moving_average: '+str(freq1)+', moving_average2: '+str(freq2)+', moving_average3: '+str(freq3))

def calculate_Frequency(data, SAMPLING_RATE, filter_func = 'moving_average3'):        
    zero_indices = detect_zero_crossings(data,PLOTTING = False, filter_func = filter_func)
    #print('Number of zero_crossings_pure: '+str(zero_indices.size))
    if (zero_indices.size % 2 != 0):
        zero_indices = zero_indices[:-1]
    #print('Number of zero_crossings: '+str(zero_indices.size))
    #samplesbetweenzeroindices = (zero_indices[-1]-zero_indices[0])
    #print(samplesbetweenzeroindices)
    frequency = float((zero_indices.size-1)/2) / ((zero_indices[-1]-zero_indices[1])) * SAMPLING_RATE
    return frequency        

def calculate_frequency_10periods(zero_indices, SAMPLING_RATE):
    time_10periods = float((zero_indices[20] - zero_indices[0])) / SAMPLING_RATE
    frequency_10periods = 10.0 / time_10periods
    return frequency_10periods

# Voltage RMS calculation
# =======================

def calculate_rms(data):
    #Der Effektivwert wird ueber alle Messpunkte gebildet             
    rms_points = np.sqrt(np.mean(np.power(data, 2)))
    rms = rms_points/V_max*Resolution
    return rms
    
def calculate_rms_half_period(data):
    #Der Effektivwert wird ueber alle Messpunkte gebildet             
    rms_points = np.sqrt(np.mean(np.power(data, 2)))
    rms_half_period = rms_points/V_max*Resolution
    if (rms_half_period <= (0.9*230) and rms_half_period >= (0.1*230)):
        pass
        #print ("Es ist eine Unterspannung aufgetreten!")
        ###----hier wird statt der ausgabe ein flag gesetzt-----######
    elif rms_half_period < 0.1*230:
        pass
        #print("Es liegt eine Spannungunterbrechung vor!")
        ###----hier wird statt der ausgabe ein flag gesetzt-----######
    elif rms_half_period > 1.1*230:
        pass
        #print ("Es ist eine Überspannung aufgetreten!")
        ###----hier wird statt der ausgabe ein flag gesetzt-----######
    else:
        pass
        #print("Alles OK!")
        ###----hier wird statt der ausgabe ein flag gesetzt-----#####
    return rms_half_period

# Harmonics & THD
# ===============

def fast_fourier_transformation(data, SAMPLING_RATE, plot_FFT=False):            
    zero_padding = 200000#2**int(np.log(SAMPLING_RATE*0.2)/np.log(2))    
    #calculation of the fft      
    FFTdata = np.fft.fftshift(np.fft.fft(data, zero_padding))/zero_padding    
    #frequencies of the harmonics    
    FFTfrequencys = np.fft.fftfreq(FFTdata.size, 1.0/SAMPLING_RATE)
    #cut off the negativ indices and double the amplitudes
    FFTdata = np.abs(FFTdata[(FFTdata.size/2):])*2
    
    if (plot_FFT):
        plt.plot(FFTfrequencys[:FFTdata.size], FFTdata)
        plt.xlabel("f in Hz") # y-Achse beschriefen
        plt.ylabel("FFT") # x-Achse beschriften
        plt.xlim([0,1500]) # länge der angezeigten x-Achse
    
    return FFTdata, FFTfrequencys

def fast_fourier_transformation2(data, SAMPLING_RATE, plot_FFT=False):            
        
    #calculation of the fft      
    FFTdata = np.fft.fftshift(np.fft.fft(data))/data.size    
    #frequencies of the harmonics    
    FFTfrequencys = np.fft.fftfreq(FFTdata.size, 1.0/SAMPLING_RATE)
    #cut off the negativ indices and double the amplitudes
    FFTdata = np.abs(FFTdata[(FFTdata.size/2):])*2
    
    if (plot_FFT):
        plt.plot(FFTfrequencys[:FFTdata.size], FFTdata)
        plt.xlabel("f in Hz") # y-Achse beschriefen
        plt.ylabel("FFT") # x-Achse beschriften
        plt.xlim([0,1500]) # länge der angezeigten x-Achse
    
    return FFTdata, FFTfrequencys
        
def fast_fourier_transformation3(data, SAMPLING_RATE, plot_FFT=False):                
    #calculation of the fft      
    FFTdata = fftpack.fftshift(fftpack.fft(data))/data.size    
    #frequencies of the harmonics    
    FFTfrequencys = np.fft.fftfreq(FFTdata.size, 1.0/SAMPLING_RATE)
    #cut off the negativ indices and double the amplitudes
    FFTdata = np.abs(FFTdata[(FFTdata.size/2):])*2
    
    if (plot_FFT):
        plt.plot(FFTfrequencys[:FFTdata.size], FFTdata)
        plt.xlabel("f in Hz") # y-Achse beschriefen
        plt.ylabel("FFT") # x-Achse beschriften
        plt.xlim([0,1500]) # länge der angezeigten x-Achse
    
    return FFTdata, FFTfrequencys
        
def calculate_harmonics_voltage(data, SAMPLING_RATE):
    FFTdata, FFTfrequencys = fast_fourier_transformation(data, SAMPLING_RATE)        
    harmonics_amplitudes = np.zeros(40)
    #area_amplitudes = round(len(FFTdata)*2/float(SAMPLING_RATE)/0.02)
    #The fundamental amplitude is located at index 10, if the window size is exactly ten periods.  
    area_amplitudes = 10          
    for i in xrange(1,41): 
        #Berechnung der Harmonischen über eine for-Schleife        
        harmonics_amplitudes[i-1] = np.sqrt(np.sum(FFTdata[int(area_amplitudes*i-1):int(area_amplitudes*i+2)]**2)) #direkter Amplitudenwert aus FFT
    return harmonics_amplitudes
    
def calculate_harmonics_standard(data, SAMPLING_RATE):
    FFTdata, FFTfrequencys = fast_fourier_transformation(data, SAMPLING_RATE)        
    harmonics_amplitudes = np.zeros(40)
    #area_amplitudes = round(len(FFTdata)*2/float(SAMPLING_RATE)/0.02)
    area_amplitudes = 10 
    for i in xrange(1,41): 
        grouping_part1 = 0.5*FFTdata[int(round(area_amplitudes*i-area_amplitudes/2))]**2
        grouping_part2 = 0.5*FFTdata[int(round(area_amplitudes*i+area_amplitudes/2))]**2
        grouping_part3 = np.sum(FFTdata[int(round(area_amplitudes*i-area_amplitudes/2)+1):int(round(area_amplitudes*i+area_amplitudes/2))]**2)       
        harmonics_amplitudes[i-1] = np.sqrt(grouping_part1+grouping_part2+grouping_part3)
    return harmonics_amplitudes

def calculate_THD(harmonics_10periods, SAMPLING_RATE):
    harmonics_10periods = harmonics_10periods**2
    THD = np.sqrt(np.sum(harmonics_10periods[1:])/harmonics_10periods[0])*100
    return THD
    
# Flicker
# =======

def convert_data_to_lower_fs(data, SAMPLING_RATE, first_value):

    #step = int(SAMPLING_RATE/4000)
    step = 250
    #takes every 250th value of the data array
    data_flicker =data[first_value::step]
    #calcutation of the new first value
    new_first_value = step-(data.size-step*(data_flicker.size-1)-first_value)
    return data_flicker, new_first_value

def convert_data_to_lower_fs2(data, SAMPLING_RATE, restdata):
    reduction_rate = int(SAMPLING_RATE / 4000)
    data = np.append(data,restdata)
    reduced_data = data[::reduction_rate]
    restdata = data[data.size % reduction_rate]
    return reduced_data, restdata

def calculate_Pst(data):    
    show_time_signals = 0           #Aktivierung des Plots der Zeitsignale im Flickermeter
    show_filter_responses = 0       #Aktivierung des Plots der Amplitudengänge der Filter.
                                    #(zu Prüfzecken der internen Filter)
    
    fs = 4000    
       
    ## Block 1: Modulierung des Spannungssignals
    
    u = data - np.mean(data)                      # entfernt DC-Anteil
    u_rms = np.sqrt(np.mean(np.power(u,2))) # Normierung des Eingangssignals
    u = u / (u_rms * np.sqrt(2))
    
    ## Block 2: Quadratischer Demulator
    
    u_0 = u**2
    
    ## Block 3: Hochpass-, Tiefpass- und Gewichtungsfilter
    
    # Konfiguration der Filter
    HIGHPASS_ORDER  = 1 #Ordnungszahl der Hochpassfilters
    HIGHPASS_CUTOFF = 0.05 #Hz Grenzfrequenz
    
    LOWPASS_ORDER = 6 #Ordnungszahl des Tiefpassfilters
    if (f_line == 50):
      LOWPASS_CUTOFF = 35.0 #Hz Grenzfrequenz
    
    if (f_line == 60):
      LOWPASS_CUTOFF = 42.0 #Hz Grenzfrequenz
    
    # subtract DC component to limit filter transients at start of simulation
    u_0_ac = u_0 - np.mean(u_0)
    
    b_hp, a_hp = signal.butter(HIGHPASS_ORDER, (HIGHPASS_CUTOFF/(fs/2)), 'highpass')
    u_hp = signal.lfilter(b_hp, a_hp, u_0_ac)
    
    # smooth start of signal to avoid filter transient at start of simulation
    smooth_limit = min(round(fs / 10), len(u_hp))
    u_hp[ : smooth_limit] = u_hp[ : smooth_limit] * np.linspace(0, 1, smooth_limit)
    
    b_bw, a_bw = signal.butter(LOWPASS_ORDER, (LOWPASS_CUTOFF/(fs/2)), 'lowpass')
    u_bw = signal.lfilter(b_bw, a_bw, u_hp)
    
    # Gewichtungsfilter (Werte sind aus der Norm)
    
    if (f_line == 50):
      K = 1.74802
      LAMBDA = 2 * np.pi * 4.05981
      OMEGA1 = 2 * np.pi * 9.15494
      OMEGA2 = 2 * np.pi * 2.27979
      OMEGA3 = 2 * np.pi * 1.22535
      OMEGA4 = 2 * np.pi * 21.9
    
    if (f_line == 60):
      K = 1.6357
      LAMBDA = 2 * np.pi * 4.167375
      OMEGA1 = 2 * np.pi * 9.077169
      OMEGA2 = 2 * np.pi * 2.939902
      OMEGA3 = 2 * np.pi * 1.394468
      OMEGA4 = 2 * np.pi * 17.31512
    
    num1 = [K * OMEGA1, 0]
    denum1 = [1, 2 * LAMBDA, OMEGA1**2]
    num2 = [1 / OMEGA2, 1]
    denum2 = [1 / (OMEGA3 * OMEGA4), 1 / OMEGA3 + 1 / OMEGA4, 1]
    
    b_w, a_w = signal.bilinear(np.convolve(num1, num2),
                               np.convolve(denum1, denum2), fs)
    u_w = signal.lfilter(b_w, a_w, u_bw)
    
    ## Block 4: Quadrierung und Varianzschätzer
    
    LOWPASS_2_ORDER  = 1
    LOWPASS_2_CUTOFF = 1 / (2 * np.pi * 300e-3)  # Zeitkonstante 300 msek.
    SCALING_FACTOR   = 1238400  # Skalierung auf eine Wahrnehmbarkeitsskala
    
    u_q = u_w**2
    
    b_lp, a_lp = signal.butter(LOWPASS_2_ORDER,(LOWPASS_2_CUTOFF/(fs/2)),'low')
    s = SCALING_FACTOR * signal.lfilter(b_lp, a_lp, u_q)
    
    ## Block 5: Statistische Berechnung
    
    p_50s = np.mean([np.percentile(s, 100-30, interpolation="linear"),
                     np.percentile(s, 100-50, interpolation="linear"),
                     np.percentile(s, 100-80, interpolation="linear")])
    p_10s = np.mean([np.percentile(s, 100-6, interpolation="linear"),
                     np.percentile(s, 100-8, interpolation="linear"), 
                     np.percentile(s, 100-10, interpolation="linear"),
                     np.percentile(s, 100-13, interpolation="linear"),
                     np.percentile(s, 100-17, interpolation="linear")])
    p_3s = np.mean([np.percentile(s, 100-2.2, interpolation="linear"),
                    np.percentile(s, 100-3, interpolation="linear"),
                    np.percentile(s, 100-4, interpolation="linear")])
    p_1s = np.mean([np.percentile(s, 100-0.7, interpolation="linear"),
                    np.percentile(s, 100-1, interpolation="linear"),
                    np.percentile(s, 100-1.5, interpolation="linear")])
    p_0_1s = np.percentile(s, 100-0.1, interpolation="linear")
    
    P_st = np.sqrt(0.0314*p_0_1s+0.0525*p_1s+0.0657*p_3s+0.28*p_10s+0.08*p_50s)
    
    if (show_time_signals):
        t = np.linspace(0, len(u) / fs, num=len(u))
        plt.figure()
        plt.clf()
        #plt.subplot(2, 2, 1)
        plt.hold(True)
        plt.plot(t, u, 'b', label="u")
        plt.plot(t, u_0, 'm', label="u_0")
        plt.plot(t, u_hp, 'r', label="u_hp")
        plt.xlim(0, len(u)/fs)
        plt.hold(False)
        plt.legend(loc=1)
        plt.grid(True)
        #plt.subplot(2, 2, 2)
        plt.figure()
        plt.clf()    
        plt.hold(True)
        plt.plot(t, u_bw, 'b', label="u_bw")
        plt.plot(t, u_w, 'm', label="u_w")
        plt.xlim(0, len(u)/fs)
        plt.legend(loc=1)
        plt.hold(False)
        plt.grid(True)
        #plt.subplot(2, 2, 3)
        plt.figure()
        plt.clf()    
        plt.plot(t, u_q, 'b', label="u_q")
        plt.xlim(0, len(u)/fs)
        plt.legend(loc=1)
        plt.grid(True)
        #plt.subplot(2, 2, 4)
        plt.figure()
        plt.clf()    
        plt.plot(t, s, 'b', label="s")
        plt.xlim(0, len(u)/fs)
        plt.legend(loc=1)
        plt.grid(True)
    
    if (show_filter_responses):
        f, h_hp = signal.freqz(b_hp, a_hp, 4096)
        f, h_bw = signal.freqz(b_bw, a_bw, 4096)
        f, h_w = signal.freqz(b_w, a_w, 4096)
        f, h_lp = signal.freqz(b_lp, a_lp, 4096)
        f = f/np.pi*fs/2    
        
        plt.figure()
        plt.clf()
        plt.hold(True)
        plt.plot(f, abs(h_hp), 'b', label="Hochpass 1. Ordnung")
        plt.plot(f, abs(h_bw), 'r', label="Butterworth Tiefpass 6.Ordnung")
        plt.plot(f, abs(h_w), 'g', label="Gewichtungsfilter")
        plt.plot(f, abs(h_lp), 'm', label="Varianzschätzer")
        plt.legend(bbox_to_anchor=(1., 1.), loc=2)    
        plt.hold(False)
        plt.grid(True)
        plt.axis([0, 35, 0, 1])
        
    return P_st 
    
def calculate_Plt(Pst_list):
    P_lt = np.power(np.power(Pst_list,3)/12,1/3)
    return P_lt
    
# Unbalance
# =========

def calculate_unbalance(rms_10min_u, rms_10min_v, rms_10min_w):
    a = -0.5+0.5j*np.sqrt(3)
    u1 =1.0/3*(rms_10min_u+rms_10min_v+rms_10min_w)
    u2 = 1.0/3 *(rms_10min_u+a*rms_10min_v+a**2*rms_10min_w)
    return np.abs(u2)/np.abs(u1)*100
    
# Other useful functions
# ======================
    
def count_up_values(values_list):
    new_value = np.sqrt(np.sum(np.power(values_list,2),axis=0)/len(values_list))
    return new_value

# writes the last n values of array into the given json file
def writeJSON(array, size, filename):
    valuesdict = {'values': [round(x,3) for x in array[-size:]]}
    with open(os.path.join('html','tests','jsondata',filename),'wb') as f:
        f.write(json.dumps(valuesdict))


# Plot functions
# ==============

class plotting_frequency():
    def __init__(self):
        self.y = np.array(np.zeros(1500))
        self.x = np.arange(0,self.y.size/5,0.2)
        self.fig, self.ax1 = plt.subplots(1,1)
        plt.xlim(300,0)
        plt.ylim(49.85, 50.15)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.title('Time course of the mains frequency:')
        plt.grid(True)
        plt.plot(self.x,self.y)
    
    def plot_frequency(self, freq): 
        self.y = np.roll(self.y,1)
        self.y[-1] = freq                
        self.ax1.clear()
        plt.xlim(300,0)
        plt.ylim(49.85, 50.15)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.title('Time course of the mains frequency:')
        plt.grid(True)
        plt.plot(self.x,self.y)
        plt.ion()
        plt.draw()

# Compare with Standard
# =====================

def test_thd(thd):
    if thd>8:
        return 'The THD is with '+str(thd)+' % too high!'
    else:
        return 'THD of 10 min: '+str(thd)+' %'

def test_harmonics(harmonics):
    limits = np.array([0.02,0.05,0.01,0.06,0.005,0.05,0.005,0.015,0.005,0.035,
                       0.005,0.03,0.005,0.005,0.005,0.02,0.005,0.015,0.005,
                       0.005,0.005,0.015,0.005,0.015])
    harmonics_boolean = harmonics[1:25]/harmonics[0] > limits
    if harmonics_boolean.any():
        index = np.where(harmonics_boolean)
        return 'The following amplitudes of the harmonics are too high: '+str(index)
    else:
        return 'Harmonics are ok!'

def test_rms(rms):
    if rms<230*0.9:
        return 'The RMS is with '+str(rms)+' V too low!'
    elif rms>230*1.1:
        return 'The RMS is with '+str(rms)+' V too high!'
    else:        
        return 'RMS voltage of 10 min: '+str(rms)+' V'

def test_frequency(frequency):
    if frequency<49.5:
        return 'The frequency is with '+str(frequency)+' Hz too low!'
    elif frequency>50.5:
        return 'The frequency is with '+str(frequency)+' Hz too high!'
    else:
        return 'Frequency of 10s: '+str(frequency)+' Hz'

def test_plt(plt):
    if plt>1:
        return 'The Plt is with a value of '+str(plt)+' too high!'
    else:
        return 'Plt: '+str(plt)  
