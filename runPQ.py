import Picoscope4000
import PQTools as pq
import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from ringarray import ring_array, ring_array_global_data
from RingArray2 import ringarray2
import logging
import json
import psutil

# Initialize Logging
pqLogger = logging.getLogger('pqLogger')
filehandler = logging.FileHandler('Logs/pqLog.log')
streamhandler = logging.StreamHandler()

pqLogger.setLevel(logging.INFO)
filehandler.setLevel(logging.INFO)
streamhandler.setLevel(logging.WARNING)

formatterq = logging.Formatter('%(asctime)s \t %(levelname)s \t %(message)s')
filehandler.setFormatter(formatterq)
streamhandler.setFormatter(formatterq)
pqLogger.addHandler(filehandler)
pqLogger.addHandler(streamhandler)
PLOTTING = 0

# Open Picoscope
pqLogger.INFO('Opening Picoscope...')
pico = Picoscope4000.Picoscope4000()
pico.run_streaming()
pqLogger.INFO('...done')

parameters = pico.get_parameters()
streaming_sample_interval = parameters['streaming_sample_interval']

# Set the minimum snippet size that is tested for 10 periods
min_snippet_length = streaming_sample_interval * 0.5

# Allocate data arrays & Initialize variables
data = ringarray2(max_size = 3 * sample_rate)
data_10seconds = ring_array(max_size = 20 * sample_rate) 
data_10min = ring_array(size = 5000000)
rms_half_period = np.zeros(20)

freq_10seconds_list = []
rms_10periods_list = []
rms_heatmap_list = []
thd_10periods_list = []
harmonics_10periods_list = []
pst_list = []
snippet_size_list = []

number_of_10periods = 0
first_value = 0
restdata = []
is_first_iteration = 1
is_first_10periods = 1
lastPst = 0
lastPlt = 0
frequency_10periods = 0
rms_10periods = 0
thd_10periods = 0
measurement_time_string = ''

start_time = time.time()

# Wait for 10 Minute switch in order to have nice data
if True:
    import datetime
    while datetime.datetime.now().minute % 10 != 0 and datetime.datetime.now().seconds > 2:
        time.sleep(0.5)

# Main PQ Measurement and Calculation loop
# ========================================
try:
    while True:
        while data.size < min_snippet_length:
            
            snippet = pico.get_queue_data()

            if snippet is not None:
                # Write snippet size to JSON
                snippet_size_list.append(snippet.size)
                pq.writeJSON(snippet_size_list,1000,'snippetsize.json')

                # Seperate arrays for data for 10 periods and data for 10 seconds
                data.attach_to_back(snippet)
                data_10seconds.attach_to_back(snippet)
                
                # Prepare data for Flicker calculation
                data_for_10min, first_value = pq.convert_data_to_lower_fs(snippet, streaming_sample_interval+1, first_value)
                data_10min.attach_to_back(data_for_10min)

                pqLogger.debug('Length of snippet:       +'+str(snippet.size))
                pqLogger.debug('Length of current data: '+str(data.size))
                    
        # Cut off everything before the first zero crossing:
        # ==================================================
        if is_first_iteration: #happens every 10 Minutes
            first_zero_crossing = data.cut_off_before_first_zero_crossing()
            is_first_iteration = 0
            counter = first_zero_crossing
            
            ten_minute_number = datetime.datetime.now().hour*6 + datetime.datetime.now().minute / 10
            if ten_minute_number == 0:
                day_number += 1
                # This fails when you start a measurement between 00:00 and 00:10 :(
        
        # Find 10 periods
        # ===============
        number_of_10periods += 1

        if any(np.diff(data.get_data_view()) > 500):
            print('Error')
            plt.plot(data.get_data_view())
            plt.grid(True)
            plt.show()

        data_10periods, zero_indices = data.cut_off_10periods2()
        # Save a backup for debugging (consistency check)
        data_10periods_backup = data_10periods.copy()

        # Check zero_indices for plausibility (45 Hz > f < 55Hz)
        if any(np.diff(zero_indices) > 11111): # < 45 Hz
            pqLogger.error('Distance between two zero crossings: '+str(max(np.diff(zero_indices))))
        if any(np.diff(zero_indices) < 9090): # > 55 Hz
            pqLogger.error('Distance between two zero crossings: '+str(min(np.diff(zero_indices))))

        pqLogger.debug('Cutting off 10 periods containing '+str(zero_indices[20])+' samples')

        #data_10periods = data.cut_off_front2(zero_indices[20], 20)

        # Write last waveform to JSON
        waveform = data_10periods[zero_indices[18]:zero_indices[20]].copy()
        if (waveform[200] < 0):
            waveform = data_10periods[zero_indices[17]:zero_indices[19]].copy()
        waveform = waveform[0::200]
        pq.writeJSON(waveform,2000,'waveform.json')


        # Calculate and store RMS values of half periods 
        # ==============================================
        for i in xrange(20):    
            rms_half_period[i] = pq.calculate_rms_half_period(data_10periods[zero_indices[i]:zero_indices[i+1]])

        # Calculate and store frequency for 10 periods
        # =============================================
        frequency_10periods = pq.calculate_frequency_10periods(zero_indices, streaming_sample_interval)
        #print(str(frequency_10periods))
        pqLogger.debug('Frequency of 10 periods: '+str(frequency_10periods))
        pqLogger.debug('Mean value of 10 periods: '+str(np.mean(data_10periods)))

        # Calculate and store RMS values of 10 periods
        # ============================================
        rms_10periods = pq.calculate_rms(data_10periods)

        # Write last values to voltage.json
        rms_10periods_list.append(rms_10periods)
        pq.writeJSON(rms_10periods_list, 1000, 'voltage.json')

        pqLogger.debug('RMS voltage of 10 periods: '+str(rms_10periods))

        # Calculate and store harmonics and THD values of 10 periods
        # ==========================================================
        harmonics_10periods = pq.calculate_harmonics_voltage(data_10periods,streaming_sample_interval)
        harmonics_10periods_list.append(harmonics_10periods)

        thd_10periods = pq.calculate_THD(harmonics_10periods, streaming_sample_interval)
        thd_10periods_list.append(thd_10periods)

        pqLogger.debug('THD of 10 periods: '+str(thd_10periods))

        # Write current harmonics to JSON
        pq.writeJSON([h / harmonics_10periods[0] * 100 for h in harmonics_10periods[1:]],40,'harmonics.json')
        pq.writeJSON(thd_10periods_list,100,'thd.json')

        # Write JSON file about current situation
        # =======================================

        # Construct pretty string about measurement time
        measurement_time = int(round(time.time() - start_time))
        days = measurement_time / 60 / 60 / 24
        hours = measurement_time%(60*60*24) / 60 / 60
        minutes = measurement_time%(60*60) / 60
        seconds = measurement_time % 60
        measurement_time_string = str(days)+'d '+str(hours)+'h '+str(minutes)+'m '+str(seconds)+'s'

        # find min, max and average frequency and voltage

        if (is_first_10periods):
            min_freq = frequency_10periods
            max_freq = frequency_10periods
            avrg_freq = frequency_10periods

            min_volt = rms_10periods
            max_volt = rms_10periods
            avrg_volt = rms_10periods
            is_first_10periods = 0

        min_freq = min(min_freq,frequency_10periods)
        max_freq = max(max_freq,frequency_10periods)
        avrg_freq = (avrg_freq*number_of_10periods + frequency_10periods) / (number_of_10periods + 1.0)

        min_volt = min(min_volt,rms_10periods)
        max_volt = max(max_volt,rms_10periods)
        avrg_volt = (avrg_volt*number_of_10periods + rms_10periods) / (number_of_10periods + 1.0)

        infoDict = {# Status Info
                    'measurement_alive':1, 
                    'samplingrate':streaming_sample_interval, 
                    'ram':round(psutil.virtual_memory()[2],1),
                    'cpu':round(psutil.cpu_percent(),1),
                    'disk':round(psutil.disk_usage('/')[3],1),
                    'measurement_time': measurement_time_string,
                    # Frequency Info
                    'currentFreq': round(frequency_10periods,3),
                    'freqmin': round(min_freq,3),
                    'freqmax': round(max_freq,3),
                    'freqavrg': round(avrg_freq,3),
                    # Voltage Info
                    'currentVoltage': round(rms_10periods,2),
                    'voltmin': round(min_volt,3),
                    'voltmax': round(max_volt,3),
                    'voltavrg': round(avrg_volt,3),
                    # Harmonics Info
                    'currentTHD': round(thd_10periods,2),
                    # Flicker Info
                    'lastPst': round(lastPst,2),
                    'lastPlt': round(lastPlt,2)}
        with open(os.path.join('html','jsondata','info.json'),'wb') as f:
            f.write(json.dumps(infoDict))


        # Calculate frequency of 10 seconds
        # =================================
        if (data_10seconds.size > 10*streaming_sample_interval):
            frequency_data = data_10seconds.cut_off_front2(10*streaming_sample_interval)
            pqLogger.debug('Samples in 10 second interval: '+str(frequency_data.size))
            #pq.compare_filter_for_zero_crossings(frequency_data, streaming_sample_interval)
            if PLOTTING:
                plt.plot(frequency_data)
                plt.grid()
                plt.show()
            frequency = pq.calculate_Frequency(frequency_data, streaming_sample_interval)

            # Write last values to json file
            freq_10seconds_list.append(frequency)
            pq.writeJSON(freq_10seconds_list, 200, 'frequency.json')
            pqLogger.info(pq.test_frequency(frequency))
            # write to freqHeatmap.json
            freq_heatmap_list.append([ten_second_number % 360, ten_second_number, frequency])
            pq.writeJSON(freq_heatmap_list, 8640, 'freqHeatmap.json')
            ten_second_number =+ 1


        # Prepare for 10 min Measurement
        # ==============================
        counter += data_10periods.size
        print(' + '+str(data_10periods.size))
        print(str(counter))
        if not np.array_equal(data_10periods, data_10periods_backup):
            pqLog.CRITICAL('data_10periods was changed')
        # Synchronize data so absolutely nothing is lost
        if (counter >= 6*streaming_sample_interval):
            #plt.plot(data_10periods[(6*streaming_sample_interval-counter):])
            #plt.show()
            #plt.plot(data.get_data_view())
            #plt.show()
            data.attach_to_front(data_10periods[(6*streaming_sample_interval-counter):])
            print(str(6*streaming_sample_interval-counter))
            #plt.plot(data.get_data_view())
            #plt.show()
            is_first_iteration = 1
            
            # Calculate RMS of 10 min
            # =======================
            rms_10min = pq.count_up_values(rms_10periods_list)
            rms_10periods_list = []
            pqLogger.info(pq.test_rms(rms_10min))

            # Write data for heatmap
            rms_heatmap_list.append([day_number, ten_minute_number, rms_10min])
            pq.writeJSON(rms_heatmap_list, 14400, 'rmsHeatmap.json')
            
            # Calculate THD of 10 min
            # =======================
            thd_10min = pq.count_up_values(thd_10periods_list)
            thd_10periods_list = []
            pqLogger.info(pq.test_thd(thd_10min))
            
            # Calculate Harmonics of 10 min
            # =======================
            harmonics_10min = pq.count_up_values(harmonics_10periods_list)
            harmonics_10periods_list = []
            pqLogger.info(pq.test_harmonics(harmonics_10min))
            
           
        # Calculate flicker of 10 min
        # ===========================
        if (data_10min.size > 10*60*4000):
            flicker_data = data_10min.cut_off_front2(600*streaming_sample_interval/250)
            Pst = pq.calculate_Pst(flicker_data)
            lastPst = Pst
            pst_list.append(Pst)
            pqLogger.info('Pst of last 10m: '+str(Pst))
            
            #pq.writeJSON(pst_list, 200, 'flicker.json')


        # Calculate flicker of 2 hours    
        # ============================
        if (len(pst_list) == 12):
            Plt = pq.calculate_Plt(pst_list)
            lastPlt = Plt
            pqLogger.info(pq.test_plt(Plt))

#except KeyboardInterrupt:
    #print('Aborting...')

#except Exception as e:
    #print(str(type(e)))
    #print(str(sys.exc_info()[:]))
    #raise(sys.exc_info()[1])
    #raise

finally:

    # Write one last JSON including death flag
    infoDict = {'measurement_alive':0, 
                'samplingrate':streaming_sample_interval, 
                'ram':round(psutil.virtual_memory()[2],1),
                'cpu':round(psutil.cpu_percent(),1),
                'disk':round(psutil.disk_usage('/')[3],1),
                'currentFreq': round(frequency_10periods,3),
                'currentVoltage': round(rms_10periods,2),
                'currentTHD': round(thd_10periods,2),
                'lastPst': round(lastPst,2),
                'lastPlt': round(lastPlt,2),
                'measurement_time': measurement_time_string}

    with open(os.path.join('html','jsondata','info.json'),'wb') as f:
        f.write(json.dumps(infoDict))

    # Stop Sampling and release Picoscope unit
    pico.close_unit()

    # Error Handling: Save and log all variables
    # ==========================================
