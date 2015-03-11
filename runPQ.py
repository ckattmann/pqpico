import Picoscope4000
import PQTools as pq
import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from ringarray import ring_array, ring_array_global_data
import logging
import json
import psutil

PLOTTING = 0

pico = Picoscope4000.Picoscope4000()
pico.run_streaming()

parameters = pico.get_parameters()
streaming_sample_interval = parameters['streaming_sample_interval']

min_snippet_length = streaming_sample_interval/2

data = ring_array_global_data(size=3000000)
data_10seconds = ring_array(size=(20*streaming_sample_interval)) 
data_10min = ring_array(size=5000000)
rms_half_period = np.array(np.zeros(20))

freq_10seconds_list = []
rms_10periods_list = []
thd_10periods_list = []
harmonics_10periods_list = []
pst_list = []
snippet_size_list = []

first_value = 0
restdata = []
is_first_iteration = 1
lastPst = 0
lastPlt = 0
#time.sleep(0.5) # Activate when first data is None and first iterations runs with None data, should be fixed

# Initialize Logging
# ==================

queueLogger = logging.getLogger('queueLogger')
queueLogger.setLevel(logging.INFO)
fhq = logging.FileHandler('Logs/queueLog.log')
fhq.setLevel(logging.INFO)
shq = logging.StreamHandler()
shq.setLevel(logging.WARNING)
formatterq = logging.Formatter('%(asctime)s \t %(levelname)s \t %(message)s')
fhq.setFormatter(formatterq)
shq.setFormatter(formatterq)
queueLogger.addHandler(fhq)
queueLogger.addHandler(shq)

dataLogger = logging.getLogger('dataLogger')
dataLogger.setLevel(logging.DEBUG)
fhd = logging.FileHandler('Logs/dataLog.log')
fhd.setLevel(logging.ERROR)
shd = logging.StreamHandler()
shd.setLevel(logging.WARNING)
formatterd = logging.Formatter('%(asctime)s \t %(levelname)s \t %(message)s')
fhd.setFormatter(formatterd)
shd.setFormatter(formatterd)
dataLogger.addHandler(fhd)
dataLogger.addHandler(shd)


start_time = time.time()

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

                data.attach_to_back(snippet)
                data_10seconds.attach_to_back(snippet)
                
                # Prepare data for Flicker calculation
                # ====================================
                data_for_10min, first_value = pq.convert_data_to_lower_fs(snippet, streaming_sample_interval+1, first_value)
                data_10min.attach_to_back(data_for_10min)

                queueLogger.debug('Length of snippet:      +'+str(snippet.size))
                queueLogger.debug('Length of current data: '+str(data.size))
                    
            else:
                pass    
       
        # Cut off everything before the first zero crossing:
        # ==================================================           
        if is_first_iteration:
            first_zero_crossing = data.get_zero_indices()[0]
            queueLogger.debug('Cut off '+str(first_zero_crossing)+' values before first zero crossing') 
            data.cut_off_front2(first_zero_crossing, 0)
            queueLogger.debug('Length of current data: '+str(data.size))
            is_first_iteration = 0
            counter = first_zero_crossing
        
        # Find 10 periods
        # ===============
        zero_indices = data.get_zero_indices()[:21]
        if zero_indices.size < 15 or zero_indices.size > 200:
            #Is this really 50Hz Sinus Data?
            dataLogger.error('Number of zero crossings in '+str(data.get_data_view().size)+': '+str(zero_indices.size))
        dataLogger.debug('Cutting off :'+str(zero_indices[20]))
        queueLogger.debug('Cutting off:            -'+str(zero_indices[20]))        

        data_10periods = data.cut_off_front2(zero_indices[20], 20)

        queueLogger.debug('Length of current data: '+str(data.size))

        # Write last waveform to JSON
        waveform = data_10periods[zero_indices[18]:zero_indices[20]]
        print('zero_indices  : '+str(zero_indices))
        print('data_10periods: '+str(data_10periods))
        if (waveform[200] < 0):
            waveform = data_10periods[zero_indices[17]:zero_indices[19]]
        waveform = pq.moving_average2(waveform, 25)
        waveform = waveform[0::20]
        pq.writeJSON(waveform,2000,'waveform.json')


        # Calculate and store RMS values of half periods 
        # ==============================================
        for i in xrange(20):    
            rms_half_period[i] = pq.calculate_rms_half_period(data_10periods[zero_indices[i]:zero_indices[i+1]])
        if PLOTTING:
            plt.plot(rms_half_period)
            plt.title(' Voltage RMS of Half Periods')
            plt.grid(True)
            plt.show()

        # Calculate and store frequency for 10 periods
        # =============================================
        frequency_10periods = pq.calculate_frequency_10periods(zero_indices, streaming_sample_interval)
        dataLogger.debug('Frequency of 10 periods: '+str(frequency_10periods))
        dataLogger.debug('Mean value of 10 periods: '+str(np.mean(data_10periods)))

        # Calculate and store RMS values of 10 periods
        # ============================================
        rms_10periods = pq.calculate_rms(data_10periods)

        # Write last values to voltage.json
        rms_10periods_list.append(rms_10periods)
        pq.writeJSON(rms_10periods_list, 1000, 'voltage.json')

        dataLogger.debug('RMS voltage of 10 periods: '+str(rms_10periods))

        # Calculate and store harmonics and THD values of 10 periods
        # ==========================================================
        harmonics_10periods = pq.calculate_harmonics_voltage(data_10periods,streaming_sample_interval)
        harmonics_10periods_list.append(harmonics_10periods)

        thd_10periods = pq.calculate_THD(harmonics_10periods, streaming_sample_interval)
        thd_10periods_list.append(thd_10periods)

        dataLogger.debug('THD of 10 periods: '+str(thd_10periods))

        # Write current harmonics to JSON
        pq.writeJSON([h / harmonics_10periods[0] * 100 for h in harmonics_10periods[1:]],40,'harmonics.json')
        pq.writeJSON(thd_10periods_list,100,'thd.json')

        # Write JSON file about current situation
        # =======================================
        # Construct pretty string about measurement time
        measurement_time = int(round(time.time() - start_time))
        days = measurement_time / 60 / 60 / 24
        hours = measurement_time / 60 / 60
        minutes = measurement_time / 60
        seconds = measurement_time % 60
        measurement_time_string = str(days)+'d, '+str(hours)+'h, '+str(minutes)+'m, '+str(seconds)+'s'
        infoDict = {'samplingrate':streaming_sample_interval, 
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


        # Calculate frequency of 10 seconds
        # =================================
        if (data_10seconds.size > 10*streaming_sample_interval):
            frequency_data = data_10seconds.cut_off_front2(10*streaming_sample_interval)
            queueLogger.debug('Size frequency_data snippet: '+str(frequency_data.size))
            #pq.compare_filter_for_zero_crossings(frequency_data, streaming_sample_interval)
            if PLOTTING:
                plt.plot(frequency_data)
                plt.grid()
                plt.show()
            frequency = pq.calculate_Frequency(frequency_data, streaming_sample_interval)

            # Write last values to json file
            freq_10seconds_list.append(frequency)
            pq.writeJSON(freq_10seconds_list, 200, 'frequency.json')
            dataLogger.info(pq.test_frequency(frequency))

        # Prepare for 10 min Measurement
        # ==============================
        counter += data_10periods.size
        # Synchronize data so absolutely nothing is lost
        if (counter >= 600*streaming_sample_interval):
            data.attach_to_front(data_10periods[:(600*streaming_sample_interval-counter)])
            queueLogger.debug('Length of current data: '+str(data.size))
            is_first_iteration = 1
            
            # Calculate RMS of 10 min
            # =======================
            rms_10min = pq.count_up_values(rms_10periods_list)
            rms_10periods_list = []
            dataLogger.info(pq.test_rms(rms_10min))
            
            # Calculate THD of 10 min
            # =======================
            thd_10min = pq.count_up_values(thd_10periods_list)
            thd_10periods_list = []
            dataLogger.info(pq.test_thd(thd_10min))
            
            # Calculate Harmonics of 10 min
            # =======================
            harmonics_10min = pq.count_up_values(harmonics_10periods_list)
            harmonics_10periods_list = []
            dataLogger.info(pq.test_harmonics(harmonics_10min))
            
           
        # Calculate flicker of 10 min
        # ===========================
        if (data_10min.size > 2400000):
            flicker_data = data_10min.cut_off_front2(600*streaming_sample_interval/250)
            Pst = pq.calculate_Pst(flicker_data)
            lastPst = Pst
            pst_list.append(Pst)
            dataLogger.info('Pst: '+str(Pst))
            
            #pq.writeJSON(pst_list, 200, 'flicker.json')


        # Calculate flicker of 2 hours    
        # ============================
        if (len(pst_list) == 12):
            Plt = pq.calculate_Plt(pst_list)
            lastPlt = Plt
            dataLogger.info(pq.test_plt(Plt))

#except KeyboardInterrupt:
    #print('Aborting...')


finally:
    pico.close_unit()

    # Error Handling: Save and log all variables
    # ==========================================
