import Picoscope4000
import PQTools as pq
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from ring_array import ring_array, ring_array_global_data
import logging

PLOTTING = 0

pico = Picoscope4000.Picoscope4000()
pico.run_streaming()

parameters = pico.get_parameters()
streaming_sample_interval = parameters['streaming_sample_interval']

min_snippet_length = streaming_sample_interval/2

data = ring_array_global_data(size=2000000)
data_10seconds = ring_array(size=(20*streaming_sample_interval)) 
data_10min = ring_array(size=5000000)
rms_half_period = np.array(np.zeros(20))
rms_10periods_list = []
thd_10periods_list = []
harmonics_10periods_list = []

first_value = 0
is_first_iteration = 1
#time.sleep(0.5) # Activate when first data is None and first iterations runs with None data, should be fixed

# Initialize Logging
# ==================

queueLogger = logging.getLogger('queueLogger')
queueLogger.setLevel(logging.DEBUG)

fhq = logging.FileHandler('Logs/queueLog.log')
fhq.setLevel(logging.DEBUG)

shq = logging.StreamHandler()
shq.setLevel(logging.INFO)

formatterq = logging.Formatter('%(asctime)s \t %(levelname)s \t %(message)s')
fhq.setFormatter(formatterq)
shq.setFormatter(formatterq)

queueLogger.addHandler(fhq)
queueLogger.addHandler(shq)

dataLogger = logging.getLogger('dataLogger')
dataLogger.setLevel(logging.DEBUG)

fhd = logging.FileHandler('Logs/dataLog.log')
fhd.setLevel(logging.DEBUG)

shd = logging.StreamHandler()
shd.setLevel(logging.INFO)

formatterd = logging.Formatter('%(asctime)s \t %(levelname)s \t %(message)s')
fhd.setFormatter(formatterd)
shd.setFormatter(formatterd)

dataLogger.addHandler(fhd)
dataLogger.addHandler(shd)

try:    
    while True:
        while data.size < min_snippet_length:
            
            snippet = pico.get_queue_data()

            if snippet is not None:
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
            dataLogger.error('Number of zero crossings in '+str(data.get_data_view().size)+': '+str(zero_indices.size))
        dataLogger.debug('Cutting off :'+str(zero_indices[20]))
        queueLogger.debug('Cutting off:            -'+str(zero_indices[20]))        
        data_10periods = data.cut_off_front2(zero_indices[20], 20)
        queueLogger.debug('Length of current data: '+str(data.size))
        
        # Calculate and store frequency for 10 periods
        # =============================================
        frequency_10periods = pq.calculate_frequency_10periods(zero_indices, streaming_sample_interval)
        dataLogger.debug('Frequency of 10 periods: '+str(frequency_10periods))
        dataLogger.debug('Mean value of 10 periods: '+str(np.mean(data_10periods)))
    
     
        # Calculate and store RMS values of half periods 
        # ==============================================
        for i in xrange(20):    
            rms_half_period[i] = pq.calculate_rms_half_period(data_10periods[zero_indices[i]:zero_indices[i+1]])
        if PLOTTING:
            plt.plot(rms_half_period)
            plt.title(' Voltage RMS of Half Periods')
            plt.grid(True)
            plt.show()

        # Calculate and store RMS values of 10 periods
        # ============================================
        rms_10periods = pq.calculate_rms(data_10periods)
        rms_10periods_list.append(rms_10periods)
        dataLogger.debug('RMS voltage of 10 periods: '+str(rms_10periods))

        # Calculate and store harmonics and THD values of 10 periods
        # ==========================================================
        harmonics_10periods = pq.calculate_harmonics_voltage(data_10periods,streaming_sample_interval)
        thd_10periods = pq.calculate_THD(harmonics_10periods, streaming_sample_interval)
        harmonics_10periods_list.append(harmonics_10periods)
        thd_10periods_list.append(thd_10periods)
        dataLogger.debug('THD of 10 periods: '+str(thd_10periods))
        
        # Prepare for 10 min Measurement
        # ==============================
        counter += data_10periods.size
        if (counter >= 600*streaming_sample_interval):
            data.attach_to_front(data_10periods[:(600*streaming_sample_interval-counter)])
            queueLogger.debug('Length of current data: '+str(data.size))
            is_first_iteration = 1
            
            # Calculate RMS of 10 min
            # =======================
            rms_10min = pq.count_up_values(rms_10periods_list)
            rms_10periods_list = []
            dataLogger.debug('RMS voltage of 10 min: '+str(rms_10min))
            
            # Calculate THD of 10 min
            # =======================
            thd_10min = pq.count_up_values(thd_10periods_list)
            thd_10periods_list = []
            dataLogger.debug('THD of 10 min: '+str(thd_10min))
            
            # Calculate Harmonics of 10 min
            # =======================
            harmonics_10min = pq.count_up_values(harmonics_10periods_list)
            harmonics_10periods_list = []
            dataLogger.debug('Harmonics of 10 min: '+str(harmonics_10min))
            
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
            dataLogger.info('Frequency of 10s: '+str(frequency))
                
        # Calculate flicker of 10 min
        # ===========================
        if (data_10min.size > 2400000):
            flicker_data = data_10min.cut_off_front2(600*streaming_sample_interval/250)
            Pst = pq.calculate_Pst(flicker_data)
            dataLogger.debug('Pst: '+str(Pst))
           
finally:
    pico.close_unit()

    # Error Handling: Save and log all variables
    # ==========================================
