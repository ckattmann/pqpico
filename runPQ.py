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
import psutil # For Performance Evaluation
import datetime

# Initialize Logging
os.remove('html/Logs/pqLog.log')
pqLogger = logging.getLogger('pqLogger')
filehandler = logging.FileHandler('html/Logs/pqLog.log')
streamhandler = logging.StreamHandler()

pqLogger.setLevel(logging.INFO)
filehandler.setLevel(logging.INFO)
streamhandler.setLevel(logging.INFO)

formatterq = logging.Formatter('%(asctime)s \t %(levelname)s \t %(message)s')
filehandler.setFormatter(formatterq)
streamhandler.setFormatter(formatterq)
pqLogger.addHandler(filehandler)
pqLogger.addHandler(streamhandler)
PLOTTING = 0

# Open Picoscope
pqLogger.info('Opening Picoscope...')
pico = Picoscope4000.Picoscope4000()
pico.run_streaming()
pqLogger.info('...done')

parameters = pico.get_parameters()
sample_rate = parameters['streaming_sample_interval']

# Set the minimum snippet size that is tested for 10 periods
min_snippet_length = sample_rate * 0.3

# Allocate data arrays & Initialize variables
data = ringarray2(max_size = 3 * sample_rate)
data_10seconds = ring_array(size = 20 * sample_rate) 
data_10min = ringarray2(max_size = 5000000)
rms_half_period = np.zeros(20)

diff_zero_indices_10seconds = []
freq_10seconds_list = []
rms_10periods_list = []
rms_10min_list = []
rms_heatmap_list = []
freq_heatmap_list = []
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
day_number = 0

start_time = time.time()

# Wait for 10 Minute switch in order to have nice data
if False:
    pqLogger.info('Waiting for 10 minute gong...')
    while not (datetime.datetime.now().minute % 10 != 0 and datetime.datetime.now().second < 2):
        time.sleep(0.1)

pqLogger.info('Starting Measurement')

ten_second_number = datetime.datetime.now().minute * 6 + datetime.datetime.now().second / 6
hour_number = datetime.datetime.now().hour

# Main PQ Measurement and Calculation loop
# ========================================
try:
    while True:
        while data.size < min_snippet_length:
            
            snippet = pico.get_queue_data()

            if snippet is not None:
                # Write snippet size to JSON
                snippet_size_list.append(snippet.size)
                pq.writeJSON(snippet_size_list,250,'snippetsize.json')

                # Seperate arrays for data for 10 periods and data for 10 seconds
                data.attach_to_back(snippet)
                data_10seconds.attach_to_back(snippet)
                
                # Prepare data for Flicker calculation
                data_for_10min, first_value = pq.convert_data_to_lower_fs(snippet, sample_rate+1, first_value)
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
            pqLogger.error('Difference between two consecutive samples greater than 500')

        data_10periods, zero_indices = data.cut_off_10periods2()
        diff_zero_indices_10seconds.append(np.diff(zero_indices))

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
        pq.writeJSON(waveform[0::200],100,'waveform.json')


        # Calculate and store RMS values of half periods 
        # ==============================================
        for i in xrange(20):    
            rms_half_period[i] = pq.calculate_rms_half_period(data_10periods[zero_indices[i]:zero_indices[i+1]])

        # Calculate and store frequency for 10 periods
        # =============================================
        frequency_10periods = pq.calculate_frequency_10periods(zero_indices, sample_rate)

        pqLogger.debug('Frequency of 10 periods: '+str(frequency_10periods))
        pqLogger.debug('Mean value of 10 periods: '+str(np.mean(data_10periods)))


        # Calculate and store RMS values of 10 periods
        # ============================================
        rms_10periods = pq.calculate_rms(data_10periods)
        rms_10periods_list.append(rms_10periods)

        pqLogger.debug('RMS voltage of 10 periods: '+str(rms_10periods))

        # Calculate and store harmonics and THD values of 10 periods
        # ==========================================================
        harmonics_10periods = pq.calculate_harmonics_voltage(data_10periods,sample_rate)
        harmonics_10periods_list.append(harmonics_10periods)

        thd_10periods = pq.calculate_THD(harmonics_10periods, sample_rate)
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
                    'samplingrate':sample_rate, 
                    'ram':round(psutil.virtual_memory()[2],1),
                    'cpu':round(psutil.cpu_percent(),1),
                    'disk':round(psutil.disk_usage('/')[3],1),
                    'measurement_time': measurement_time_string,
                    # Frequency Info
                    'currentFreq': '{:.3f}'.format(frequency_10periods),
                    'freqmin': '{:.3f}'.format(min_freq),
                    'freqmax': '{:.3f}'.format(max_freq),
                    'freqavrg': '{:.3f}'.format(avrg_freq),
                    # Voltage Info
                    'currentVoltage': '{:.2f}'.format(rms_10periods),
                    'voltmin': '{:.2f}'.format(min_volt),
                    'voltmax': '{:.2f}'.format(max_volt),
                    'voltavrg': '{:.2f}'.format(avrg_volt),
                    # Harmonics Info
                    'currentTHD': '{:.3f}'.format(thd_10periods),
                    # Flicker Info
                    'lastPst': '{:.2f}'.format(lastPst),
                    'lastPlt': '{:.2f}'.format(lastPlt)}
        with open(os.path.join('html','jsondata','info.json'),'wb') as f:
            f.write(json.dumps(infoDict))


        # Calculate frequency of 10 seconds
        # =================================
        if (data_10seconds.size > 10*sample_rate):
            frequency_data = data_10seconds.cut_off_front2(10*sample_rate)
            pqLogger.debug('Samples in 10 second interval: '+str(frequency_data.size))
            frequency = pq.calculate_Frequency(frequency_data, sample_rate)

            # Write last values to json file
            freq_10seconds_list.append(frequency)
            pq.writeJSON(freq_10seconds_list, 2160, 'frequency.json')
            pqLogger.debug(pq.test_frequency(frequency))

            # Write to freqHeatmap.json
            freq_heatmap_list.append([hour_number, ten_second_number, frequency])
            pq.writeJSON(freq_heatmap_list, 8640, 'freqHeatmap.json')
            ten_second_number += 1
            if ten_second_number == 360:
                ten_second_number = 0
                hour_number += 1
            if hour_number == 24:
                hour_number = 0


        # Prepare for 10 min Measurement
        # ==============================
        counter += data_10periods.size
         
        if not np.array_equal(data_10periods, data_10periods_backup):
            pqLog.CRITICAL('data_10periods was changed')
             
        # Synchronize data so absolutely nothing is lost
        if (counter >= 600*sample_rate):
            data.attach_to_front(data_10periods[(600*sample_rate-counter):])
            is_first_iteration = 1
            
            # Calculate RMS of 10 min
            # =======================
            rms_10min = pq.count_up_values(rms_10periods_list)
            rms_10min_list.append(rms_10min)
            rms_10periods_list = []
            pqLogger.debug(pq.test_rms(rms_10min))

            # Write 10min rms to json
            pq.writeJSON(rms_10min_list, 144, 'voltage.json')

            # Write data for heatmap
            rms_heatmap_list.append([day_number, ten_minute_number, rms_10min])
            pq.writeJSON(rms_heatmap_list, 14400, 'rmsHeatmap.json')

            # Write rms data to csv
            pq.writeCSV(rms_10min, 'voltage.csv')
            
            # Calculate THD of 10 min
            # =======================
            thd_10min = pq.count_up_values(thd_10periods_list)
            thd_10periods_list = []
            pqLogger.debug(pq.test_thd(thd_10min))
            
            # Calculate Harmonics of 10 min
            # =======================
            harmonics_10min = pq.count_up_values(harmonics_10periods_list)
            harmonics_10periods_list = []
            pqLogger.debug(pq.test_harmonics(harmonics_10min))
            
           
        # Calculate flicker of 10 min
        # ===========================
        if (data_10min.size > 10*60*4000):
            pqLogger.info('Size of data10min at 10 Minutes: '+str(data_10min.size))
            flicker_data = data_10min.cut_off_front2(600*sample_rate/250)
            Pst, maxs = pq.calculate_Pst(flicker_data)
            lastPst = Pst
            pst_list.append(Pst)
            pqLogger.debug('Pst of last 10m: '+str(Pst))
            
            pq.writeJSON(pst_list, 360, 'flicker.json')


        # Calculate flicker of 2 hours    
        # ============================
        if (len(pst_list) == 12):
            Plt = pq.calculate_Plt(pst_list)
            lastPlt = Plt
            pqLogger.debug(pq.test_plt(Plt))

#except KeyboardInterrupt:
    #print('Aborting...')

except Exception, e:
    locs = locals().copy()
    pqLogger.CRITICAL('Error: '+str(e))

    with open('.mailinfo','r') as f:
        mailinfo = f.readlines()
    mailinfo = [x.strip() for x in mailinfo]

    # Prepare to send Alert Mail
    s = smtplib.SMTP('smtp.gmail.com')
    s.ehlo()
    s.starttls()
    s.login(mailinfo[0], mailinfo[1])

    # Make a pretty table with local variables
    x = PrettyTable(['Variable','Content'])
    x.align['Variable'] + 'l'
    x.align['Content'] + 'l'
    x.padding_width = 1
    for k,v in locs.iteritems():
        x.add_row([str(k),str(v)])
    pqLogger.INFO('Sending from '+str(mailinfo[0])+' to '+str(mailinfo[2]))
    msg = 'From: PQpico\nSubject: Error in PQpico\n\nError in PQpico at ' + str(datetime.datetime.now().strftime('%A %x %X:%f')) +'\n\n'+ str(traceback.format_exc()) + '\n\n' + str(x)
    for recipient in mailinfo[2:]:
        s.sendmail(mailinfo[0], recipient, str(msg))
    s.quit()

finally:

    # Write one last JSON including death flag
    infoDict = {'measurement_alive':0, 
                'samplingrate':sample_rate, 
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
