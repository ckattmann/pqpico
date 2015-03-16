import json
import time
import random
import psutil

start_time = time.time()
time.sleep(random.random() *10)

measurement_time = int(round(time.time() - start_time)) * 100000
days = measurement_time / 60 / 60 / 24
hours = measurement_time%(60*60*24) / 60 / 60
minutes = measurement_time%(60*60) / 60
seconds = measurement_time % 60
measurement_time_string = str(days)+'d '+str(hours)+'h '+str(minutes)+'m '+str(seconds)+'s'

print(measurement_time_string)

infoDict = {'measurement_alive':0,
            'samplingrate':999999, 
            'ram':psutil.virtual_memory()[2],
            'cpu':psutil.cpu_percent(),
            'disk':psutil.disk_usage('/')[3],
            'currentFreq': 50.36,
            'freqmin': round(48.9678,3),
            'freqmax': '{:.3f}'.format(51.32983),
            'freqavrg': round(50,3),
            'currentVoltage': 235.38,
            'voltmin': round(230.123876,3),
            'voltmax': round(238.283764278,3),
            'voltavrg': round(234.348756,3),
            'currentTHD': 2.12,
            'lastPst': round(0.01419283764,2),
            'lastPlt': round(0.1123945786,2),
            'measurement_time': measurement_time_string}

print(str(infoDict))

with open('info.json','wb') as f:
    f.write(json.dumps(infoDict))
