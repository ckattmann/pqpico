import json
import time
import random
import psutil

start_time = time.time()
time.sleep(random.random() *10)

measurement_time = int(round(time.time() - start_time))
days = measurement_time / 60 / 60 / 24
hours = measurement_time / 60 / 60
minutes = measurement_time / 60
seconds = measurement_time % 60
measurement_time_string = str(days)+'d '+str(hours)+'h '+str(minutes)+'m '+str(seconds)+'s'

print(measurement_time_string)

infoDict = {'samplingrate':999999, 
            'ram':psutil.virtual_memory()[2],
            'cpu':psutil.cpu_percent(),
            'disk':psutil.disk_usage('/')[3],
            'currentFreq': 50.368723,
            'currentVoltage': 235.38479562,
            'currentTHD': 2.12315,
            'lastPst': round(0.01419283764,2),
            'lastPlt': round(0.1123945786,2),
            'measurement_time': measurement_time_string}

print(str(infoDict))

with open('info.json','wb') as f:
    f.write(json.dumps(infoDict))
