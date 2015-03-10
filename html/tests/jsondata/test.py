import json
import psutil
infoDict = {'samplingrate':999999, 
            'ram':psutil.virtual_memory()[2],
            'cpu':psutil.cpu_percent(),
            'disk':psutil.disk_usage('/')[3],
            'currentFreq': 50.368723,
            'currentVoltage': 235.38479562,
            'currentTHD': 2.12315}

with open('info.json','wb') as f:
    f.write(json.dumps(infoDict))
