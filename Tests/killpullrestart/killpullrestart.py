import os
import time
import subprocess
import signal

print('Own processid: '+str(os.getpid()))
try:
    with open('.processid','r') as f:
        processid = int(f.read().strip())
    os.system('kill -2 '+str(processid))
    print('Killed processid '+str(processid))
except IOError:
    print('No .processid found, assuming loop.py isnt running')
except OSError:
    print('Could not kill process #'+str(processid))


print('Pulling Code from Github')

#subprocess.call('nohup python loop.py &')
os.system('nohup python loop.py &')

print('loop.py started')
