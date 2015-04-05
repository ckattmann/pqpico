#!/usr/bin/env python

import os
import signal

os.chdir('/home/kipfer/pqpico')

#1 Kill the old runPQ process
print('Killing old instance of runPQ...')
print('--------------------------------')
try:
    with open('.processid','r') as f:
        processid = int(f.read().strip())
    os.kill(processid, signal.SIGINT)
except IOError:
    print('No file .processid found, assuming runPQ is NOT running...')
except OSError:
    print('Could not kill '+str(processid))

#2 Pull the new code
print('Pulling new Code from github...')
print('-------------------------------')
os.system('git fetch')
os.system('git pull')

#3 Start runpq anew
print('Starting new instance of runPQ...')
print('---------------------------------')
os.system('nohup python runPQ.py &')
