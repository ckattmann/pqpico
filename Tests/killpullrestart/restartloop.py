import os
import signal

with open('.processid','r') as f:
    processid = f.read().strip()
    print(processid + ' shall be killed!')
#os.system('kill -2 '+str(processid))
os.kill(int(processid), signal.SIGINT)


