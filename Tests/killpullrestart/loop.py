import time
import os

with open('.processid','w') as f:
    pid_string = str(os.getpid())
    f.write(pid_string)
    print(str(pid_string))

li = 0
try:
    while True:
        li+=1
        time.sleep(2)
        with open('report.txt','a') as f:
            f.write('Loop iteration '+str(li)+' ... processid='+str(os.getpid())+'\n')
        print('2 seconds have passed')
except KeyboardInterrupt as e:
    with open('report.txt','a') as f:
        f.write('Keyboardinterrupt received!')

finally:
    with open('report.txt','a') as f:
        f.write('Finally!')
