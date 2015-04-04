import sys

try:
    a = 3/0
except Exception, e:
    print(str(dir(e)))
    print('Error    : '+str(e))
    print('Message  : '+str(e.message))
    print('Args     : '+str(e.args))
    print('lastTB   : '+str(sys.last_traceback))
a = 3/0
