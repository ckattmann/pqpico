import sys
import traceback
import script_with_error
import smtplib
import datetime
from prettytable import PrettyTable

try:
    #script_with_error.do_shit()
    a = 3/0
except Exception, e:
    locs = locals().copy()
    #print(str(dir(e)))
    #print('Error    : '+str(e))
    #print('Message  : '+str(e.message))
    #print(str(sys.exc_info()[2]))
    #print('lastTB   : '+str(dir(sys.last_traceback)))
    #print('lastTB   : '+str(sys.last_traceback))
    #print(str(traceback.format_exc()))
    with open('.mailinfo','r') as f:
        mailinfo = f.readlines()
    mailinfo = [x.strip() for x in mailinfo]
    datetime.datetime.now().strftime('%A %x %X:%f')

    s = smtplib.SMTP('smtp.gmail.com')
    s.ehlo()
    s.starttls()
    s.login(mailinfo[0], mailinfo[1])
    x = PrettyTable(['Variable','Content'])
    x.align['Variable'] + 'l'
    x.align['Content'] + 'l'
    x.padding_width = 1
    for k,v in locs.iteritems():
        x.add_row([str(k),str(v)])
    #print(x)
    print('Sending from '+str(mailinfo[0])+' to '+str(mailinfo[2]))
    msg = '\nError in PQpico at ' + str(datetime.datetime.now().strftime('%A %x %X:%f')) +'\n\n'+ str(traceback.format_exc()) + '\n\n' + str(x)
    print('------------')
    print(str(msg))
    print('------------')
    #s.sendmail(mailinfo[0], mailinfo[2], str(msg))
    s.quit()
finally:
    print('This is the last thing')
