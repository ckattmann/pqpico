# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 09:41:22 2014

@author: ckattmann
"""

import sys
import ctypes
import numpy as np
import datetime
import os

libname_christoph = 'C:\Program Files\Pico Technology\PicoScope6\ps4000A.dll'
libname_micha = 'C:\Program Files (x86)\Pico Technology\PicoScope6\ps4000A.dll'

# if 1, prints diagnostics to standard output
VERBOSE = 1
# If 1, generates profile.txtpicotec
PROFILING = 0 # Attention, may redirect standard print output, restart python kernel if output disappears

## Constants of PS2000.dll
# channel identifiers
PS4000_CHANNEL_A = 0
PS4000_CHANNEL_B = 1
PS2000_NONE = 5

# channel range values/codes
RANGE_20MV  = 1  # 20 mV
RANGE_50MV  = 2  # 50 mV
RANGE_100MV = 3  # 100 mV
RANGE_200MV = 4  # 200 mV
RANGE_500MV = 5  # 500 mV
RANGE_1V    = 6  # 1 V
RANGE_2V    = 7  # 2 V
RANGE_5V    = 8  # 5 V
RANGE_10V   = 9  # 10 V
RANGE_20V   = 10 # 20 V
RANGE_50V   = 11 # 50 V
RANGE_1ß0V   = 12 # 100 V
RANGE_20V   = 13 # 200 V

# map the range the the scale factor
RANGE_SCALE_MAP = {
RANGE_20MV  : 0.02,
RANGE_50MV  : 0.05,
RANGE_100MV : 0.1,
RANGE_200MV : 0.2,
RANGE_500MV : 0.5,
RANGE_1V    : 1.0,
RANGE_2V    : 2.0,
RANGE_5V    : 5.0,
RANGE_10V   : 10.0,
RANGE_20V   : 20.0,
}

#analog offset inital valiue
ANALOG_OFFSET_0V = 0# 0V offset

# Y Resolution Limits
MAX_Y = 32768
MIN_Y = -32767

# Flank Definitions for Triggering
PS2000_RISING = 0
PS2000_FALLING = 1

# Time Units
FEMTOSECONDS = 0
PICOSECONDS = 1
NANOSECONDS = 2
MICROSECONDS = 3
MILLISECONDS = 4
SECONDS = 5

                

# Set the correct dll as  LIBNAME
if sys.platform == 'win32':
    LIBNAME = libname_micha
    
else:
    LIBNAME = '/usr/local/lib/libps2000.so.2.0.7'
     
     
class Picoscope4000:
    
    def __init__(self):
        self.handle = None
        self.channels = [0,0]
        self.streaming_sample_interval = ctypes.c_uint(1000)
        self.streaming_buffer_length =100000
        
        # load the library
        if sys.platform == 'win32':
            self.lib = ctypes.windll.LoadLibrary(LIBNAME)
        else:
            self.lib = ctypes.cdll.LoadLibrary(LIBNAME)
        if not self.lib:
            print('---ERROR: LIB NOT FOUND---')
        # open the picoscope
        self.handle = self.open_unit()
     
        
# Basic Open and Close operations
    def open_unit(self):
        '''open interface to unit'''
        if VERBOSE == 1:
            print('Opening Picoscope')
            
        self.handle = ctypes.c_int16()
#        self.serial = ctypes.c_int32()
        self.serial = ctypes.create_string_buffer(16)
        picoStatus = self.lib.ps4000aOpenUnit(ctypes.byref(self.handle),None)
        print('PicoStatus: '+str(picoStatus))
        print('Handle is '+str(self.handle.value))
        
#       change Power Source Setup if applied to USB 2.0 / 1.0 with doubled-headed cable
        if picoStatus == 286:
            res=self.lib.ps4000aChangePowerSource(self.handle, picoStatus)
            if VERBOSE:
                print('Wrong Powersupply detected, try changing supply mode')
            if res >0:
                self.close_unit()
                if VERBOSE:
                    print('Failed to change USB Power Supply')
            else:
                if VERBOSE:
                    print('OK: Supply mode changed')
                    
        if self.handle.value == -1:
            print('Failed to open oscilloscope')
        elif self.handle.value == 0:
            print('No oscilloscope found')
        elif self.handle.value > 0:
            if VERBOSE:
                print('Oscilloscope handle: '+str(self.handle.value))
        else:
            print('WTFWTFWTF DEEP SHIT')
        return self.handle

    def close_unit(self):
        '''close the interface to the unit'''
        if VERBOSE == 1:
            print('Closing Picoscope')
        res = self.lib.ps4000aCloseUnit(self.handle.value)
        print(res)
        self.handle = None
        return res
        
      
    def get_handle(self):
        '''returns oscilloscope handle'''
        return self.handle
        
        
# Setup Operations
    def set_channel(self, channel=PS4000_CHANNEL_A, enabled=True, dc=True, vertrange=RANGE_1V,analogOffset=ANALOG_OFFSET_0V):
        '''Default Values: channel: Channel A | channel enabled: true | ac/dc coupling mode: dc(=true) | vertical range: 2Vpp'''
        try:
            res= self.lib.ps4000aSetChannel(self.handle, channel, enabled, dc, vertrange,analogOffset)
            if channel == PS4000_CHANNEL_A:
                self.channels[0] = 1
            elif channel == PS4000_CHANNEL_B:
                self.channels[1] = 1
            if VERBOSE == 1:
                print('Channel set to Channel '+str(channel))
                print('Status of setChannel '+str(res)+' (0 = PICO_OK)')
        finally:
            pass
        
    def set_trigger(self, source=PS2000_NONE, threshold=int(MAX_Y/2), direction=PS2000_RISING, delay=-25, auto_trigger_ms=1):
        self.lib.ps2000_set_trigger(self.handle, source, threshold, direction, delay, auto_trigger_ms)
        
    def construct_buffer_callback(self):
        if VERBOSE == 1:
            print('Constructing Buffer Callback')
        # Buffer callback C function template
        C_BUFFER_CALLBACK = ctypes.CFUNCTYPE( None,ctypes.POINTER(ctypes.POINTER(ctypes.c_int16)),ctypes.c_short,
                        ctypes.c_ulong,ctypes.c_short,ctypes.c_short,ctypes.c_ulong)
        
        # Callback function
        def get_buffer_callback(overviewBuffers,overflow,triggeredAt,triggered,auto_stop,nValues):
            
            #print('Callback for saving to disk')
            #create filename based on actual timestamp
            #filename = time.strftime("%Y%m%d_%H_%M_%S_%f.csv")
            filename=datetime.datetime.now()
            filename= filename.strftime("%Y%m%d_%H_%M_%S_%f")
            CH1='CH1_' + filename 
            #CH2='CH2_' + filename
            
            #cast 2d-pointer from c- callback into python pointer 
            ob = ctypes.cast(overviewBuffers,ctypes.POINTER(ctypes.POINTER(ctypes.c_short)))
            
            #create array from pointer data ob[0]-> CH1 ob[1]-> CH2
            streamed_data_CH1=np.fromiter(ob[0], dtype=np.short, count=nValues)
            #streamed_data_CH2=np.fromiter(ob[1], dtype=np.short, count=nValues)
                        
            #save array data into numpy fileformat
            path1 = os.path.normpath('C:\\Users\ckattmann\Documents\GitHub\pqpico\Data')+'/'+CH1
            #path2 = os.path.normpath('C:\\Users\ckattmann\Documents\GitHub\pqpico\Data')+'/'+CH2
                        
            np.save(path1,streamed_data_CH1)
            #np.save(path2,streamed_data_CH2)
            #print('File saved:',CH1,CH2)
            
            return 0
            
        return C_BUFFER_CALLBACK(get_buffer_callback)


  #Set Data Buffer for each channel of the PS4824 scope      
    def set_data_buffer(self,channel=PS4000_CHANNEL_A, bufferlength=101372,segmentIndex=0,mode=0):
        try:
            if channel == PS4000_CHANNEL_A: #channel A is set
                self.channel_A_buffer=(ctypes.c_short* bufferlength)()
                self.streaming_buffer_length=bufferlength
                res=self.lib.ps4000aSetDataBuffer(self.handle,channel,ctypes.byref(self.channel_A_buffer),self.streaming_buffer_length,segmentIndex,mode)
            if VERBOSE:
                print ('Try setting data buffer')
                print('Result: '+str(res)+' (0= PICO_OK)')
        finally:
            pass        
        
# Running and Retrieving Data NOTE: Bufferlength must be the same as set in set_data_buffer function
    def run_streaming(self,sampleIntervalTimeUnit=2, downSampleRatio=1,downSampleRatioMode=0):
        try:
            autoStop=0
            maxPreTriggerSamples=None
            maxPostTriggerSamples=None
            res = self.lib.ps4000aRunStreaming(self.handle,ctypes.byref(self.streaming_sample_interval),sampleIntervalTimeUnit,maxPreTriggerSamples,maxPostTriggerSamples,autoStop,downSampleRatio,downSampleRatioMode,self.streaming_buffer_length)
            # DOC of ps4000aRunStreaming(handler, pointer to sampleInterval, sampleIntervalTimeUnit, maxPretriggerSamples=none, maxPosttriggerSamples=none,autostop=none,downsamplingrate=no, downsamlingratiomode=0,bifferlength= must be the same as in setbuffer)
            if VERBOSE:
                print('Try starting streaming mode')
                print('Result: '+str(res)+' (0= PICO_OK, 64 = PICO_INVALID_SAMPLERATIO)')
                print(self.streaming_sample_interval)
        finally:
            pass
        
    def get_values(self, no_of_values=1000):
        buffer_a = (ctypes.c_short * no_of_values)()
        buffer_b = (ctypes.c_short * no_of_values)()
        buffer_c = None
        buffer_d = None
        overflow = ctypes.c_short()
        no_of_values = ctypes.c_long(no_of_values)
        
        self.lib.ps2000_get_values(self.handle, 
                                    ctypes.byref(buffer_a), ctypes.byref(buffer_b), 
                                    buffer_c, buffer_d, overflow,
                                    ctypes.byref(no_of_values))
                                    
        a = np.ctypeslib.as_array(buffer_a)
        b = np.ctypeslib.as_array(buffer_b)
        
        return np.vstack((a,b))
        
    def get_streaming_last_values(self, buffer_callback):
        res = self.lib.ps2000_get_streaming_last_values(self.handle, buffer_callback)
        return res
    
    def stop_sampling(self):
        try:
            res=self.lib.ps4000aStop(self.handle)
            if VERBOSE:
                print('Stopping sampling of Scope')
                print('Result: '+str(res)+' (0= PICO_OK)')
        finally:
            pass
        return res    

# Checking Buffer Overflow
    def overview_buffer_status(self):
        streaming_buffer_overflow = ctypes.c_bool(1)
        res = self.lib.ps2000_overview_buffer_status(self.handle, ctypes.byref(streaming_buffer_overflow))
        print('Overflow Error: ',str(res))
        return streaming_buffer_overflow.value
        


if __name__ == '__main__':
    try:
        #Set up Picoscope for continuous streaming
        pico = Picoscope4000()
        #pico.open_unit()
        pico.set_channel()
        pico.get_Timebase()
        pico.set_data_buffer()
        pico.run_streaming()
        pico.stop_sampling()
 
        
    finally:
        pico.close_unit()
        print('Picoscope closed')