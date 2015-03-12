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

libname_christoph = 'C:\Program Files\Pico Technology\PicoScope6\ps4000a.dll'
libname_micha = 'C:\Program Files (x86)\Pico Technology\PicoScope6\ps4000a.dll'

# if 1, prints diagnostics to standard output
VERBOSE = 1
# If 1, generates profile.txt
PROFILING = 0 # Attention, may redirect standard print output, restart python kernel if output disappears

## Constants of PS2000.dll
# channel identifiers
PS2000_CHANNEL_A = 0
PS2000_CHANNEL_B = 1
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
    LIBNAME = 'C:\Program Files\Pico Technology\PicoScope6\PS2000.dll'
else:
    LIBNAME = '/usr/local/lib/libps2000.so.2.0.7'
     
     
class Picoscope:
    
    def __init__(self):
        self.handle = None
        self.channels = [0,0]

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
        self.handle = self.lib.ps2000_open_unit()
        if self.handle == -1:
            print('Failed to open oscilloscope')
        elif self.handle == 0:
            print('No oscilloscope found')
        elif self.handle > 0:
            if VERBOSE:
                print('Oscilloscope handle: '+str(self.handle))
        else:
            print('WTFWTFWTF DEEP SHIT')
        return self.handle

    def close_unit(self):
        '''close the interface to the unit'''
        if VERBOSE == 1:
            print('Closing Picoscope')
        res = self.lib.ps2000_close_unit(self.handle)
        self.handle = None
        return res
        
    def get_handle(self):
        '''returns oscilloscope handle'''
        return self.handle
        
        
# Setup Operations
    def set_channel(self, channel=PS2000_CHANNEL_A, enabled=True, dc=True, vertrange=RANGE_1V):
        '''Default Values: channel: Channel A | channel enabled: true | ac/dc coupling mode: dc(=true) | vertical range: 2Vpp'''
        try:
            self.lib.ps2000_set_channel(self.handle, channel, enabled, dc, vertrange)
            if channel == PS2000_CHANNEL_A:
                self.channels[0] = 1
            elif channel == PS2000_CHANNEL_B:
                self.channels[1] = 1
            if VERBOSE == 1:
                print('Channel set to Channel '+str(channel))
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

        
        
# Running and Retrieving Data
    def run_streaming(self, sample_interval_ms=10, max_samples=100, windowed=0):
        '''Default Values: Sample Rate: 10ms | Max Samples: 100 | Windowed: No (=0) '''
        try:
            res = self.lib.ps2000_run_streaming(self.handle, sample_interval_ms, max_samples, windowed)
        finally:
            pass
        return res
        
    def run_streaming_ns(self, sample_interval=10, time_units=MICROSECONDS,
                         max_samples=1000, auto_stop=0, noOfSamplesPerAggregate=1, overview_buffer_size=800000):
        '''Starts recording data at given speed'''
        if VERBOSE == 1:
            print('Starting Streaming at '+str(sample_interval)+' us')
        self.lib.ps2000_run_streaming_ns(self.handle, sample_interval, time_units, 
                                         max_samples, auto_stop, noOfSamplesPerAggregate, overview_buffer_size)
                                         
        
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

# Checking Buffer Overflow
    def overview_buffer_status(self):
        streaming_buffer_overflow = ctypes.c_bool(1)
        res = self.lib.ps2000_overview_buffer_status(self.handle, ctypes.byref(streaming_buffer_overflow))
        print('Overflow Error: ',str(res))
        return streaming_buffer_overflow.value


if __name__ == '__main__':
    try:
        import matplotlib.pyplot as plt
        
        if PROFILING:
            import cProfile
            pr=cProfile.Profile()
            pr.enable()
            
        # Picoscope continuous streaming setup parameters
        resolution = 1 #in ms        
        samplepoints = 40 # samples collected, max is 60000
        vertrange = RANGE_50MV
        
        # Numpy and Matplotlib init
        plotarray = np.array([])
        yticklocations = [MIN_Y, MIN_Y/2, 0, MAX_Y/2, MAX_Y]
        ytickstrings = [    str(-RANGE_SCALE_MAP[vertrange])+' V', 
                            str(-RANGE_SCALE_MAP[vertrange]/2)+' V', 
                            str(0*RANGE_SCALE_MAP[vertrange])+' V', 
                            str(RANGE_SCALE_MAP[vertrange]/2)+' V',
                            str(RANGE_SCALE_MAP[vertrange])+' V']
                            
        xticklocations = np.arange(0,(samplepoints+1),5)
        xtickstrings = [str(x)+' ms' for x in xticklocations]
        
        #Set up Picoscope for continuous streaming
        pico = Picoscope()
        pico.set_channel(channel = PS2000_CHANNEL_A, vertrange=vertrange)
        pico.run_streaming(sample_interval_ms=resolution, max_samples=samplepoints)
        plt.ion()
        plt.pause(0.1)
        
        #Get values and plot
        for i in range(4):
            if VERBOSE:
                if pico.channels[0] == 1:
                    print('Channel A active')
                if pico.channels[1] == 1:
                    print('Channel B active')
                
            b = pico.get_values()
            b = b[0,:]
            b = np.trim_zeros(b)
            print(str(b.size)+' sample points collected')
            
            plotarray = np.append(plotarray,b)
            plotarray = plotarray[-samplepoints:]
            plt.plot(plotarray)
            plt.yticks(yticklocations, ytickstrings)
            plt.xticks(xticklocations, xtickstrings)
            plt.grid(True)
            plt.hold(False)
            
            plt.draw()
            plt.pause(0.001)
            plt.grid(1)
            
        if PROFILING:
            pr.disable()
            sys.stdout = open('profile.txt','w')
            pr.print_stats(sort='cumtime')
        
    finally:
        pico.close_unit()
        print('Picoscope closed')
