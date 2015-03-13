import numpy as np
import PQTools as pq
import time
import logging

import matplotlib.pyplot as plt

class ring_array_global_data():
    def __init__(self, size = 2000000):
        self.ringBuffer = np.array(np.zeros(size))
        self.zero_indices = np.array(np.zeros(2000), dtype=np.int32)
        self.size = 0 
        self.size_zero_indices = 0

    def get_data_view(self):
        return self.ringBuffer[:self.size]

    def get_index(self, index):
        return self.ringBuffer[index]
    
    def attach_to_back(self, data_to_attach):
        zero_indices_to_attach = np.array(pq.detect_zero_crossings(data_to_attach))

        #print('----------------------------------')
        #print('zero_indices : '+str(self.zero_indices))
        #print('size_zero_indices : '+str(self.size_zero_indices))
        #print('zero_indices_to_attach.size : '+str(zero_indices_to_attach.size))
        #print('zero_indices_to_attach'+str(zero_indices_to_attach))
        #print('self.size : '+str(self.size))
    
        # Check if zero crossing gets lost in between data and data_to_attach
        # if one negative and one positive...:
        if data_to_attach[0] != 0 and np.sign(self.ringBuffer[-1]) == -np.sign(data_to_attach[0]):
            # ... there should one or sometimes two zero crossings
            # if not,...:
            if zero_indices_to_attach[0] > 10 and self.zero_indices[-1] < self.size - 10:
                # force create zero crossing
                print('Force Create Zero Crossing')
                pqLogger.WARNING('Force created zero crossing between snippets')
                zero_indices_to_attach = np.append(0, zero_indices_to_attach)

        # Check for double zero crossings at the end of data_to_attach caused by bad filtering
        if zero_indices_to_attach.size > 1 and zero_indices_to_attach[-1] - zero_indices_to_attach[-2] < 10:  
            #print('Double ZC at end of data_to_attach')
            zero_indices_to_attach = np.delete(zero_indices_to_attach,-1)

        # TEMP for plotting
        rb = self.ringBuffer[:self.size]
        #    

        self.zero_indices[self.size_zero_indices:self.size_zero_indices + zero_indices_to_attach.size] = zero_indices_to_attach + self.size

        #print('zero_indices : '+str(self.zero_indices))
        #print('size_zero_indices : '+str(self.size_zero_indices))

        #print('distance between ZC of adjacent segments'+str(self.zero_indices[self.size_zero_indices] - self.zero_indices[self.size_zero_indices-1]))

        # Check for two zero crossings directly after attachment
        if zero_indices_to_attach.size > 0 and self.zero_indices[self.size_zero_indices] - self.zero_indices[self.size_zero_indices-1] < 10:
            #print('Close zero_indices')
            #print('last ZC of old data : '+str(self.zero_indices[self.size_zero_indices-1]))
            #print('first ZC of data_to_attach : '+str(self.zero_indices[self.size_zero_indices]))
            self.zero_indices = np.delete(self.zero_indices,self.size_zero_indices)
            self.size_zero_indices -= 1

        self.ringBuffer[self.size:self.size + data_to_attach.size] = data_to_attach
        self.size += data_to_attach.size
        self.size_zero_indices += zero_indices_to_attach.size

        if any(np.diff(self.zero_indices[:self.size_zero_indices]) > 11111): # < 45 Hz
            print('Distance between two zero crossings: '+str(max(np.diff(self.zero_indices[:self.size_zero_indices]))))
            print(str(self.zero_indices[:self.size_zero_indices]))
            plt.plot(data_to_attach,'b')
            plt.plot(pq.moving_average2(data_to_attach),'r')
            plt.grid()
            plt.show()
            plt.plot(rb,'b')
            plt.plot(pq.moving_average2(rb),'r')
            plt.grid()
            plt.show()

        if any(np.diff(self.zero_indices[:self.size_zero_indices]) < 9090): # < 55 Hz
            print('Distance between two zero crossings: '+str(min(np.diff(self.zero_indices[:self.size_zero_indices]))))
            print(str(self.zero_indices[:self.size_zero_indices]))
            plt.plot(data_to_attach,'b')
            plt.plot(pq.moving_average2(data_to_attach),'r')
            plt.show()
            plt.plot(rb,'b')
            plt.plot(pq.moving_average2(rb),'r')
            plt.show()

    def cut_off_front(self, index, zero_crossing):
        #print(str(type(self.ringBuffer))+' is '+str(self.ringBuffer)+' and has len '+str(self.size) )
        #print(str(type(index))+' is '+str(index))        
        self.ringBuffer = np.roll(self.ringBuffer,-index)
        self.zero_indices = np.roll(self.zero_indices,-zero_crossing)-index        
        self.size -= index
        self.size_zero_indices -= zero_crossing
        #print('Size of ringBuffer: '+ str(self.size))
        #print('Size of zero_indices: '+ str(self.size_zero_indices))
        return self.ringBuffer[-index:]
        
    def cut_off_front2(self, index, zero_crossing):
        #print(str(type(self.ringBuffer))+' is '+str(self.ringBuffer)+' and has len '+str(self.size) )
        #print(str(type(index))+' is '+str(index))        
        cut_of_data = self.ringBuffer[:index].copy()
        self.ringBuffer[:self.size-index] = self.ringBuffer[index:self.size]    
        if zero_crossing == 0:
            self.zero_indices[:self.size_zero_indices] = self.zero_indices[:self.size_zero_indices]-index
        else:
            self.zero_indices[:self.size_zero_indices - zero_crossing] = self.zero_indices[zero_crossing:self.size_zero_indices]-index        
        self.size -= index
        self.size_zero_indices -= zero_crossing
        #print('Size of ringBuffer: '+ str(self.size))
        #print('Size of zero_indices: '+ str(self.size_zero_indices))
        return cut_of_data
    
    def get_zero_indices(self):
        #print(str(type(self.zero_indices[:self.size_zero_indices]))+' is '+str(self.zero_indices[:self.size_zero_indices]))
        return self.zero_indices[:self.size_zero_indices]
        
    def attach_to_front(self, data_to_attach):
            self.ringBuffer[data_to_attach.size:] = self.ringBuffer[:-data_to_attach.size]
            self.ringBuffer[:data_to_attach.size] = data_to_attach      
            self.size += data_to_attach.size
            self.size_zero_indices = pq.detect_zero_crossings(self.ringBuffer[:self.size]).size
            self.zero_indices[:self.size_zero_indices] = pq.detect_zero_crossings(self.ringBuffer[:self.size])


    def attach_to_front2(self, data_to_attach):
        self.ringBuffer[data_to_attach.size : data_to_attach.size + self.size] = self.ringBuffer[:self.size]
        self.ringBuffer[:data_to_attach.size] = data_to_attach
        self.size += data_to_attach.size

        new_zero_indices = pq.detect_zero_crossings(data_to_attach)
        
        self.zero_indices[new_zero_indices.size : new_zero_indices.size + self.size_zero_indices] = self.zero_indices[:self.size_zero_indices]
        self.zero_indices[:new_zero_indices.size] = new_zero_indices
        self.size_zero_indices += new_zero_indices.size

class ring_array():
    def __init__(self, size = 2000000):
        self.ringBuffer = np.array(np.zeros(size))
        self.size = 0

    def get_data_view(self):
        return self.ringBuffer[:self.size]

    def get_index(self, index):
        return self.ringBuffer[index]
    
    def attach_to_back(self, data_to_attach):       
        self.ringBuffer[self.size:self.size + data_to_attach.size] = data_to_attach
        self.size += data_to_attach.size
        
    def cut_off_front(self,index):
        self.ringBuffer = np.roll(self.ringBuffer,-index)
        self.size -= index
        #print('Size of ringBuffer: '+ str(self.size))
        return self.ringBuffer[-index:].copy()
    
    def cut_off_front2(self,index):
        cut_of_data = self.ringBuffer[:index].copy()
        self.ringBuffer[:-index] = self.ringBuffer[index:].copy()       
        self.size -= index
        #print('Size of ringBuffer: '+ str(self.size))
        return cut_of_data

if __name__ == '__main__':
    a = ring_array()
    #print(str(a.get_data_view()))
    #b = np.array([4,5,6])
    b = np.load('20150123_17_50_52_201410.npy')
    #print(str(b))
    a.attach_to_back(b)
    a.attach_to_back(b)    
    for i in range(5):
        #print(str(a.get_data_view()))
        #a.attach_to_back(b)
        #print(str(a.get_zero_indices()))
        t1 = time.time() 
        #zero_crossing = a.get_zero_indices()
        w = a.cut_off_front(1000000)
        t2 = time.time()
        c = a.get_data_view()
        #t = a.get_zero_indices()
        print(str(t2-t1))

def test_ring_array():
    
    a = ring_array()
    b = np.array([4,5,6])
    a.attach_to_back(b)
