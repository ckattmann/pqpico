import numpy as np
import PQTools as pq
import gc
import logging
pqLogger = logging.getLogger('pqLogger')

class ringarray2():
    def __init__(self, max_size = 2000000):
        self.ringBuffer = np.zeros(max_size)
        self.max_size = max_size
        self.size = 0

    # Are those two even necessary? Lets find out
    def get_data_view(self):
        return self.ringBuffer[:self.size]
    def get_index(self, index):
        return self.ringBuffer[index]
    
    def attach_to_back(self, data_to_attach):       
        self.check_buffer_overflow(data_to_attach.size)
        self.ringBuffer[self.size:self.size + data_to_attach.size] = data_to_attach
        self.size += data_to_attach.size

    def cut_off_front2(self,index):
        cut_of_data = self.ringBuffer[:index].copy() # is this copy necessary?
        self.ringBuffer[:self.size - index] = self.ringBuffer[index:self.size]
        self.size -= index
        return cut_of_data

    def cut_off_before_first_zero_crossing(self):
        first_zero_index = pq.detect_zero_crossings(self.ringBuffer)[0]
        #print('First detected zero_index: '+str(first_zero_index))
        self.ringBuffer[:self.size - first_zero_index] = self.ringBuffer[first_zero_index:self.size]
        #print('First value of new data  : '+str(self.ringBuffer[0]))
        self.size = self.size - first_zero_index
        return first_zero_index

    def cut_off_10periods(self):
        zero_indices = pq.detect_zero_crossings(self.ringBuffer)[:21]
        data_10periods = self.ringBuffer[:zero_indices[-1]]
        
        self.ringBuffer[:self.size - zero_indices[-1]] = self.ringBuffer[zero_indices[-1]:self.size]
        self.size = self.size - zero_indices[-1]

        #print('Last Value of data_10periods1 : '+str(data_10periods[-1]))
        #print('First Value of ringBuffer    :'+str(self.ringBuffer[-1]))
        return data_10periods, zero_indices

    def cut_off_10periods2(self):
        #self.plot_ringBuffer()
        zero_indices = np.zeros(21)
        zc = 0
        for i in xrange(1,21): 
            #print(str(i)+' : '+str(zero_indices))
            dataslice = self.ringBuffer[zero_indices[i-1] + 9500 : zero_indices[i-1] + 10500]
            #print('dataslice border lower: '+str(zero_indices[i-1] + 9500))
            #print('dataslice border upper: '+str(zero_indices[i-1] + 10500))
            #print(str(pq.detect_zero_crossings(dataslice)))
            zero_indices[i] = zero_indices[i-1] + pq.detect_zero_crossings(dataslice) + 9500
        #print('ZC: '+str(zero_indices))
        data_10periods = self.ringBuffer[:zero_indices[-1]].copy()
        
        self.ringBuffer[:self.size - zero_indices[-1]] = self.ringBuffer[zero_indices[-1]:self.size]
        self.size = self.size - zero_indices[-1]

        #print('Last Value of data_10periods2 : '+str(data_10periods[-1]))
        #print('First Value of ringBuffer    :'+str(self.ringBuffer[-1]))
        #print('Zero_indices : '+str(zero_indices))
        return data_10periods, zero_indices

    def attach_to_front(self, data_to_attach):
        self.check_buffer_overflow(data_to_attach.size)
        # Move current ringBuffer content out of the way
        self.ringBuffer[data_to_attach.size : data_to_attach.size + self.size] = self.ringBuffer[:self.size]
        self.size += data_to_attach.size
        # Add new content to front
        self.ringBuffer[:data_to_attach.size] = data_to_attach

    def check_buffer_overflow(self,size_to_attach):
        while self.size + size_to_attach > self.max_size:
            print('==Reallocating Buffer to '+str(self.max_size * 1.7)+'==')
            pqLogger.warning('Reallocating Buffer to '+str(self.max_size * 1.7))
            # Allocate new buffer, 1.7 times bigger than the old one
            self.max_size *= 1.7 # if this resolves to float no problem, np.zeros can handle it
            newRingBuffer = np.zeros(self.max_size)
            newRingBuffer[:self.size] = self.ringBuffer[:self.size]
            self.ringBuffer = newRingBuffer
            # Manually call garbage collection, does this make sense?
            #gc.collect()


    # Helper Functions:
    def plot_ringBuffer(self):
        import matplotlib.pyplot as plt
        plt.plot(self.ringBuffer[:self.size])
        plt.grid(True)
        plt.show()
