import numpy as np
import RingArray2 as ra

r = ra.ring_array(10)

r.attach_to_back(np.array([1,2,3,4,5,6,7,8,9]))
print(str(r.get_data_view()))

r1 = r.cut_off_front2(4)
print(str(r1))
print(str(r.get_data_view()))


r.attach_to_front(np.array([11,12,13]))
print('After attach_to_front : '+str(r.get_data_view()))

# Cause buffer overflow
print(r.size)
print(r.max_size)
r.attach_to_back(np.array([10,11,12,13,14,15,16]))
print(str(r.get_data_view()))
print(r.size)
print(r.max_size)

