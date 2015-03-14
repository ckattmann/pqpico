import numpy as np
import PQTools as pq


n = np.array([1,2,3,4,5,6,7,8,9])
print(str(n))
print(str(pq.moving_average2(n,7)))

n = np.array([1,2,3,4,5,6,7,8,9])[::-1]
print(str(n))
print(str(pq.moving_average2(n,7)))
