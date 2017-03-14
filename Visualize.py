import numpy as np
import matplotlib.pyplot as plt

lung_data = np.load('lungdata-50-50-20.npy')

fig = plt.figure()

for num,each_slice in enumerate(lung_data[:1]):
    y = fig.add_subplot(4,5,num+1)
    y.imshow(each_slice)
plt.show()
