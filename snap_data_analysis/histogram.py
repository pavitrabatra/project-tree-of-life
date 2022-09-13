import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("size.txt", usecols=(1,), unpack=True)
plt.hist(data, bins=100)
plt.ylabel('Frequency')
plt.xlabel('Network Size')
plt.show()
