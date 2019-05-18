import os,sys
import numpy as np
import matplotlib.pyplot as plt

filename = 'preprocessed_nchwRGB.txt'
assert len(sys.argv)>1, str(len(sys.argv))
filename = sys.argv[1]

x = np.asarray([i for i in range(288*352*3)])

with open(filename) as f:
    data = f.read().strip().split()
    data = np.asarray([int(i,16) for i in data])
print(data.shape)
plt.plot(x,data)
plt.show()
