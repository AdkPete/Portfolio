import matplotlib.pyplot as plt
import numpy as np
gs = []

f = open("data1.dat"    , "r")
for line in f.readlines():
    gs.append(float(line))
    
print (np.mean(gs) , np.std(gs))
plt.plot(range(len(gs)) , gs)
plt.show()