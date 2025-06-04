import matplotlib.pyplot as plt
import numpy as np

gs = []

f = open("rsamp.dat"    , "r")
for line in f.readlines():
    gs.append(float(line))
    
print (np.mean(gs) , np.std(gs))
plt.hist(gs , bins = 50)
plt.show()