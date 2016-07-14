#!/usr/bin/python3
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.io import readsav
from sys import argv
import numpy as np

dat = readsav(argv[1])
x = dat['x']
y = dat['y']
cnt = dat['cnt_arr']

ave = np.mean(cnt,axis=1)

plt.plot(y,ave)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Averaged counts')
plt.xlabel('Energy')

plt.show()
