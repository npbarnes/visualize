#!/usr/bin/python3
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.io import readsav
from sys import argv

dat = readsav(argv[1])
x = dat['x']
y = dat['y']
cnt = dat['cnt_arr']

plt.pcolormesh(x,y,cnt, norm=LogNorm())
plt.yscale('log')
plt.colorbar()

plt.show()
