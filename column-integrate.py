#!/usr/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from HybridReader2 import HybridReader2 as hr

flow_speed = 2.5 # km/s

h = hr('data', 'np_CH4')
qx,qy,qz = h.para['grid_points']
Lx = abs(qx[0] - qx[-1]) # km
Ly = abs(qy[0] - qy[-1]) # km
Lz = abs(qz[0] - qz[-1]) # km
m, data = h.get_timestep(21)
t = h.para['dt']*m
integrated = np.sum(data, axis=(1,2))*Ly*Lz/1000 # m

plt.axvline(-flow_speed*t, label="release point", color='black', linestyle='--')
plt.axvline(0, label="ballistic trajectory", color='black', linestyle='-.')

plt.plot(qx, integrated, label="ions", color='purple')
plt.xlabel("distance (km)")
plt.ylabel("integrated ion density (m^-1)")
plt.yscale('log')
plt.legend()
plt.show()
