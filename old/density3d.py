#!/usr/bin/python
import FortranFile as ff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d
#import matplotlib.animation as animation
from matplotlib.widgets import Slider, RadioButtons
import hybridReader as hr
import matplotlib.cm as cm
from matplotlib.colors import Normalize

nx = 119
ny = 59
nz = 13
numProc = 4

reader = hr.hybridReader("c.np_3d_",nx,ny,nz,numProc)
data = reader.data

def findAveDensity():
    ave = 0
    n = 1
    for time in data:
        for xyz in time:
            for yz in xyz:
                for z in yz:
                    ave = ave + (z-ave)/n
                    n = n+1
    return ave

def findPoints(t,threshold):
    return [(x,y,z) for x in range(len(data[t]))
                    for y in range(len(data[t][x]))
                    for z in range(len(data[t][x][y]))
                    if (data[t][x][y][z] > threshold) ]


# Set initial slice and time
s0 = 'xy'
t0 = 30

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

ave = findAveDensity()
numframes = len(data)
pointslst = [findPoints(t, ave*3.5) for t in range(numframes)]
xs,ys,zs = zip(*pointslst[t0])
thing = zip(xs,ys,zs)


colors = [data[t0][x][y][z] for x,y,z in thing]
norm = Normalize(vmin=min(colors),vmax=max(colors))
cmap = cm.hot
m = cm.ScalarMappable(norm=norm, cmap=cmap)
rgbacolors = [m.to_rgba(c) for c in colors]
rgbcolors = [(r,g,b) for (r,g,b,a) in rgbacolors]

print(len(xs),len(ys),len(zs),len(rgbcolors))

sc, = ax.plot(xs,ys,zs,c='red', marker = 'o', linestyle='None')

#def update(num):
#    global t0
#    t0 = t0+1
#    x,y,z = zip(*pointslst[num])
#    # Can't set_data in 3-d so instead set 2-d data then
#    # use set_3d_properties to add z component
#    sc.set_data(x,y)
#    sc.set_3d_properties(z)
#    return sc,
#
#ani = animation.FuncAnimation(fig, update, frames=numframes, interval=50, blit=False)


plt.show()
