#!/usr/bin/python
import FortranFile as ff
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

nx = 119
ny = 59
nz = 13

def getData(filename):
    datafile = ff.FortranFile(filename)

    data =  []
    while(True):
        try:
            # For whatever reason m (time step) is output every other record.
            # Burn these values they won't be used.
            datafile.readReals()
            data.append(np.reshape(datafile.readReals(),[nx,ny,nz],'F'))
        except IOError:
            break

    return data

# Slicing multidimensional matrices seems to be non-trivial in python. Syntax is confusing.
# This solution works for now, but the 'right' way to do it would be to convert data to a
# numpy array and use numpy array slicing. I think that would be more elegant. 
def consSlice(data,t,axs):
    if(axs == 'xy'):
        return [[data[t][i][j][nz/2] for i in range(40,100)] for j in range(ny)]
    elif(axs == 'xz'):
        return [[data[t][i][ny/2][j] for i in range(40,100)] for j in range(nz)]
    elif(axs == 'yz'):
        return [[data[t][nx/2][i][j] for i in range(ny)] for j in range(nz)]


datalst = [getData("c.np_3d_"+str(i)+".dat") for i in range(11,14)]
# TODO: stitch together data from each process.
data = getData("c.np_3d_12.dat")




# Set initial slice and time
s0 = 'xy'
t0 = 0

# Make initial plot
fig, ax = plt.subplots()
obj = plt.imshow(consSlice(data,t0,s0),interpolation='nearest',origin='lower') 
plt.colorbar()

# Setup UI
# TODO: Play/Pause button
# TODO: Save image, save animation
# TODO: Checkboxes for: Density, velocity streamlines, B field streamlines, etc.
# TODO: arrange everything so it's not overlapping
rax = plt.axes([0.05,0.7,0.15,0.15])
radio = RadioButtons(rax, ('xy','xz','yz'))

axtime = plt.axes([0.25,0.1,0.65,0.03])
stime = Slider(axtime, 'Time', 0, 19, valinit=t0)

# UI update functions
def radioFunc(s):
    global s0 
    s0 = s
    obj.set_data(consSlice(data,t0,s0))
    if(s0 == 'xy'):
        obj.set_extent([0,60,0,ny])
    elif(s0 == 'xz'):
        obj.set_extent([0,60,0,nz])
    elif(s0 == 'yz'):
        obj.set_extent([0,ny,0,nz])
    plt.draw()
radio.on_clicked(radioFunc)

def update(val):
    global t0 
    t0 = int(stime.val)
    obj.set_data(consSlice(data,t0,s0))
    plt.draw()
stime.on_changed(update)

# Show figure
plt.show()
