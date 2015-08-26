#!/usr/bin/python
import FortranFile as ff
import numpy as np
import matplotlib.pyplot as plt
import hybridReader as hr
from matplotlib.widgets import Slider, RadioButtons

nx = 119
ny = 59
nz = 13
numProc = 4

reader = hr.hybridReader(nx,ny,nz,numProc)
zrange = reader.zrange
getSlice = reader.getSlice

# Set initial slice and time
s0 = 'xy'
t0 = 0


# Make initial plot
fig, ax = plt.subplots()
obj = plt.imshow(getSlice(t0,s0),interpolation='nearest',origin='lower') 
plt.colorbar()

# Setup UI
# TODO: Play/Pause button
# TODO: Save image, save animation
# TODO: Checkboxes for: Density, velocity streamlines, B field streamlines, etc.
rax = plt.axes([0.01,0.7,0.07,0.15])
radio = RadioButtons(rax, ('xy','xz','yz'))

axtime = plt.axes([0.25,0.03,0.65,0.03])
stime = Slider(axtime, 'Time', 0, 19, valinit=t0)

# UI update functions
def radioFunc(s):
    global s0 
    s0 = s
    obj.set_data(getSlice(t0,s0))
    if(s0 == 'xy'):
        obj.set_extent([0,60,0,ny])
    elif(s0 == 'xz'):
        obj.set_extent([0,60,0,zrange])
    elif(s0 == 'yz'):
        obj.set_extent([0,ny,0,zrange])
    plt.draw()
radio.on_clicked(radioFunc)

def update(val):
    global t0 
    t0 = int(stime.val)
    obj.set_data(getSlice(t0,s0))
    plt.draw()
stime.on_changed(update)

# Show figure
plt.show()
