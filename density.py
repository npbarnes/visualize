#!/usr/bin/python
import FortranFile as ff
import numpy as np
import matplotlib.pyplot as plt
import hybridReader as hr
from matplotlib.widgets import Slider, RadioButtons, Button
from matplotlib.colors import Normalize
import matplotlib.animation as animation

nx = 119
ny = 59
nz = 13
numProc = 4


Dreader = hr.hybridReader("c.np_3d_",nx,ny,nz,numProc)
zrange = Dreader.zrange
getSlice = Dreader.getSlice

Breader = hr.hybridReader("c.up_3d_",nx,ny,nz,numProc,True)
getProjection = Breader.getProjection

# Set initial slice and time
s0 = 'xy'
t0 = 0


# Make initial plot
fig = plt.figure()
ax = plt.axes()
density = plt.imshow(getSlice(t0,s0),interpolation='nearest',origin='lower') 
density.set_cmap('jet')
#density.set_cmap('seismic')
density.set_norm(Normalize(vmin=0,vmax=5*pow(10,13)))
plt.colorbar()

xcomp, ycomp = getProjection(t0,s0)
Bfield = plt.quiver(np.arange(60),np.arange(ny),xcomp,ycomp, color='white', scale=10000)

# Setup UI
# TODO: Save image, save animation
# TODO: Checkboxes for: Density, velocity streamlines, B field streamlines, etc.
rax = plt.axes([0.01,0.7,0.07,0.15])
radio = RadioButtons(rax, ('xy','xz','yz'))

axtime = plt.axes([0.1,0.04,0.65,0.03])
stime = Slider(axtime, 'Time', 0, 34, valinit=t0)

axscale = plt.axes([0.1,0.01,0.65,0.03])
sscale = Slider(axscale, 'Scale', 0, 10, valinit=5)

axplay = plt.axes([0.84,0.04,0.12,0.04])
bplay = Button(axplay, 'Play/Pause')
playing = True

# UI update functions
def radioUpdate(s):
    global s0 
    s0 = s
    density.set_data(getSlice(t0,s0))
    xcomp, ycomp = getProjection(t0,s0)
    Bfield.set_UVC(xcomp,ycomp)
    if(s0 == 'xy'):
        density.set_extent([0,60,0,ny])
    elif(s0 == 'xz'):
        density.set_extent([0,60,0,zrange])
    elif(s0 == 'yz'):
        density.set_extent([0,ny,0,zrange])
    plt.draw()
radio.on_clicked(radioUpdate)

def drawSlice():
    density.set_data(getSlice(t0,s0))

    xcomp, ycomp = getProjection(t0,s0)
    Bfield.set_UVC(xcomp,ycomp)

    #plt.draw()

def timeUpdate(val):
    global t0 
    global playing
    t0 = int(val)
    playing = False
    drawSlice()

stime.on_changed(timeUpdate)

def scaleUpdate(val):
    val=max(val,1)
    density.set_norm(Normalize(vmax=val*pow(10,13)))
    plt.draw()
sscale.on_changed(scaleUpdate)

def playUpdate(event):
    global playing
    playing = not playing
bplay.on_clicked(playUpdate)

def animUpdate(num):
    global t0
    global playing
    global stime
    if(playing):
        t0 = t0+1
        if(t0 > 34):
            t0=0
        drawSlice()
        stime.set_val(t0)
        #Set_val calls timeUpdate that turns playing off.
        playing = True
    return density, Bfield, 

ani = animation.FuncAnimation(fig, animUpdate, frames=34, interval=50, blit=True)

# Show figure
plt.show()
