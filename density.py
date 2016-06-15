#!/usr/bin/python
import FortranFile as ff
import numpy as np
import matplotlib.pyplot as plt
import HybridReader as hr
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from matplotlib.colors import Normalize
import matplotlib.animation as animation

import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.register_cmap(name='inferno', cmap=cmaps.inferno)
plt.register_cmap(name='plasma', cmap=cmaps.plasma)


dov = False

nx = 150
ny = 100
nz = 10
numProc = 10

Dreader = hr.hybridReader("c.np_3d_",nx,ny,nz,numProc)
zrange = Dreader.zrange
getSlice = Dreader.getSlice

if(dov):
    Vreader = hr.hybridReader("c.up_3d_",nx,ny,nz,numProc,True)
    getProjection = Vreader.getProjection

Rp = 1.216
lengthx = nx*Rp
plutox = (int(nx/2)+70)*Rp
heightz = zrange*Rp
heighty = ny*Rp

# Globals
s0 = 'xz'
t0 = 0
d0 = 0.5
showDensity = True
showVelocity = True
playing = True


# Make initial plot
fig = plt.figure()
ax = plt.axes()
density = plt.imshow(getSlice(t0,s0,d0),interpolation='none',origin='lower') 
density.set_extent([-plutox,lengthx-plutox,-heightz/2,heightz/2])
density.set_cmap(cmaps.viridis)
#density.set_cmap(cmaps.inferno)
#density.set_cmap(cmaps.plasma)
#density.set_cmap("PuBuGn")
#density.set_norm(Normalize(vmin=0,vmax=5*pow(10,13)))
#density.set_norm(Normalize(vmin=0,vmax=10*pow(10,13)))
#density.set_norm(Normalize(vmin=-20,vmax=20))
#density.set_norm(Normalize(vmin=0,vmax=30))
#density.set_norm(Normalize(vmin=1.8,vmax=4.5))
plt.colorbar()

#def sparce(sdf,n):
#    return sdf[::n,::n]
def sparce(sdf,n):
    ret = np.zeros((len(sdf) ,len(sdf[0])))
    for x in range(0,len(sdf),n):
        for y in range(0,len(sdf[0]),n):
            ret[x,y] = sdf[x,y]

    return ret

if(dov):
    xcomp, ycomp = getProjection(t0,s0,d0)
    x = np.linspace(0,nx,len(xcomp)/10)
    y = np.linspace(0,zrange,len(xcomp[0]/10))
    xv, yv = np.meshgrid(x,y)
    #Bfield = plt.quiver(xcomp,ycomp, color='white', scale=50000)
    Bfield = plt.quiver(sparce(xcomp,10),sparce(ycomp,10), color='white',scale=8000, minlength=0)

# Setup UI
# TODO: Save image, save animation
# TODO: Checkboxes for: Density, velocity streamlines, B field streamlines, etc.
rax = plt.axes([0.01,0.7,0.07,0.15])
radio = RadioButtons(rax, ('xy','xz','yz'))

axdepth = plt.axes([0.1,0.06,0.65,0.02])
sdepth = Slider(axdepth, 'Depth', 0, 1, valinit=0.5)

axtime = plt.axes([0.1,0.03,0.65,0.02])
stime = Slider(axtime, 'Time', 0, Dreader._getTimesteps(), valinit=t0)

axscale = plt.axes([0.1,0.0,0.65,0.02])
sscale = Slider(axscale, 'Scale', 0, 50, valinit=20)

axplay = plt.axes([0.84,0.04,0.12,0.04])
bplay = Button(axplay, 'Play/Pause')

axDisp = plt.axes([0.9,0.7,0.1,0.09])
cDisp = CheckButtons(axDisp,["up","np"],[showVelocity,showDensity])

# UI update functions
def radioUpdate(s):
    global s0 
    s0 = s
    density.set_data(getSlice(t0,s0,d0))
    if(dov):
        xcomp, ycomp = getProjection(t0,s0,d0)
        Bfield.set_UVC(sparce(xcomp,10),sparce(ycomp,10))

    if(s0 == 'xy'):
        #density.set_extent([-plutox,lengthx-plutox,-heighty/2,heighty/2])
        density.set_extent([0,nx,0,ny])
    elif(s0 == 'xz'):
        #density.set_extent([-plutox,lengthx-plutox,-heightz/2,heightz/2])
        density.set_extent([0,nx,0,zrange])
    elif(s0 == 'yz'):
        #density.set_extent([-heighty/2,heighty/2,-heightz/2,heightz/2])
        density.set_extent([0,ny,0,zrange])
    plt.draw()
radio.on_clicked(radioUpdate)

def drawSlice():
    global showDensity
    global showVelocity
    density.set_data(getSlice(t0,s0,d0))

    if(dov):
        xcomp, ycomp = getProjection(t0,s0,d0)
        #Bfield.set_UVC(xcomp,ycomp)
        Bfield.set_UVC(sparce(xcomp,10),sparce(ycomp,10))
        Bfield.set_visible(showVelocity)
    
    density.set_visible(showDensity)
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
    #density.set_norm(Normalize(vmax=val*pow(10,13)))
    #density.set_norm(Normalize(vmax=val))
    #density.set_norm(Normalize(vmin=-val))
    plt.draw()
sscale.on_changed(scaleUpdate)

def depthUpdate(val):
    global d0
    d0 = max(val,0)
    d0 = min(d0,1)
    drawSlice()
sdepth.on_changed(depthUpdate)

def playUpdate(event):
    global playing
    playing = not playing
bplay.on_clicked(playUpdate)

def dispUpdate(label):
    global showDensity
    global showVelocity
    global dov
    if(label == 'np'):      
        showDensity = not showDensity
    elif (label == 'up'):   
        showVelocity = not showVelocity
    drawSlice()
cDisp.on_clicked(dispUpdate)

def animUpdate(num):
    global t0
    global playing
    global stime
    if(playing):
        t0 = (t0+1) % Dreader._getTimesteps()
        drawSlice()
        stime.set_val(t0)
        #Set_val calls timeUpdate that turns playing off.
        playing = True
    if(dov):
        return density, Bfield,
    else:
        return density,

def init():
    if(dov):
        return density, Bfield,
    else:
        return density,
ani = animation.FuncAnimation(fig, animUpdate, init_func=init, frames=34, interval=3, blit=True)

#ani.save('testvideo.mp4', writer='avconv')
# Show figure
plt.show()
