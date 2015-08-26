#!/usr/bin/python
import FortranFile as ff
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

nx = 119
ny = 59
nz = 13
numProc = 4
zrange = (nz-2)*numProc

def getData(filename):
    datafile = ff.FortranFile(filename)

    data = []
    while(True):
        try:
            # For whatever reason m (time step) is output every other record.
            # Burn these values they won't be used.
            datafile.readReals()
            data.append(datafile.readReals().tolist())
        except IOError:
            break

    return np.array(data)


# Construct a 2-d slice to view.
def consSlice(data,t,axs):
    if(axs == 'xy'):
        #return data[t,40:100,:,nz/2]
        return [[data[t][i][j][nz*3/2] for i in range(40,100)] for j in range(ny)]
    elif(axs == 'xz'):
        #return data[t,40:100,ny/2,:]
        return [[data[t][i][ny/2][j] for i in range(40,100)] for j in range(zrange)]
    elif(axs == 'yz'):
        #return data[t,nx/2,:,:]
        return [[data[t][nx/2][i][j] for i in range(ny)] for j in range(zrange)]


# taken in reversed order to make them go from top to bottom.
datalst = [getData("c.np_3d_"+str(i)+".dat") for i in reversed(range(11,11+numProc))]
timesteps = len(datalst[0])

# TODO: This function is very inefficient, make the whole thing in numpy arrays.
def shapeData(datalst):
    ret = []
    for i in range(timesteps):
        tmp = []
        for proc in datalst:
            tmp.append(proc[i][:-2*nx*ny])
        # flatten
        tmp = [val for xyz in tmp for val in xyz]
        # reshape to 3-d grid
        tmp = np.reshape(tmp, [nx,ny,zrange],'F').tolist()
        # timestep is ready to be returned
        ret.append(tmp)
    return np.array(ret)


#### Main Program ####
data = shapeData(datalst)

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
        obj.set_extent([0,60,0,zrange])
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
