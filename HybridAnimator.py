#!/usr/bin/python
import FortranFile as ff
import numpy as np
import matplotlib.pyplot as plt
import HybridReader2 as hr
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from matplotlib.colors import Normalize, LogNorm
import matplotlib.animation as animation
from scipy.interpolate import RegularGridInterpolator

import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.register_cmap(name='inferno', cmap=cmaps.inferno)
plt.register_cmap(name='plasma', cmap=cmaps.plasma)

class HybridAnimator():
    def __init__(self,prefix,variable):
        # Read hybrid files
        self.prefix = prefix
        self.variable = variable
        self.h = hr.HybridReader2(self.prefix,self.variable)

        # Grab some of the parameters directly from there to be used
        # to compute some extra parameters
        qx = self.h.para['qx']
        qy = self.h.para['qy']
        qzrange = self.h.para['qzrange']
        nx = self.h.para['nx']
        ny = self.h.para['ny']
        zrange = self.h.para['zrange']
        try:
            offset = self.h.para['pluto_offset']
        except KeyError:
            offset = 30

        # Constant pluto radius
        self.Rp = 1186 # km

        # Compute extra params
        self.lengthx = (qx[-1] - qx[0])/self.Rp
        self.plutox = qx[(int(nx/2)+offset)]/self.Rp
        self.heightz = (qzrange[-1] - qzrange[0])/self.Rp
        self.heighty = (qy[-1]- qy[0])/self.Rp
        self.cx = nx/2
        self.cy = ny/2
        self.cz = zrange/2

        # Shift grid so that Pluto lies at (0,0,0) and convert from km to Rp
        self.qx = (qx - qx[len(qx)/2 + offset])/self.Rp
        self.qy = (qy - qy[len(qy)/2])/self.Rp
        self.qzrange = (qzrange - qzrange[len(qzrange)/2])/self.Rp

        #self.X,self.Y = np.meshgrid(self.qx,qzrange)
        X,Y = np.meshgrid(self.qx,self.qy)
        

        # get figure and axes objects
        self.fig, (self.ax1, self.ax2) = plt.subplots(2,1)

        # set the title
        self.ax1.set_title("Density $(m^{-3})$")
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Z')

        data_slice = self.h.get_next_timestep()[-1][:,:,self.cz]
        X,Y = np.meshgrid(self.qx,self.qy)
        self.artist_xy = self.ax1.pcolormesh(X,Y,data_slice.transpose(), cmap=cmaps.viridis, norm=LogNorm())

        data_slice = self.h.get_next_timestep()[-1][:,self.cy,:]
        X,Z = np.meshgrid(self.qx,qzrange)
        self.artist_xz = self.ax2.pcolormesh(X,Z,data_slice.transpose(), cmap=cmaps.viridis, norm=LogNorm())

        self.animation_cache = []
        self.reading_data = True

    def animate(self):
        self.ani = animation.FuncAnimation(self.fig, self.update_animation, interval=0)
        plt.show()

    def update_animation(self, i):
        if self.reading_data:
            try:
                # Skip the last element in each dimension to make set_array work correctly
                #data_slice = self.h.get_next_timestep()[-1][:-1,self.cy,:-1]
                data = self.h.get_next_timestep()[-1]
            except IOError:
                # done reading in the data
                self.reading_data = False
            else:
                data_slice_xy = data[:-1,:-1,self.cz]
                data_slice_xz = data[:-1,self.cy,:-1]
                self.animation_cache.append((data_slice_xy, data_slice_xz))

        if not self.reading_data:
            data_slice_xy, data_slice_xz = self.animation_cache[i%len(self.animation_cache)]

        self.artist_xy.set_array(data_slice_xy.T.ravel())
        self.artist_xz.set_array(data_slice_xz.T.ravel())
        return self.artist_xy, self.artist_xz,
        
        
