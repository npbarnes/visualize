#!/usr/bin/python
import FortranFile as ff
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import HybridReader2 as hr
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
import matplotlib.animation as animation
from scipy.interpolate import RegularGridInterpolator
from bisect import bisect

import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.register_cmap(name='inferno', cmap=cmaps.inferno)
plt.register_cmap(name='plasma', cmap=cmaps.plasma)


class StoppableFrames:
    def __init__(self):
        self.done_playing = False

    def __iter__(self):
        return self

    def next(self):
        if self.done_playing:
            raise StopIteration
        else:
            return self

class HybridAnimator():
    def __init__(self,prefix,variable, coordinate=None,save=False):
        # Read hybrid files
        self.prefix = prefix
        self.variable = variable
        self.coordinate = coordinate
        self.save = save


        self.h = hr.HybridReader2(self.prefix,self.variable)

        if self.h.isScalar and self.coordinate is not None:
            raise ValueError("Don't specify a coordinate for scalars.")
        if not self.h.isScalar and self.coordinate is None:
            raise ValueError("Must specify a coordinate for vectors.")

        # Grab some of the parameters directly from there to be used
        # to compute some extra parameters
        qx = self.h.para['qx']
        qy = self.h.para['qy']
        qzrange = self.h.para['qzrange']
        nx = self.h.para['nx']
        ny = self.h.para['ny']
        zrange = self.h.para['zrange']
        try:
            offset = int(self.h.para['pluto_offset'])
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

        # get figure and axes objects
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        self.ax1.set_aspect('equal', adjustable='box-forced')
        self.ax2.set_aspect('equal', adjustable='box-forced')
        self.fig.subplots_adjust(hspace=0, wspace=0)

        # set the title
        self.ax1.set_title(self.variable)
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Z')

        if self.h.isScalar:
            data_slice = self.h.get_next_timestep()[-1][:,:,self.cz]
        else:
            data_slice = self.h.get_next_timestep()[-1][:,:,self.cz,self.coordinate]
        X,Y = np.meshgrid(self.qx,self.qy)
        #self.artist_xy = self.ax1.pcolormesh(X,Y,data_slice.transpose(), cmap=cmaps.viridis, norm=LogNorm(), vmin=1e11, vmax=1e13)
        self.artist_xy = self.ax1.pcolormesh(X,Y,data_slice.transpose(), cmap=cmaps.viridis, norm=SymLogNorm(.001))

        if self.h.isScalar:
            data_slice = self.h.get_next_timestep()[-1][:,self.cy,:]
        else:
            data_slice = self.h.get_next_timestep()[-1][:,self.cy,:,self.coordinate]
        X,Z = np.meshgrid(self.qx,self.qzrange)
        #self.artist_xz = self.ax2.pcolormesh(X,Z,data_slice.transpose(), cmap=cmaps.viridis, norm=LogNorm(), vmin=1e11, vmax=1e13)
        self.artist_xz = self.ax2.pcolormesh(X,Z,data_slice.transpose(), cmap=cmaps.viridis, norm=SymLogNorm(.001))

        self.fig.colorbar(self.artist_xy, ax=self.ax1)
        self.fig.colorbar(self.artist_xz, ax=self.ax2)

        self.animation_cache = []
        self.reading_data = True

    def animate(self):
        if self.save:
            self.ani = animation.FuncAnimation(self.fig, frames=StoppableFrames(),
                            func=self.update_animation, interval=1, blit=True, repeat=False)
            self.ani.save('pluto.mp4', fps=20, bitrate=5000, writer='avconv')
        else:
            self.ani = animation.FuncAnimation(self.fig, func=self.update_animation, interval=1, blit=True)
            plt.show()

    def update_animation(self, frame):
        if self.reading_data:
            try:
                data = self.h.get_next_timestep()[-1]
                if not self.h.isScalar:
                    data = data[:,:,:,self.coordinate]
            except ff.NoMoreRecords:
                # done reading in the data
                if self.save:
                    frame.done_playing = True
                    return self.artist_xy, self.artist_xz
                else:
                    self.reading_data = False
                    print("Done reading data")
            else:
                # Skip the last element in each dimension to make set_array work correctly
                data_slice_xy = data[:-1,:-1,self.cz]
                data_slice_xz = data[:-1,self.cy,:-1]
                self.animation_cache.append((data_slice_xy, data_slice_xz))

        if not self.reading_data:
            data_slice_xy, data_slice_xz = self.animation_cache[frame%len(self.animation_cache)]

        self.artist_xy.set_array(data_slice_xy.T.ravel())
        self.artist_xz.set_array(data_slice_xz.T.ravel())
        return self.artist_xy, self.artist_xz
