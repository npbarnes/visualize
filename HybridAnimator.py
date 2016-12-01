#!/usr/bin/python
import FortranFile as ff
import numpy as np
import matplotlib.pyplot as plt
import HybridReader2 as hr
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from matplotlib.colors import Normalize
import matplotlib.animation as animation
from scipy.interpolate import RegularGridInterpolator

import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.register_cmap(name='inferno', cmap=cmaps.inferno)
plt.register_cmap(name='plasma', cmap=cmaps.plasma)


s0 = 'xz'
t0 = 0
d0 = 0.5
playing = True

class HybridAnimator():
    def __init__(self,prefix,variable):
        self.prefix = prefix
        self.variable = variable
        self.h = hr.HybridReader2(self.prefix,self.variable)

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

        self.Rp = 1186 # km
        self.lengthx = (qx[-1] - qx[0])/self.Rp
        self.plutox = qx[(int(nx/2)+offset)]/self.Rp
        self.heightz = (qzrange[-1] - qzrange[0])/self.Rp
        self.heighty = (qy[-1]- qy[0])/self.Rp

        self.fig = plt.figure()
        self.ax = plt.axes()

        self.rax = plt.axes([0.01,0.7,0.07,0.15])
        self.radio = RadioButtons(self.rax, ('xy','xz','yz'))
        self.radio.on_clicked(self._radioUpdate)

        self.axdepth = plt.axes([0.1,0.04,0.65,0.02])
        self.sdepth = Slider(self.axdepth, 'Depth', 0, 1, valinit=0.5)
        self.sdepth.on_changed(self._depthUpdate)

        self.axtime = plt.axes([0.1,0.01,0.65,0.02])
        self.stime = Slider(self.axtime, 'Time', 0, self.h.para['saved_steps'], valinit=t0)
        self.stime.on_changed(self._timeUpdate)

        self.axplay = plt.axes([0.84,0.04,0.12,0.04])
        self.bplay = Button(self.axplay, 'Play/Pause')
        self.bplay.on_clicked(self._playUpdate)

        self._has_data = False

    def get_data(self):
        self.data = self.h.get_all_timesteps()['data']
        self._has_data = True

    def _resample_slice(self,data2d,qa,qb,na,nb):
        rgi = RegularGridInterpolator(points=[qa,qb], values=data2d)

        # mgrid gives us new_grid[:,i,j] == [x,y] coordinates of point i,j
        new_grid = np.mgrid[qa[0]:qa[-1]:na*1j, qb[0]:qb[-1]:nb*1j]
        # rollaxis turns it into new_grid[i,j] == [x,y]
        new_grid = np.rollaxis(new_grid,0,len(new_grid.shape))

        return rgi(new_grid)

    def animate(self):
        if not self._has_data:
            self.get_data()

        self.im = self.ax.imshow(self._getSlice(t0,s0,d0),interpolation='none',origin='lower') 
        self.im.set_extent([-self.plutox,self.lengthx-self.plutox,-self.heightz/2,self.heightz/2])
        self.im.set_cmap(cmaps.viridis)

        self.ani = animation.FuncAnimation( self.fig, self._animUpdate, init_func=self._anim_init,
                                            frames=34, interval=3, blit=True)
        plt.show()

    def _getSlice(self,t,s,d):
        if(s == 'xy'):
            qa = self.h.para['qx']
            qb = self.h.para['qy']
            na = self.h.para['nx']
            nb = self.h.para['ny']
            resampled = self._resample_slice(self.data[t][:,:,d*self.h.para['zrange']],qa,qb,na,nb)
            return resampled.transpose()
        elif(s == 'xz'):
            qa = self.h.para['qx']
            qb = self.h.para['qzrange']
            na = self.h.para['nx']
            nb = self.h.para['zrange']
            resampled = self._resample_slice(self.data[t][:,d*self.h.para['ny'],:],qa,qb,na,nb)
            return resampled.transpose()
        elif(s == 'yz'):
            qa = self.h.para['qy']
            qb = self.h.para['qzrange']
            na = self.h.para['ny']
            nb = self.h.para['zrange']
            resampled = self._resample_slice(self.data[t][d*self.h.para['nx'],:,:],qa,qb,na,nb)
            return resampled.transpose()
        
    # UI update functions
    def _radioUpdate(self,s):
        global s0 
        s0 = s
        self.im.set_data(self._getSlice(t0,s0,d0))

        if(s0 == 'xy'):
            self.im.set_extent([-self.plutox,self.lengthx-self.plutox,-self.heighty/2,self.heighty/2])
        elif(s0 == 'xz'):
            self.im.set_extent([-self.plutox,self.lengthx-self.plutox,-self.heightz/2,self.heightz/2])
        elif(s0 == 'yz'):
            self.im.set_extent([-self.heighty/2,self.heighty/2,-self.heightz/2,self.heightz/2])
        plt.draw()

    def _drawSlice(self):
        self.im.set_data(self._getSlice(t0,s0,d0))

    def _timeUpdate(self,val):
        global t0
        global playing
        t0 = min(self.h.para['saved_steps'],int(val))
        t0 = max(0,t0)
        playing = False
        self._drawSlice()

    def _depthUpdate(self,val):
        global d0
        d0 = max(val,0)
        d0 = min(d0,1)
        self._drawSlice()

    def _playUpdate(self,event):
        global playing
        playing = not playing

    def _animUpdate(self,num):
        global t0
        global playing
        if(playing):
            t0 = (t0+1) % self.h.para['saved_steps']
            self._drawSlice()
            self.stime.set_val(t0)
            #Set_val calls _timeUpdate that turns playing off.
            playing = True
        return self.im,

    def _anim_init(self):
        return self.im,

if __name__ == "__main__":
    ha = HybridAnimator('databig6','np_3d')
    ha.animate()
