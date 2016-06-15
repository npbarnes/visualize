#!/usr/bin/python
import FortranFile as ff
import numpy as np
import matplotlib.pyplot as plt
import HybridReader2 as hr
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from matplotlib.colors import Normalize
import matplotlib.animation as animation

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

        self.Rp = 1.216
        self.lengthx = self.h.para['nx']*self.Rp
        self.plutox = (int(self.h.para['nx']/2)+70)*self.Rp
        self.heightz = self.h.para['zrange']*self.Rp
        self.heighty = self.h.para['ny']*self.Rp

        self.fig = plt.figure()
        self.ax = plt.axes()

        self.rax = plt.axes([0.01,0.7,0.07,0.15])
        self.radio = RadioButtons(self.rax, ('xy','xz','yz'))
        self.radio.on_clicked(self._radioUpdate)

        self.axdepth = plt.axes([0.1,0.06,0.65,0.02])
        self.sdepth = Slider(self.axdepth, 'Depth', 0, 1, valinit=0.5)
        self.sdepth.on_changed(self._depthUpdate)

        self.axtime = plt.axes([0.1,0.03,0.65,0.02])
        self.stime = Slider(self.axtime, 'Time', 0, self.h.para['saved_steps'], valinit=t0)
        self.stime.on_changed(self._timeUpdate)

        self.axscale = plt.axes([0.1,0.0,0.65,0.02])
        self.sscale = Slider(self.axscale, 'Scale', 0, 50, valinit=20)
        self.sscale.on_changed(self._scaleUpdate)

        self.axplay = plt.axes([0.84,0.04,0.12,0.04])
        self.bplay = Button(self.axplay, 'Play/Pause')
        self.bplay.on_clicked(self._playUpdate)

        self._has_data = False

    def get_data(self):
        self.data = self.h.get_all_timesteps()['data']
        self._has_data = True

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
            return self.data[t][:,:,d*self.h.para['zrange']].transpose()
        elif(s == 'xz'):
            return self.data[t][:,d*self.h.para['ny'],:].transpose()
        elif(s == 'yz'):
            return self.data[t][d*self.h.para['nx'],:,:].transpose()
        
    # UI update functions
    def _radioUpdate(self,s):
        global s0 
        s0 = s
        self.im.set_data(self._getSlice(t0,s0,d0))

        if(s0 == 'xy'):
            self.im.set_extent([0,nx,0,ny])
        elif(s0 == 'xz'):
            self.im.set_extent([0,nx,0,zrange])
        elif(s0 == 'yz'):
            self.im.set_extent([0,ny,0,zrange])
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

    def _scaleUpdate(self,val):
        val=max(val,1)
        plt.draw()

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
