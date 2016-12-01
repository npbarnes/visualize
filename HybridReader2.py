#!/usr/bin/python
import FortranFile as ff
import numpy as np
from os import SEEK_SET,SEEK_CUR,SEEK_END
from os import listdir
from os.path import isfile, join
import re
from operator import itemgetter
from functools import partial
from HybridParams import HybridParams

class HybridReader2:
    def __init__(self, prefix, variable):
        self.prefix = prefix
        self.grid = join(prefix,'grid')
        self.particle = join(prefix,'particle')
        self.var = variable
        self.para = self._getParameters()

    def _getParameters(self):
        """Reads parameters from para.dat and coords.dat and computes
        some additional parameters that HybridParams cannot."""
        # Read para.dat and coords.dat
        para = HybridParams(self.prefix).para

        # Compute additional useful parameters
        paths = map(partial(join,self.grid),self.sort_filenames())
        zrange = (para['nz']-2)*len(paths)
        saved_steps = para['nt']/para['nout']
        para.update({'zrange':zrange, 'saved_steps':saved_steps})

        nz = para['nz']
        qz = para['qz']
        qzrange = np.empty(zrange+2)
        qzrange[0] = qz[0]
        qzrange[1] = qz[1]
        for i in range(len(paths)):
            qzrange[i*nz-2*i+2:(i+1)*nz-2*(i+1)+2] = qzrange[i*nz-2*i]+qz[2:]

        # Cut the last two for periodic boundaries
        para.update({'qzrange':qzrange[:-2]})

        return para

    ######################################################################################
    # Utilities
    def filenames(self):
        return [f for f in listdir(self.grid) if isfile(join(self.grid,f)) and self.var in f
                                                                           and f.startswith('c.')]

    def _get_number(self, filename):
        dim = re.findall(r'_3d_', filename)
        if len(dim) != 0 and len(dim) != 1:
            raise ValueError("Found %d instances of \'_3d_\'. expected 0 or 1" % len(dim))
        nums = re.findall(r'\d+', filename)
        if len(nums) != 1+len(dim):
            raise ValueError("Found %d numbers in %s. Expected %d" % (len(nums),filename,1+len(dim)))
        return int(nums[len(dim)])

    def sort_filenames(self):
        names = self.filenames()
        nums = map(self._get_number, names)

        aug_list = zip(nums,names)
        aug_list.sort(key=itemgetter(0),reverse=True)

        return list(zip(*aug_list)[1])

    def _cut_overlap(self,a):
        return a[:-2*self.para['nx']*self.para['ny']]
    ###################################################################################

    def get_saved_timesteps(self):
        """returns a list of the time step numbers the simulation saved"""
        filename = self.filenames()[0]
        f = ff.FortranFile(join(self.grid,filename))
        ms = []
        while(True):
            try:
                ms.append(f.readReals()[0])
            except IOError:
                break
            f.skipRecord()
            
        f.close()
        return ms

    def get_last_timestep(self):
        paths = map(partial(join,self.grid),self.sort_filenames())
        handles = map(ff.FortranFile,paths)
        map(lambda x:x.seek(0,SEEK_END), handles)
        # (xyz)
        flat_data = np.concatenate(map(lambda x:self._cut_overlap(x.readBackReals()), handles))
        # (x,y,z)
        data = np.reshape(flat_data,[self.para['nx'],self.para['ny'],(self.para['nz']-2)*len(paths)],'F')

        m = handles[0].readBackInts()[0]
        map(lambda x: x.close(), handles)
        return m, self.para['dt']*m, data

    def get_all_timesteps(self):
        paths = map(partial(join,self.grid),self.sort_filenames())
        handles = map(ff.FortranFile,paths)

        steps = self.get_saved_timesteps()
        nx = self.para['nx']
        ny = self.para['ny']
        nz = self.para['nz']
        zrange = (nz-2)*len(paths)
        dtype = np.dtype([('step',np.int32),('time',np.float32),('data',np.float32,(nx,ny,zrange))])
        ret = np.empty(len(steps),dtype=dtype)

        for n in range(len(steps)):
            # First skip the time step record
            map(lambda x: x.skipRecord(),handles)
            # (xyz)
            flat_data = np.concatenate(map(lambda x:self._cut_overlap(x.readReals()), handles))
            # (x,y,z)
            data = np.reshape(flat_data,[self.para['nx'],self.para['ny'],(self.para['nz']-2)*len(paths)],'F')

            ret[n] = (steps[n],self.para['dt']*steps[n],data)

        map(lambda x: x.close(), handles)
        return ret

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import colormaps as cmaps
    from sys import argv

    plt.register_cmap(name='viridis', cmap=cmaps.viridis)
    plt.register_cmap(name='inferno', cmap=cmaps.inferno)
    plt.register_cmap(name='plasma', cmap=cmaps.plasma)

    h = HybridReader2(argv[1],argv[2])
    im = plt.imshow(h.get_last_timestep()[-1][:,h.para['ny']/2,:].transpose(),origin='lower')
    im.set_cmap(cmaps.viridis)
    im.set_norm(Normalize(vmin=0,vmax=10*pow(10,13)))
    plt.colorbar()
    plt.show()
