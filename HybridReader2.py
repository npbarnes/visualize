import FortranFile as ff
import numpy as np
from os import SEEK_SET,SEEK_CUR,SEEK_END
from os import listdir
from os.path import isfile, join
import re
from operator import itemgetter
from functools import partial

class HybridReader2:
    def __init__(self, prefix, variable):
        self.prefix = prefix
        self.grid = join(prefix,'grid')
        self.particle = join(prefix,'particle')
        self.var = variable
        self.para = self._getParameters()

    def _getParameters(self):
        # Read from para.dat
        f = ff.FortranFile(join(self.prefix,'para.dat'))
        record = f.readOther([  ('nx',np.int32),
                            ('ny',np.int32),
                            ('nz',np.int32),
                            ('dx',np.float32),
                            ('dy',np.float32),
                            ('delz',np.float32)])
        para = dict(zip(record.dtype.names,record[0]))

        record = f.readOther([  ('nt',np.int32),
                            ('dtsub_init',np.float32),
                            ('ntsub',np.int32),
                            ('dt',np.float32),
                            ('nout',np.int32)])
        para.update(zip(record.dtype.names,record[0]))

        out_dir = f.readString()
        para.update({'out_dir':out_dir})

        record = f.readReals()
        para.update([('vtop',record[0]),('vbottom',record[1])])

        record = f.readInts()
        para.update({'Ni_max':record[0]})

        record = f.readOther([  ('mproton',np.float64),
                            ('mpu',np.float64),
                            ('mheavy',np.float32)])
        para.update(zip(record.dtype.names,record[0]))

        record = f.readReals()
        para.update([('np_top',record[0]),('np_bottom',record[1])])

        record = f.readReals()
        para.update([('b0_top',record[0]),('b0_bottom',record[1])])

        record = f.readReals()
        para.update([('vth_top',record[0]),('vth_bottom',record[1])])

        record = f.readOther([  ('alpha',np.float64),
                            ('beta',np.float32)])
        para.update(zip(record.dtype.names,record[0]))

        record = f.readReals()
        para.update({'RIo':record[0]})

        # Compute additional useful parameters
        paths = map(partial(join,self.grid),self.sort_filenames())
        zrange = (para['nz']-2)*len(paths)
        saved_steps = para['nt']/para['nout']
        para.update({'zrange':zrange, 'saved_steps':saved_steps})


        return para

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

    def get_saved_timesteps(self):
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

    plt.register_cmap(name='viridis', cmap=cmaps.viridis)
    plt.register_cmap(name='inferno', cmap=cmaps.inferno)
    plt.register_cmap(name='plasma', cmap=cmaps.plasma)

    h = HybridReader2('databig6','np_3d')
    boop = plt.imshow(h.get_last_timestep()[-1][:,h.para['ny']/2,:].transpose(),origin='lower')
    boop.set_cmap(cmaps.viridis)
    boop.set_norm(Normalize(vmin=0,vmax=10*pow(10,13)))
    plt.colorbar()
    plt.show()
