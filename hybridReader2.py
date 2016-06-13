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
        self.var = variable
        self.para = self._getParameters()

    def _getParameters(self):
        f = ff.FortranFile(join(self.prefix,'para.dat'))
        r1 = f.readOther([  ('nx',np.int32),
                            ('ny',np.int32),
                            ('nz',np.int32),
                            ('dx',np.float32),
                            ('dy',np.float32),
                            ('delz',np.float32)])
        para = dict(zip(r1.dtype.names,r1[0]))

        r2 = f.readOther([  ('nt',np.int32),
                            ('dtsub_init',np.float32),
                            ('ntsub',np.int32),
                            ('dt',np.float32),
                            ('nout',np.int32)])
        para.update(zip(r2.dtype.names,r2[0]))

        out_dir = f.readString()
        para.update(('out_dir',out_dir))

        r3 = f.readReals()
        para.update(('vtop',r3[0]),('vbottom',r3[1]))

        Ni_max = para.readInts()[0]
        para.update(('Ni_max',Ni_max))

        r4 = f.readOther([   ('mproton',np.float64),
                            ('mpu',np.float64),
                            ('mheavy',np.float32)])
        para.update(zip(r4.dtype.names,r4[0]))

        r5 = f.readReals()
        para.update(('np_top',r5[0]),('np_bottom',r5[1]))

        r6 = f.readReals()
        para.update(('b0_top',r6[0]),('b0_bottom',r6[1]))

        r7 = f.readReals()
        para.update(('vth_top',r7[0]),('vth_bottom',r7[1]))

        r8 = f.readOther([  ('alpha',np.float64),
                            ('beta',np.float32)])
        para.update(zip(r8.dtype.names,r8[0]))

        RIo = f.readReals()[0]
        para.update(('RIo',RIo))

        return para

    def filenames(self):
        return [f for f in listdir(self.prefix) if isfile(join(self.prefix,f)) and self.var in f]

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

    def get_last_timestep(self):
        paths = map(partial(join,self.prefix),self.sort_filenames())
        handles = map(ff.FortranFile,paths)
        map(lambda x:x.seek(0,SEEK_END), handles)
        # (xyz)
        flat_data = np.concatenate(map(lambda x:self._cut_overlap(x.readBackReals()), handles))
        # (x,y,z)
        data = np.reshape(flat_data,[self.para['nx'],self.para['ny'],(self.para['nz']-2)*len(paths)],'F')
        return data

    def get_all_timesteps(self):
        paths = map(partial(join,self.prefix),self.sort_filenames())
        handles = map(ff.FortranFile,paths)

        #TODO:



if __name__ == "__main__":
#    import matplotlib.pyplot as plt
#    from matplotlib.colors import Normalize
#    import colormaps as cmaps
#
#    plt.register_cmap(name='viridis', cmap=cmaps.viridis)
#    plt.register_cmap(name='inferno', cmap=cmaps.inferno)
#    plt.register_cmap(name='plasma', cmap=cmaps.plasma)
#
    h = HybridReader2('databig8','np_3d')
    h.get_last_timestep()
#    boop = plt.imshow(h.get_last_timestep()[:,h.para['ny']/2,:].transpose(),origin='lower')
#    boop.set_cmap(cmaps.viridis)
#    boop.set_norm(Normalize(vmin=0,vmax=10*pow(10,13)))
#    plt.colorbar()
