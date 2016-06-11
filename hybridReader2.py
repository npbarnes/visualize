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
        self.para = self.getParameters()

    def getParameters(self):
        para = ff.FortranFile(join(self.prefix,'para.dat'))
        one = para.readStuff(['i','i','i','f','f','f'])
        two = para.readStuff(['i','f','i','f','i'])
        out_dir = para.readString()
        three = para.readReals()
        Ni_max = para.readInts()
        masses = para.readStuff(['d','d','f'])
        densities = para.readReals()
        fields = para.readReals()
        thermal = para.readReals()
        alpha, beta = para.readStuff(['d','f'])
        RIo = para.readReals()
        return {'nx':one[0],'ny':one[1],'nz':one[2],
                'dx':one[3],'dy':one[4],'delz':one[5],
                'nt':two[0],'dtsub_init':two[1],'ntsub':two[2],'dt':two[3],'nout':two[4],
                'out_dir':out_dir,'vtop':three[0],'vbottom':three[1],
                'Ni_max':Ni_max,'mproton':masses[0],'mpu':masses[1],'mheavy':masses[2],
                'np_top':densities[0],'np_bottom':densities[1],
                'b0_top':fields[0],'b0_bottom':fields[1],
                'vth_top':thermal[0],'vth_bottom':thermal[1],
                'alpha':alpha,'beta':beta,'RIo':RIo}
        

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
