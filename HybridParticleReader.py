import os
import math
import numpy as np
import HybridParams
import FortranFile as ff
from os.path import join

def pluto_position(p):
    """get the position of pluto in simulation coordinates"""
    print("HybridParticleReader: setting pluto_offset to 30 since hpara is wrong")
    p['pluto_offset'] = 30
    return p['qx'][p['nx']/2 + p['pluto_offset']]

def particle_data(hybrid_folder, n=0):
    hybrid_folder, data_folder = os.path.split(hybrid_folder)
    hpr = HybridParticleReader(hybrid_folder, n, data_folder=data_folder)

    return hpr.x.para, hpr.x[-1], hpr.v[-1], hpr.mrat[-1], hpr.beta[-1], hpr.tags[-1]

class CombinedParticleData:
    @property
    def varname(self):
        raise NotImplemented

    def combine(self, raw_data_list):
        raise NotImplemented

    def __init__(self, folder, n=0, data_folder='data'):
        self.offsets = range(-n, n+1) if isinstance(n,int) else n

        self.para = HybridParams.HybridParams(join(folder, data_folder)).para
        self.center_proc = int(math.ceil(self.para['num_proc']/2.0))
        self.procs = [ self.center_proc + offset for offset in self.offsets ]
        
        self.files = [ ff.FortranFile(join(folder,data_folder,'particle','c.{}_{}.dat'.format(self.varname, proc)))
                                    for proc in self.procs ] 

        self._build_indexes()

        self.numsteps = len(self.indexes[0])

        self.cache = {}

    def __len__(self):
        return self.numsteps

    def __getitem__(self, step):
        if step < 0:
            step = self.numsteps + step

        if step not in self.cache:
            for index,f in zip(self.indexes, self.files):
                f.seek( index[step] )
            self.cache[step] = self.combine( [ f.readReals().astype(np.float64) for f in self.files ] )

        return self.cache[step]

    def _build_indexes(self):
        self.indexes = []
        for f in self.files:
            # Skip the step number records and only record the seek positions
            # for the actual particle data
            self.indexes.append( f.index()[1::2] )

class xp(CombinedParticleData):
    varname = 'xp'

    def combine(self, raw_data):
        pp = pluto_position(self.para)

        lst = []
        for cur_proc, d in zip(self.procs, raw_data):
            dd = d.reshape((-1,3), order='F')
            #from_bottom = self.para['num_proc'] - cur_proc
            from_bottom = cur_proc - 1

            dd[:,0] -= pp
            dd[:,1] -= np.max(self.para['qy'])/2
            dd[:,2] += -self.para['qz'][1] + self.para['qz'][-2]*from_bottom - self.para['simulation_height']/2

            lst.append(dd)

        return np.concatenate(lst)


class vp(CombinedParticleData):
    varname = 'vp'

    def combine(self, raw_data):
        lst = []
        for d in raw_data:
            dd = d.reshape((-1,3), order='F')

            lst.append(dd)

        return np.concatenate(lst)

class beta(CombinedParticleData):
    varname = 'beta_p'

    def combine(self, raw_data):
        ret = np.concatenate(raw_data)
        np.multiply(ret, self.para['beta'], out=ret)
        return ret

class SimpleParticleData(CombinedParticleData):
    combine = np.concatenate

class mrat(SimpleParticleData):
    varname = 'mrat'

class beta_p(SimpleParticleData):
    varname = 'beta_p'

class tags(SimpleParticleData):
    varname = 'tags'

class HybridParticleReader:
    def __init__(self, folder, n=0, data_folder='data'):
        self.x = xp(folder, n, data_folder)
        self.v = vp(folder, n, data_folder)
        self.beta = beta(folder, n, data_folder)
        self.beta_p = beta_p(folder, n, data_folder)
        self.mrat = mrat(folder, n, data_folder)
        self.tags = tags(folder, n, data_folder)

