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
    para = HybridParams.HybridParams(hybrid_folder).para

    if para['num_proc'] % 2 == 0:
        raise NotImplemented("Tell Nathan he needs to write the code for concatenating an even number of processors")
    else:
        center = int(math.ceil(para['num_proc']/2.0))

        pp = pluto_position(para)

        x_list = []
        v_list = []
        mrat_list = []
        beta_list = []
        tags_list = []
        for offset in (range(-n,n+1) if isinstance(n,int) else n):
            cur_rank = center+offset
            from_bottom = para['num_proc'] - cur_rank

            _, x, v, mrat, beta, tags = read_particle_files(hybrid_folder, cur_rank)

            # Convert processor local coordinate to pluto coordinates
            x[:,0] -= pp
            x[:,1] -= np.max(para['qy'])/2
            x[:,2] += np.max(para['qz'])*from_bottom - from_bottom*para['delz'] - np.max(para['qzrange'])/2

            x_list.append(x)
            v_list.append(v)
            mrat_list.append(mrat)
            beta_list.append(beta)
            tags_list.append(tags)

        ret_x = np.concatenate(x_list)
        ret_v = np.concatenate(v_list)
        ret_mrat = np.concatenate(mrat_list)
        ret_beta = np.concatenate(beta_list)
        ret_tags = np.concatenate(tags_list)

    return para, ret_x, ret_v, ret_mrat, ret_beta, ret_tags

def read_particle_files(hybrid_folder, procnum, dtype=np.float64):
    """Read datafiles"""
    para = HybridParams.HybridParams(hybrid_folder).para

    xp_file = ff.FortranFile(join(hybrid_folder,'particle','c.xp_{}.dat'.format(procnum)))
    vp_file = ff.FortranFile(join(hybrid_folder,'particle','c.vp_{}.dat'.format(procnum)))
    mrat_file = ff.FortranFile(join(hybrid_folder,'particle','c.mrat_{}.dat'.format(procnum)))
    beta_p_file = ff.FortranFile(join(hybrid_folder,'particle','c.beta_p_{}.dat'.format(procnum)))
    tags_file = ff.FortranFile(join(hybrid_folder,'particle','c.tags_{}.dat'.format(procnum)))

    xp_file.seek(0,os.SEEK_END)
    vp_file.seek(0,os.SEEK_END)
    mrat_file.seek(0,os.SEEK_END)
    beta_p_file.seek(0,os.SEEK_END)
    tags_file.seek(0,os.SEEK_END)

    xp = xp_file.readBackReals().reshape((-1, 3), order='F')
    vp = vp_file.readBackReals().reshape((-1, 3), order='F')
    mrat = mrat_file.readBackReals()
    beta_p = beta_p_file.readBackReals()
    tags = tags_file.readBackReals()

    xp_file.close()
    vp_file.close()
    mrat_file.close()
    beta_p_file.close()
    tags_file.close()

    return (para, xp.astype(dtype), vp.astype(dtype), mrat.astype(dtype),
                    para['beta']*beta_p.astype(dtype), tags)

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

        self.cache = {}

    def __getitem__(self, step):
        if step not in self.cache:
            for index,f in zip(self.indexes, self.files):
                f.seek( index[step] )
            self.cache[step] = self.combine( [ f.readReals() for f in self.files ] )

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
            from_bottom = self.para['num_proc'] - (cur_proc)

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

