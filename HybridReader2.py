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


class HybridError(ValueError):
    pass

class NoSuchVariable(HybridError):
    pass

class HybridReader2:
    def __init__(self, prefix, variable, mode='r', double=False, force_version=None):
        self.doublereals = double
        self.mode = mode

        if double:
            self.real_prec = 'd'
        else:
            self.real_prec = 'f'

        self.var = variable
        self.hp = HybridParams(prefix, force_version=force_version)
        self.para = self.hp.para

        self.filename_format_string = 'c\.{}_3d_(\d+)\.dat'.format(self.var)
        self.rx = re.compile(self.filename_format_string)

        self.paths = map(partial(join, self.hp.grid), self.sort_filenames())
        self.handles = map(partial(ff.FortranFile, mode=mode),self.paths)
        if len(self.paths) == 0:
            raise NoSuchVariable()
        self.isScalar = self._check_scalar()

    def restart(self):
        map(lambda x: x.seek(0), self.handles)

    ######################################################################################
    # Utilities
    def _check_scalar(self):
        if self.doublereals:
            realsize = 8
        else:
            realsize = 4
        h = self.handles[0]
        s = h.tell()
        # skip timestep number
        h.skipRecord()
        size = h._read_leading_indicator()
        h.seek(s)
        number = size/realsize
        if number == self.para['nx']*self.para['ny']*self.para['nz']:
            return True
        elif number == self.para['nx']*self.para['ny']*self.para['nz']*3:
            return False
        else:
            raise HybridError(str(number) + ' != ' + str(self.para['nx']*self.para['ny']*self.para['nz']))


    def filenames(self):
        return [f for f in listdir(self.hp.grid) if self.rx.match(f)]

    def _get_number(self, name):
        match = self.rx.search(name)
        if match is None:
            raise RuntimeError("Filename '{}' does not match the format '{}'.".format(name, self.filename_format_string))
        return int(match.group(1))

    def sort_filenames(self):
        names = self.filenames()
        names.sort(key=self._get_number,reverse=True)

        return names

    def _scalar_cut_overlap(self,a):
        return a[:-2*self.para['nx']*self.para['ny']]

    def _vector_cut_overlap(self,a):
        return a[:-2*self.para['nx']*self.para['ny']*3]
    ###################################################################################

    def get_saved_timesteps(self):
        """returns a list of the time step numbers the simulation saved"""
        f = self.handles[0]
        start = f.tell()
        f.seek(0)
        ms = []
        while(True):
            try:# try to read the step number
                ms.append(f.readInts()[0])
            except ff.NoMoreRecords:# Error indicates EOF
                break
            f.skipRecord()# Skip the data record for this step
            
        f.seek(start, SEEK_SET)
        return np.array(ms)

    def get_next_timestep(self):
        """Returns the next timestep number and data leaving the file position after that data"""
        # read the time step record
        mrecords = np.array(map(lambda x: x.readInts(),self.handles))
        assert mrecords.shape == (len(self.handles),1)
        #assert np.all(mrecords == mrecords[0])
        m = mrecords[0]

        if self.isScalar:
            # (xyz)
            flat_data = np.concatenate(map(lambda x:self._scalar_cut_overlap(x.readReals(self.real_prec)), self.handles))
            # (x,y,z)
            data = np.reshape(flat_data,[self.para['nx'],self.para['ny'],self.para['zrange']],'F')
        else:
            # convert flattened data into 3d array of vectors
            datalst = map(lambda x:x.readReals(self.real_prec), self.handles)
            # shapes data from (p,xyzc) to (p,x,y,z,c)
            redatalst = np.reshape(datalst,[len(self.handles),self.para['nx'],self.para['ny'],self.para['nz'],3], 'F')
            cutOverlap = redatalst[:,:,:,:-2,:]
            # (p,x,y,z,c) -> (x,y,p,z,c)
            rolledlst = np.rollaxis(cutOverlap,0,3)
            # (p,t,x,y,zrange,c)
            data = np.reshape(rolledlst,[self.para['nx'],self.para['ny'],self.para['zrange'],3])

        return m, data

    def get_bizaro_next_timestep(self):
        """Returns the next timestep number and data leaving the file position after that data"""
        # read the time step record
        mrecords = np.array(map(lambda x: x.readInts(),self.handles))
        assert mrecords.shape == (len(self.handles),1)
        assert np.all(mrecords == mrecords[0])
        m = mrecords[0]

        if self.isScalar:
            # (xyz)
            flat_data = np.concatenate(map(lambda x:x.readReals(self.real_prec), self.handles))
            # (x,y,z)
            data = np.reshape(flat_data,[self.para['nx'],self.para['ny'],self.para['zrange']+2*self.para['num_proc']],'F')
        else:
            # convert flattened data into 3d array of vectors
            datalst = map(lambda x:x.readReals(self.real_prec), self.handles)
            # shapes data from (p,xyzc) to (p,x,y,z,c)
            redatalst = np.reshape(datalst,[len(self.handles),self.para['nx'],self.para['ny'],self.para['nz'],3], 'F')
            cutOverlap = redatalst[:,:,:,:-2,:]
            # (p,x,y,z,c) -> (x,y,p,z,c)
            rolledlst = np.rollaxis(cutOverlap,0,3)
            # (p,t,x,y,zrange,c)
            data = np.reshape(rolledlst,[self.para['nx'],self.para['ny'],self.para['zrange'],3])

        return m, data

    def get_timestep(self, n):
        if n < 0:
            map(lambda x:x.seek(0,SEEK_END), self.handles)
            for n in range(-n):
                self.skip_back_timestep()
        else:
            for n in range(n-1):
                self.skip_next_timestep()

        return self.get_next_timestep()

    def skip_next_timestep(self):
        map(lambda x: x.skipRecord(),self.handles)
        map(lambda x: x.skipRecord(),self.handles)

    def skip_back_timestep(self):
        map(lambda x: x.skipBackRecord(),self.handles)
        map(lambda x: x.skipBackRecord(),self.handles)

    def get_prev_timestep(self):
        """Returns the previous timestep number and data leaving the file position before that timestep record"""
        # Skip back over the data and the timestep
        map(lambda x: x.skipBackRecord(),self.handles)
        map(lambda x: x.skipBackRecord(),self.handles)
        # Read the data and timestep
        m, data = self.get_next_timestep()
        # Skip back
        map(lambda x: x.skipBackRecord(),self.handles)
        map(lambda x: x.skipBackRecord(),self.handles)

        return m, data

    def get_last_timestep(self):
        """Returns time step number, time, and data for the last saved step of the simulation"""
        map(lambda x:x.seek(0,SEEK_END), self.handles)
        m, data = self.get_prev_timestep()
        map(lambda x:x.seek(0,SEEK_END), self.handles)
        return m, self.para['dt']*m, data

    def get_all_timesteps(self):
        """Return data from all timesteps and step numbers and physical times"""
        steps = self.get_saved_timesteps()
        nx = self.para['nx']
        ny = self.para['ny']
        zrange = self.para['zrange']
        if self.isScalar:
            ret = np.empty((len(steps), nx, ny, zrange), dtype=np.float32)
        else:
            ret = np.empty((len(steps), nx, ny, zrange, 3), dtype=np.float32)

        for n in range(len(steps)):
            m, data = self.get_next_timestep()

            if self.isScalar:
                ret[n,:,:,:] = data
            else:
                ret[n,:,:,:,:] = data

        map(lambda x: x.close(), self.handles)
        return steps, np.array([self.para['dt']*m for m in steps]), ret
    
    def repair_and_reset(self):
        # Start at the begining
        map(lambda x:x.seek(0, SEEK_SET), self.handles)
        # Repair all handles
        map(lambda x:x.repair(), self.handles)
        # Go back to the begining
        map(lambda x:x.seek(0, SEEK_SET), self.handles)
        # If a file is left with a hanging step number without cooresponding data then remove it
        # index will have an even length if there is a hanging step number
        which_odd = map(lambda x:len(x.index())%2, self.handles)
        for i,odd in enumerate(which_odd):
            if not odd:
                self.handles[i].skipBackRecord()
                self.handles[i].truncate()
        
        # Go back the the begining
        map(lambda x:x.seek(0, SEEK_SET), self.handles)

    def __del__(self):
        map(lambda x: x.close(),self.handles)


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
