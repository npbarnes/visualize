#!/usr/bin/python
import FortranFile as ff
import numpy as np
from os import SEEK_SET,SEEK_CUR,SEEK_END
from os import listdir
from os.path import isfile, join, getsize
import re
from operator import itemgetter
from functools import partial
from HybridParams import HybridParams

def _find_nonempty(paths):
    for p in paths:
        if getsize(p) != 0:
            return p
    return None

class FakeFortranFile:
    def __init__(self, nx, ny, nz, dtype, isScalar):
        if isScalar:
            self.arr = np.zeros(nx*ny*nz, dtype=dtype)
        else:
            self.arr = np.zeros(nx*ny*nz*3, dtype=dtype)

    def skipRecord(self):
        pass
    def skipBackRecord(self):
        pass
    def readInts(self):
        return None
    def readReals(self, p=None):
        return self.arr
    def readBackInts(self):
        return self.readInts()
    def readBackReals(self, p=None):
        return self.readReals()

    def seek(self, n, m=None):
        pass

class HybridError(ValueError):
    pass

class NoSuchVariable(HybridError):
    pass

class HybridReader2:
    def __init__(self, prefix, variable, mode='r', double=False, force_version=None, force_procs=None):
        self.doublereals = double
        self.mode = mode

        if double:
            self.real_prec = 'd'
        else:
            self.real_prec = 'f'

        self.grid = join(prefix,'grid')

        self.var = variable

        self.filename_format_string = '^c\.{}_3d_(\d+)\.dat$'.format(self.var)
        self.rx = re.compile(self.filename_format_string)

        self.all_paths = [join(self.grid, f) for f in self.sort_filenames()]
        N = len(self.all_paths)
        if N == 0:
            raise NoSuchVariable(str(variable))

        if force_procs is None:
            self.paths = self.all_paths
        elif isinstance(force_procs, int):
            if N%2 == 1:
                middle = N//2 + 1
                middle_index = middle-1
                self.paths = self.all_paths[middle_index-force_procs:middle_index+force_procs+1]
            else: # N%2 == 0:
                lowermid = N//2
                lowermid_index = lowermid-1
                self.paths = self.all_paths[lowermid_index-(force_procs-1):lowermid_index+force_procs+1]
        else:
            self.paths = self.all_paths[force_procs]
        num_procs = len(self.paths)

        self.hp = HybridParams(prefix, force_version=force_version, force_numprocs=num_procs)
        self.para = self.hp.para

        self.representative_path = _find_nonempty(self.paths)
        if self.representative_path is None:
            raise HybridError(f"All {variable} files appear to be empty")
        self.representative = ff.FortranFile(self.representative_path, mode=mode)
        self.isScalar = self._check_scalar()

        fake = FakeFortranFile(self.para['nx'], self.para['ny'], self.para['nz'], self.real_prec, self.isScalar)
        self.handles = []
        for p in self.paths:
            if getsize(p) == 0:
                self.handles.append(fake)
            else:
                self.handles.append(ff.FortranFile(p, mode=mode))
        self.real_handles = [h for h in self.handles if isinstance(h, ff.FortranFile)]

    def restart(self):
        for h in self.real_handles:
            h.seek(0)

    ######################################################################################
    # Utilities
    def _check_scalar(self):
        if self.doublereals:
            realsize = 8
        else:
            realsize = 4
        h = self.representative
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
        return [f for f in listdir(self.grid) if self.rx.match(f)]

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
    ###################################################################################

    def get_saved_timesteps(self):
        """returns a list of the time step numbers the simulation saved"""
        f = self.representative
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
        mrecords = np.array([h.readInts() for h in self.real_handles])
        assert mrecords.shape == (len(self.real_handles),1)
        mrecords = mrecords.flatten()
        assert np.all(mrecords == mrecords[0])
        m = mrecords[0]

        if self.isScalar:
            # (xyz)
            flat_data = np.concatenate([self._scalar_cut_overlap(h.readReals(self.real_prec)) for h in self.handles])
            # (x,y,z)
            data = np.reshape(flat_data,[self.para['nx'],self.para['ny'],self.para['zrange']],'F')
        else:
            # convert flattened data into 3d array of vectors
            datalst = [h.readReals(self.real_prec) for h in self.handles]
            # shapes data from (p,xyzc) to (p,x,y,z,c)
            redatalst = np.reshape(datalst,[len(self.handles),self.para['nx'],self.para['ny'],self.para['nz'],3], 'F')
            cutOverlap = redatalst[:,:,:,:-2,:]
            # (p,x,y,z,c) -> (x,y,p,z,c)
            rolledlst = np.rollaxis(cutOverlap,0,3)
            # (p,t,x,y,zrange,c)
            data = np.reshape(rolledlst,[self.para['nx'],self.para['ny'],self.para['zrange'],3])

        return m, data

    def get_timestep(self, n):
        """
        If n is positive, then load the n'th timestep. (i.e. one based indexing)
        If n is negative, then count from the back. (i.e. n=-1 means load the last step saved)
        if n is zero, then raise a ValueError.
        """
        if n < 0:
            for h in self.real_handles:
                h.seek(0, SEEK_END)
            for n in range(-n):
                self.skip_back_timestep()
        elif n == 0:
            raise ValueError("step number must be positive or negative, not zero")
        else:
            # should we seek(0) first?
            for n in range(n-1):
                self.skip_next_timestep()

        return self.get_next_timestep()

    def skip_next_timestep(self):
        for h in self.real_handles:
            h.skipRecord()
            h.skipRecord()

    def skip_back_timestep(self):
        for h in self.real_handles:
            h.skipBackRecord()
            h.skipBackRecord()

    def get_prev_timestep(self):
        """Returns the previous timestep number and data leaving the file position before that timestep record"""
        self.skip_back_timestep()
        m, data = self.get_next_timestep()
        self.skip_back_timestep()

        return m, data

    def get_last_timestep(self):
        """Returns time step number, time, and data for the last saved step of the simulation"""
        for h in self.real_handles:
            h.seek(0, SEEK_END)
        self.skip_back_timestep()
        return self.get_next_timestep()

    def get_all_timesteps(self):
        """Return data from all timesteps and step numbers and physical times"""
        for h in self.real_handles:
            h.seek(0, SEEK_SET)
        nx = self.para['nx']
        ny = self.para['ny']

        ms = []
        ret_lst = []
        while True:
            try:
                m, data = self.get_next_timestep()
            except ff.NoMoreRecords:
                break
            ms.append(m)
            ret_lst.append(data)

        ret = np.stack(ret_lst, axis=0)
        return np.array(ms), ret

    def repair_and_reset(self):
        for h in self.real_handles:
            # Start at the begining
            h.seek(0, SEEK_SET)
            # Repair
            h.repair()
            # Go back to the begining
            h.seek(0, SEEK_SET)
            # if there are an even number of entries in the index,
            # then there is a hanging record number without any data
            # so we remove it.
            if len(h.index())%2 == 0:
                h.skipBackRecord() # is this right?
                h.truncate()
            # Finally, make sure we leave it at the begining
            h.seek(0, SEEK_SET)


    def __del__(self):
        try:
            for h in self.real_handles:
                h.close()
        except AttributeError:
            pass

def step_iter(h):
    while True:
        try:
            m, data = h.get_next_timestep()
        except ff.NoMoreRecords:
            return
        yield m, data

def monotonic_step_iter(h):
    # Read the first timestep (number and data)
    # If the file is empty, then just return
    try:
        m0, data0 = h.get_next_timestep()
    except ff.NoMoreRecords:
        return
    yield m0, data0

    # Read data until there are no more records
    # Only yield the timestep when the step number is increasing
    prev_m = m0
    while True:
        try:
            m, data = h.get_next_timestep()
        except ff.NoMoreRecords:
            return
        if m > prev_m:
            prev_m = m
            yield m, data

def equal_spacing_step_iter(h):
    iterator = monotonic_step_iter(h)
    m0, data0 = next(iterator)
    yield m0, data0

    m1, data1 = next(iterator)
    yield m1, data1

    dm = m1 - m0
    prev_m = m1
    for m,data in iterator:
        if m-prev_m == dm:
            yield m, data
        prev_m = m
