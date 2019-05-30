import FortranFile as ff
from os import listdir
from os.path import isfile,join
import numpy as np
from functools import partial
import re

class ParameterReadError(RuntimeError):
    pass

class FormatError(ParameterReadError):
    pass

class VersionError(ParameterReadError):
    pass

class HybridParams:
    def __init__(self,prefix, force_version=None):
        self.prefix = prefix
        self.grid = join(prefix,'grid')
        self.particle = join(prefix,'particle')
        self.force_version = force_version

        self.para = self._getParameters()

    def num_procs(self):
        """Count how many density files were output to get the number of processors"""
        filename_format_string = 'c\.np_3d_(\d+)\.dat'
        rx = re.compile(filename_format_string)
        return len( [f for f in listdir(self.grid) if rx.match(f)] )

    def _readPara(self):
        para = {}
        f = ff.FortranFile(join(self.prefix,'para.dat'))

        # Try to read a version number
        record = f.readInts()
        try:
            assert len(record)==1
        except AssertionError:
            # If there is no version number go back to the begining.
            # Older versions of para.dat don't have a version number.
            f.seek(0)
            # Give it a version of zero for completeness.
            para.update({'para_dat_version':0})
        else:
            # Otherwise, add the indicated version to the dictionary.
            para.update({'para_dat_version':record[0]})
        if self.force_version is not None:
            self.version = self.force_version
        else:
            self.version = para['para_dat_version']

        # This will be the first record in old versions of para.dat
        record = f.readOther([  ('nx',np.int32),
                            ('ny',np.int32),
                            ('nz',np.int32),
                            ('dx',np.float32),
                            ('dy',np.float32),
                            ('delz',np.float32)])
        assert len(record)==1
        para = dict(zip(record.dtype.names,record[0]))

        record = f.readOther([  ('nt',np.int32),
                            ('dtsub_init',np.float32),
                            ('ntsub',np.int32),
                            ('dt',np.float32),
                            ('nout',np.int32)])
        assert len(record)==1
        para.update(zip(record.dtype.names,record[0]))

        out_dir = f.readString()
        para.update({'out_dir':out_dir})

        record = f.readReals()
        assert len(record)==2
        para.update([('vtop',record[0]),('vbottom',record[1])])

        record = f.readInts()
        assert len(record)==1
        para.update({'Ni_max':record[0]})

        if self.version <= 3:
            record = f.readOther([  ('mproton',np.float64),
                                ('m_pu',np.float64),
                                ('m_heavy',np.float32)])
        else:
            record = f.readOther([  ('mproton',np.float32),
                                ('m_pu',np.float32),
                                ('m_heavy',np.float32)])
        assert len(record)==1
        para.update(zip(record.dtype.names,record[0]))

        record = f.readReals()
        assert len(record)==2
        para.update([('np_top',record[0]),('np_bottom',record[1])])

        record = f.readReals()
        assert len(record)==2
        para.update([('b0_top',record[0]),('b0_bottom',record[1])])

        record = f.readReals()
        assert len(record)==2
        para.update([('vth_top',record[0]),('vth_bottom',record[1])])

        record = f.readOther([  ('alpha',np.float64),
                            ('beta',np.float32)])
        assert len(record)==1
        para.update(zip(record.dtype.names,record[0]))

        # For some versions of para.dat this will be the last record
        record = f.readReals()
        assert len(record)==1
        # Currently (circa 2015-17) this should be the radius of pluto, but in the past
        # it was the radius of Io, a moon of Jupiter. We record both names here for completeness
        para.update({'RPluto':record[0], 'RIo':record[0]})

        # If there are no more records return what we do have
        # otherwise, continue.
        try: 
            record = f.readReals()
        except ff.NoMoreRecords:
            return para

        assert len(record)==1
        para.update({'b0_init':record[0]})

        record = f.readInts()
        assert len(record)==1
        para.update({'ion_amu':record[0]})

        if self.version <= 3:
            record = f.readReals('d')
        else:
            record = f.readReals()
        assert len(record)==1
        para.update({'mpu':record[0]})

        record = f.readReals()
        assert len(record)==1
        para.update({'nf_init':record[0]})

        record = f.readReals()
        assert len(record)==1
        para.update({'dt_frac':record[0]})

        record = f.readReals()
        assert len(record)==1
        para.update({'vsw':record[0]})

        record = f.readReals()
        assert len(record)==1
        para.update({'vth':record[0]})

        record = f.readReals()
        assert len(record)==1
        para.update({'Ni_tot_frac':record[0]})

        record = f.readReals()
        assert len(record)==1
        para.update({'dx_frac':record[0]})

        record = f.readReals()
        assert len(record)==1
        para.update({'nu_init_frac':record[0]})

        record = f.readInts()
        assert len(record)==1
        para.update({'mrestart':record[0]})

        record = f.readInts()
        assert len(record)==1
        # Do a quick sanity check since some older runs have this stored as a real insead of an int.
        if record[0] > 100 or record[0] < 0:
            f.skipBackRecord()
            record = f.readInts()
            assert len(record)==1
        # ri0 is the older name for this that shouldn't be used because it looks too similar to the parameter RIo
        # which is also old (radius of the moon Io even though we simulate pluto now).
        para.update({'pluto_offset':int(record[0]), 'ri0':record[0]}) 

        record = f.readInts()
        assert len(record)==1
        para.update({'part_nout':record[0]})

        return para

    def _readCoord(self):
        f = ff.FortranFile(join(self.prefix,'c.coord.dat'))

        record = f.readInts()
        assert len(record)==1
        para = {'nx':record[0]}

        record = f.readInts()
        assert len(record)==1
        para['ny'] = record[0]

        record = f.readInts()
        assert len(record)==1
        para['nz'] = record[0]

        record = f.readReals()
        assert len(record)==para['nx']
        para.update({'qx':record})

        record = f.readReals()
        assert len(record)==para['ny']
        para.update({'qy':record})

        record = f.readReals()
        assert len(record)==para['nz']
        para.update({'qz':record})

        record = f.readReals()
        assert len(record)==para['nz']
        para.update({'dz_grid':record})

        record = f.readReals()
        assert len(record)==para['nz']
        para.update({'dz_cell':record})

        return para

    def _mergeDicts(self,d1,d2):
        k1 = d1.keys()
        k2 = d2.keys()

        shared = [k for k in k1 if k in k2]

        for key in shared:
            assert d1[key] == d2[key]

        ret = d1.copy()
        ret.update(d2)
        return ret

    def _getParameters(self):
        para = self._readPara()
        coord = self._readCoord()

        allParams = self._mergeDicts(para,coord)

        # Add the number of processors
        allParams['num_proc'] = self.num_procs()

        # Compute additional useful parameters
        zrange = (allParams['nz']-2)*allParams['num_proc']
        allParams['zrange'] = zrange
        saved_steps = allParams['nt']/allParams['nout']
        allParams['saved_steps'] = saved_steps

        nz = allParams['nz']
        qz = allParams['qz']
        qzrange = np.empty(zrange+2)
        qzrange[0] = qz[0]
        qzrange[1] = qz[1]
        for i in range(allParams['num_proc']):
            qzrange[i*nz-2*i+2:(i+1)*nz-2*(i+1)+2] = qzrange[i*nz-2*i]+qz[2:]

        allParams['simulation_height'] = qzrange[-2]
        # Cut the last two for periodic boundaries
        allParams['qzrange'] = qzrange[:-2]

        # Get grid points in pluto centered coords
        def pluto_position(p):
            """get the position of pluto in simulation coordinates"""
            try:
                return p['qx'][p['nx']/2 + p['pluto_offset']]
            except IndexError:
                return 0.0

        qx = allParams['qx'] - pluto_position(allParams)
        qy = allParams['qy'] - np.max(allParams['qy'])/2
        qz = allParams['qzrange'] - allParams['qzrange'][-1]/2

        allParams['grid_points'] = qx, qy, qz

        return allParams
