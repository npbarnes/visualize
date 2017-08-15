import FortranFile as ff
from os.path import join
import numpy as np
from functools import partial

class ParameterReadError(RuntimeError):
    pass

class FormatError(ParameterReadError):
    pass

class VersionError(ParameterReadError):
    pass

class HybridParams:
    def __init__(self,prefix, int_prec='i', real_prec='f'):
        self.int_prec = int_prec
        self.real_prec = real_prec

        if int_prec == 'i':
            self.inttype = np.int32
        else:
            raise NotImplemented()

        if real_prec == 'f':
            self.realtype = np.float32
        elif real_prec == 'd':
            self.realtype = np.float64
        else:
            raise NotImplemented()

        self.prefix = prefix
        self.grid = join(prefix,'grid')
        self.particle = join(prefix,'particle')
        self.para = self._getParameters()

    def _readV1(self, f):
        record = f.readOther([  ('nx',self.inttype),
                            ('ny',self.inttype),
                            ('nz',self.inttype),
                            ('dx',self.realtype),
                            ('dy',self.realtype),
                            ('delz',self.realtype)])
        assert len(record)==1
        para = dict(zip(record.dtype.names,record[0]))

        record = f.readOther([  ('nt',self.inttype),
                            ('dtsub_init',self.realtype),
                            ('ntsub',self.inttype),
                            ('dt',self.realtype),
                            ('nout',self.inttype)])
        assert len(record)==1
        para.update(zip(record.dtype.names,record[0]))

        out_dir = f.readString()
        para.update({'out_dir':out_dir})

        record = f.readReals(self.real_prec)
        assert len(record)==2
        para.update([('vtop',record[0]),('vbottom',record[1])])

        record = f.readInts(self.int_prec)
        assert len(record)==1
        para.update({'Ni_max':record[0]})

        record = f.readOther([  ('mproton',self.realtype),
                            ('m_pu',self.realtype),
                            ('m_heavy',self.realtype)])
        assert len(record)==1
        para.update(zip(record.dtype.names,record[0]))

        record = f.readReals(self.real_prec)
        assert len(record)==2
        para.update([('np_top',record[0]),('np_bottom',record[1])])

        record = f.readReals(self.real_prec)
        assert len(record)==2
        para.update([('b0_top',record[0]),('b0_bottom',record[1])])

        record = f.readReals(self.real_prec)
        assert len(record)==2
        para.update([('vth_top',record[0]),('vth_bottom',record[1])])

        record = f.readOther([  ('alpha',self.realtype),
                            ('beta',self.realtype)])
        assert len(record)==1
        para.update(zip(record.dtype.names,record[0]))

        record = f.readReals(self.real_prec)
        assert len(record)==1
        para.update({'RIo':record[0]})

        return para

    def _readV2(self, f):
        para = self._readV1(f)

        try:
            record = f.readReals(self.real_prec)
        except IOError:
            raise
        else:
            assert len(record)==1
            para.update({'bo_init':record[0]})

            record = f.readInts(self.int_prec)
            assert len(record)==1
            para.update({'ion_amu':record[0]})

            record = f.readInts(self.int_prec)
            assert len(record)==1
            para.update({'mpu':record[0]})

            record = f.readReals(self.real_prec)
            assert len(record)==1
            para.update({'nf_init':record[0]})

            record = f.readReals(self.real_prec)
            assert len(record)==1
            para.update({'dt_frac':record[0]})

            record = f.readReals(self.real_prec)
            assert len(record)==1
            para.update({'vsw':record[0]})

            record = f.readReals(self.real_prec)
            assert len(record)==1
            para.update({'vth':record[0]})

            record = f.readReals(self.real_prec)
            assert len(record)==1
            para.update({'Ni_tot_frac':record[0]})

            record = f.readReals(self.real_prec)
            assert len(record)==1
            para.update({'dx_frac':record[0]})

            record = f.readReals(self.real_prec)
            assert len(record)==1
            para.update({'nu_init_frac':record[0]})

            record = f.readInts(self.int_prec)
            assert len(record)==1
            para.update({'mrestart':record[0]})

            record = f.readReals(self.real_prec)
            assert len(record)==1
            para.update({'pluto_offset':record[0], 'ri0':record[0]})

            return para

    def _readV3(self, f):
        record = f.readInts(self.int_prec)
        if len(record) != 1:
            raise FormatError
        #assert record[0] == 3, 'record was '+ str(record[0])

        para = self._readV2(f)

        record = f.readInts(self.int_prec)
        assert len(record)==1
        para.update({'part_nout':record[0]})

        return para

    def _readV4(self, f):
        record = f.readInts(self.int_prec)
        if len(record) != 1:
            raise FormatError

        para = self._readV1(f)

        try:
            record = f.readReals(self.real_prec)
        except IOError:
            raise
        else:
            assert len(record)==1
            para.update({'bo_init':record[0]})

            record = f.readInts(self.int_prec)
            assert len(record)==1
            para.update({'ion_amu':record[0]})

            record = f.readReals(self.real_prec)
            assert len(record)==1
            para.update({'mpu':record[0]})

            record = f.readReals(self.real_prec)
            assert len(record)==1
            para.update({'nf_init':record[0]})

            record = f.readReals(self.real_prec)
            assert len(record)==1
            para.update({'dt_frac':record[0]})

            record = f.readReals(self.real_prec)
            assert len(record)==1
            para.update({'vsw':record[0]})

            record = f.readReals(self.real_prec)
            assert len(record)==1
            para.update({'vth':record[0]})

            record = f.readReals(self.real_prec)
            assert len(record)==1
            para.update({'Ni_tot_frac':record[0]})

            record = f.readReals(self.real_prec)
            assert len(record)==1
            para.update({'dx_frac':record[0]})

            record = f.readReals(self.real_prec)
            assert len(record)==1
            para.update({'nu_init_frac':record[0]})

            record = f.readInts(self.int_prec)
            assert len(record)==1
            para.update({'mrestart':record[0]})

            record = f.readReals(self.real_prec)
            assert len(record)==1
            para.update({'pluto_offset':record[0], 'ri0':record[0]})

            return para

    def _readPara(self):
        f = ff.FortranFile(join(self.prefix,'para.dat'))
        try:
            para = self._readV4(f)
        except ValueError:
            f.seek(0)
        else:
            para.update({'version':4})
            return para

        try:
            print "trying V3"
            para = self._readV3(f)
        except FormatError:
            f.seek(0)
        else:
            para.update({'version':3})
            return para

        try:
            para = self._readV2(f)
        except IOError:
            f.seek(0)
        else:
            para.update({'version':2})
            return para

        para = self._readV1(f)
        para.update({'version':1})
        return para

    def _readCoord(self):
        # Read from c.coord.dat
        f = ff.FortranFile(join(self.prefix,'c.coord.dat'))

        record = f.readInts(self.int_prec)
        assert len(record)==1
        para = {'nx':record[0]}

        record = f.readInts(self.int_prec)
        assert len(record)==1
        para['ny'] = record[0]

        record = f.readInts(self.int_prec)
        assert len(record)==1
        para['nz'] = record[0]

        record = f.readReals(self.real_prec)
        assert len(record)==para['nx']
        para.update({'qx':record})

        record = f.readReals(self.real_prec)
        assert len(record)==para['ny']
        para.update({'qy':record})

        record = f.readReals(self.real_prec)
        assert len(record)==para['nz']
        para.update({'qz':record})

        record = f.readReals(self.real_prec)
        assert len(record)==para['nz']
        para.update({'dz_grid':record})

        record = f.readReals(self.real_prec)
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

        # Compute additional useful parameters
        #paths = map(partial(join,self.grid),self.sort_filenames())
        #zrange = (para['nz']-2)*len(paths)
        #saved_steps = para['nt']/para['nout']
        #para.update({'zrange':zrange, 'saved_steps':saved_steps})

        #nz = para['nz']
        #qz = para['qz']
        #qzrange = np.empty(zrange+2)
        #qzrange[0] = qz[0]
        #qzrange[1] = qz[1]
        #for i in range(len(paths)):
        #    qzrange[i*nz-2*i+2:(i+1)*nz-2*(i+1)+2] = qzrange[i*nz-2*i]+qz[2:]

        # Cut the last two for periodic boundaries
        #para.update({'qzrange':qzrange[:-2]})

        return allParams
