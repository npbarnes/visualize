#!/usr/bin/python
import FortranFile as ff

from os.path import join
import numpy as np
from pprint import pprint
import sys

def getParameters(prefix):
    f = ff.FortranFile(join(prefix,'para.dat'))
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
    para.update({'out_dir':out_dir})

    r3 = f.readReals()
    para.update([('vtop',r3[0]),('vbottom',r3[1])])

    Ni_max = f.readInts()[0]
    para.update({'Ni_max':Ni_max})

    r4 = f.readOther([  ('mproton',np.float64),
                        ('mpu',np.float64),
                        ('mheavy',np.float32)])
    para.update(zip(r4.dtype.names,r4[0]))

    r5 = f.readReals()
    para.update([('np_top',r5[0]),('np_bottom',r5[1])])

    r6 = f.readReals()
    para.update([('b0_top',r6[0]),('b0_bottom',r6[1])])

    r7 = f.readReals()
    para.update([('vth_top',r7[0]),('vth_bottom',r7[1])])

    r8 = f.readOther([  ('alpha',np.float64),
                        ('beta',np.float32)])
    para.update(zip(r8.dtype.names,r8[0]))

    RIo = f.readReals()[0]
    para.update({'RIo':RIo})

    return para

pprint(getParameters(sys.argv[1]))
