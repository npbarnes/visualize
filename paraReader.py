#!/usr/bin/python
import FortranFile as ff

para = ff.FortranFile('tmp/para.dat')

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
print 'nx,ny,nz,dx,dy,delz',one
print 'nt,dtsub_init,ntsub,dt,nout',two
print 'out_dir', out_dir
print 'vtop,vbottom', three
print 'Ni_max', Ni_max
print 'mproton,mpu,mheavy',masses
print 'np_top,np_bottom',densities
print 'b0_top,b0_bottom',fields
print 'vth_top,vth_bottom',thermal
print 'alpha,beta',alpha,beta
print 'RIo',RIo
