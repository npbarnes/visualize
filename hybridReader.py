import FortranFile as ff
import numpy as np

class hybridReader:

    def _grabData(self,filename):
        datafile = ff.FortranFile(filename)

        data = []
        while(True):
            try:
                # For whatever reason m (time step) is output every other record.
                # Burn these values they won't be used.
                datafile.readReals()
                data.append(datafile.readReals().tolist())
            except IOError:
                break

        return np.array(data)

    def _getDatalst(self):
        # taken in reversed order to make them go from top to bottom.
        return [self._grabData("data/c.np_3d_"+str(i)+".dat") for i in reversed(range(11,11+self.numProc))]

    def _getTimesteps(self):
        return len(self.datalst[0])


    # TODO: This function is very inefficient, make the whole thing in numpy arrays.
    def shapeData(self):
        ret = []
        for i in range(self.timesteps):
            tmp = []
            for proc in self.datalst:
                tmp.append(proc[i][:-2*self.nx*self.ny])
            # flatten
            tmp = [val for xyz in tmp for val in xyz]
            # reshape to 3-d grid
            tmp = np.reshape(tmp, [self.nx,self.ny,self.zrange],'F').tolist()
            # timestep is ready to be returned
            ret.append(tmp)
        return np.array(ret)

    def _get3dData(self):
        return self.shapeData()

    def __init__(self,nx,ny,nz,numProc):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.numProc = numProc
        self.zrange = (nz-2)*numProc
        self.datalst = self._getDatalst()
        self.timesteps = self._getTimesteps()
        self.data = self._get3dData()

    # Construct a 2-d slice to view.
    def getSlice(self,t,axs):
        if(axs == 'xy'):
            #return data[t,40:100,:,nz/2]
            return [[self.data[t][i][j][self.nz*3/2] for i in range(40,100)] for j in range(self.ny)]
        elif(axs == 'xz'):
            #return data[t,40:100,ny/2,:]
            return [[self.data[t][i][self.ny/2][j] for i in range(40,100)] for j in range(self.zrange)]
        elif(axs == 'yz'):
            #return data[t,nx/2,:,:]
            return [[self.data[t][self.nx/2][i][j] for i in range(self.ny)] for j in range(self.zrange)]
