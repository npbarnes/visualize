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
        return np.array([self._grabData("data/"+self.filename+str(i)+".dat") 
                            for i in reversed(range(11,11+self.numProc))])

    def _getTimesteps(self):
        return len(self.datalst[0])


    # TODO: This function is very inefficient, make it faster.
    # TODO: Eliminate code duplication.
    def shapeData(self):
        if(self.isVec):
            ret = np.empty(shape=(self.timesteps,self.nx,self.ny,self.zrange,3))

            # convert flattened data into 3d array of vectors
            # shapes data from (p,t,xyzc) to (p,t,x,y,z,c)
            redatalst = np.reshape(self.datalst,[self.numProc,self.timesteps,self.nx,self.ny,self.nz,3], 'F')
            # (p,t,x,y,z,c) -> (t,x,y,p,z,c)
            rolledlst = np.rollaxis(redatalst,0,4)
            # (p,t,x,y,zrange,c)
            return np.reshape(rolledlst,[self.timesteps,self.nx,self.ny,self.numProc * self.nz,3])
        else:
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

    def __init__(self,filename,nx,ny,nz,numProc,isVec=False):
        self.filename = filename
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.numProc = numProc
        self.isVec = isVec
        self.zrange = (nz-2)*numProc
        self.datalst = self._getDatalst()
        self.timesteps = self._getTimesteps()
        self.data = self._get3dData()

    # Construct a 2-d slice to view.
    def getSlice(self,t,axs):
        if(axs == 'xy'):
            return np.array([[self.data[t][i][j][self.nz*3/2] for i in range(40,100)] for j in range(self.ny)])
        elif(axs == 'xz'):
            return np.array([[self.data[t][i][self.ny/2][j] for i in range(40,100)] for j in range(self.zrange)])
        elif(axs == 'yz'):
            return np.array([[self.data[t][self.nx/2][i][j] for i in range(self.ny)] for j in range(self.zrange)])

    def getProjection(self,t,axs):
        ret = np.rollaxis(self.getSlice(t,axs),2,0)
        if(axs == 'xy'):
            return (ret[0],ret[1]) 
        elif(axs == 'xz'):
            return (ret[0],ret[2]) 
        elif(axs == 'yz'):
            return (ret[1],ret[2]) 
