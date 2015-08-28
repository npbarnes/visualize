import FortranFile as ff
import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import sys

def getData(filename):
    datafile = ff.FortranFile(filename)

    data = []
    while(True):
        try:
            # For whatever reason m (presumably mass) is output every other record.
            # Burn these values they won't be used.
            datafile.readReals()
            data.append(numpy.reshape(datafile.readReals(),[3,3,500,3],'F'))
        except IOError:
            break

    return data

def extractBxField(step):
    bx = []
    for z in step[1][1]:
        bx.append(z[0])

    return bx


fig = plt.figure()

numplots = len(sys.argv)-1
all_ax = [fig.add_subplot(numplots,1,i,xlim=(0,500), ylim=(-0.8,0.8)) for i in range(numplots)]
all_lines = [ax.plot([],[])[0] for ax in all_ax]
all_data = [map(extractBxField, getData(filename)) for filename in sys.argv[1:]]

def init():
    for line in all_lines:
        line.set_data([],[])
    return all_lines[0],

def update(i):
    for line, data in zip(all_lines,all_data):
        line.set_data(range(len(data[i])), data[i])
    return tuple(all_lines)

animate = animation.FuncAnimation(fig, update, init_func=init, frames=len(all_data[0]), interval=100, blit=True)

plt.show()
