import FortranFile as ff
import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

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

ax1 = fig.add_subplot(2,1,1,xlim=(0,500), ylim=(-0.8,0.8))
line1, = ax1.plot([],[])

ax2 = fig.add_subplot(2,1,2,xlim=(0,500), ylim=(-0.8,0.8))
line2, = ax2.plot([],[])

data1 = map(extractBxField, getData('c.b1.dat'))
data2 = map(extractBxField, getData('c.b1.dat_5'))

def init():
    line1.set_data([],[])
    line2.set_data([],[])
    return line1,

def update(i):
    line1.set_data(range(len(data1[i])),data1[i])
    line2.set_data(range(len(data2[i])),data2[i])
    return line1, line2,


animate = animation.FuncAnimation(fig, update, init_func=init, frames=len(data1), interval=100, blit=True)

#plt.ion()
plt.show()

#for step1, step2 in zip(data1,data2):
#    plt.subplot(211)
#    plt.plot(range(len(step1)),step1)
#    plt.axis([0,500,-0.8,0.8])
#    plt.draw()
#
#    plt.subplot(212)
#    plt.plot(range(len(step2)),step2)
#    plt.axis([0,500,-0.8,0.8])
#    plt.draw()
#
#    time.sleep(0.01)
#    plt.cla()
#    plt.subplot(211)
#    plt.cla()
