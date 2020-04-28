'''
Prepare a nonuniform inlet profile for U
To use this script, first use calcPatchWallDist utility to generate the wall distance values
for inlet patch and save it as wallDistInlet.dat. Then run python interpUin.py.
This script will read the experimental profile ExpUProfile.dat and interpolate the velocity
according to the wall distance output. You can copy the generated UInNonUniformField.dat to
the inlet patch of U
'''
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy

U0=8.4

f=open('ExpUProfile.dat','r')
lines=f.readlines()
expY=[]
expU=[]
for line in lines:
    cols=line.split()
    expY.append(float(cols[0]))
    expU.append(float(cols[1]))
f.close()

numpy.asarray(expY)
numpy.asarray(expU)

interpF=interpolate.interp1d(expY,expU)

f=open('wallDistInlet.dat','r')
lines=f.readlines()
wallDistY=[]
for line in lines:
    cols=line.split()
    wallDistY.append(float(cols[0]))
f.close()

maxY=max(wallDistY)
numpy.asarray(wallDistY)
counterI=0
for val in wallDistY:
    wallDistY[counterI]=val/maxY/2.0
    counterI+=1

wallDistU=interpF(wallDistY)

f=open('UInNonUniformField.dat','w')
for val in wallDistU:
    f.write('(%f 0 0)\n'%(val*U0))
f.close()

plt.plot(expY,expU,'o',wallDistY,wallDistU,'s')
plt.show()

