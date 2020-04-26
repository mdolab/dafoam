#!/usr/bin/env python
import sys,copy, os,shutil
from mpi4py import MPI
from regression_helper import *
from pygeo import *
from idwarp import *
from pyspline import *
from dafoam import *
from collections import OrderedDict
import numpy as np

# ###################################################################
# Test: calcDeltaVolPointMat
# ###################################################################

def printHeader(testName):
    if MPI.COMM_WORLD.rank == 0:
        print('+' + '-'*78 + '+')
        print('| Test Name: ' + '%-66s'%testName + '|')
        print('+' + '-'*78 + '+')

printHeader('calcDeltaVolPointMat')
sys.stdout.flush()
                    
defOpts = {
            'outputdirectory':          './',
            'writesolution':            False,
            'updatemesh':               True,
            'usecoloring':              False,

            'designsurfacefamily':      'designsurfaces',
            'designsurfaces':           ['wallsbump'],
            'rasmodel':                 'SpalartAllmaras',
            'divdev2':                  True,
            'flowcondition':            'Incompressible',
            'adjointsolver':            'simpleDAFoam',
            'objfuncs':                 ['CD'],
            'objfuncgeoinfo':           [['wallbbump']],
            'dragdir':                  [1.0,0.0,0.0],
            'setflowbcs':               False,                            
            'nffdpoints':               1,
            'epsderivffd':              1.0e-4, 
            'adjjacmatordering':        'cell',
            'mpispawnrun':              True,
            'adjdvtypes':               ['FFD'],
           }
           
meshOptions = {'gridFile':'./',
                'fileType':'openfoam',
                'symmetryPlanes':[[[0.,0., 0.],[0., 0., 0.]]]}
def resetCase():

    if MPI.COMM_WORLD.rank == 0:
        os.system('rm -fr 0/* [1-9]* *.txt *.dat *.clr *Log* *000 *Coloring* objFunc* psi* *.sh *.info processor* postProcessing')
        if os.path.exists('constant/polyMesh/points_orig.gz'):
            shutil.copyfile('constant/polyMesh/points_orig.gz','constant/polyMesh/points.gz')
            os.remove('constant/polyMesh/points_orig.gz')
    MPI.COMM_WORLD.Barrier()

def runTests(defOpts):

    # first write the reference values
    #refFile='refs/dafoam_test_deltaVolumeMat_reg.ref'
    #if os.path.isdir(refFile):
    #    os.remove(refFile)
    #f=open(refFile,'w')
    #f.write('Eval Functions:\n')
    #f.write('deltaVolumeMatPass\n')
    #f.write('@value 1 1e-10 1e-10\n')
    #f.close()

    aeroOptions = copy.deepcopy(defOpts)

    #change directory to the correct test case
    os.chdir('input/CurvedCubeSnappyHexMesh/')
    resetCase()
    if MPI.COMM_WORLD.rank == 0:
        os.system('cp deltaVolPointMatPlusEps_4.bin deltaVolPointMatPlusEps_4_Ref.bin')
    MPI.COMM_WORLD.Barrier()
            
    # =====================
    # Setup Geometry Object
    # =====================
    
    FFDFile = './FFD/bumpFFD.xyz'
    DVGeo = DVGeometry(FFDFile)
    
    # ref axis
    x = [0.5,0.5]
    y = [0.0,0.1]
    z = [0.5,0.5]
    c1 = pySpline.Curve(x=x, y=y, z=z, k=2)
    DVGeo.addRefAxis('bodyAxis', curve = c1,axis='y')
    # select points
    pts=DVGeo.getLocalIndex(0) 
    indexList=pts[1,1,1].flatten()
    PS=geo_utils.PointSelect('list',indexList)
    DVGeo.addGeoDVLocal('shapey',lower=-0.1, upper=0.1,axis='y',scale=1.0,pointSelect=PS)
    
    # setup CFDSolver
    CFDSolver = PYDAFOAM(options=aeroOptions)
    CFDSolver.setDVGeo(DVGeo)
    mesh = USMesh(options=meshOptions)
    CFDSolver.addFamilyGroup(CFDSolver.getOption('designsurfacefamily'),CFDSolver.getOption('designsurfaces'))
    CFDSolver.setMesh(mesh)
    CFDSolver._writeDeltaVolPointMat()
        
    output={}
    output['deltaVolumeMatPass']=1
    file1='deltaVolPointMatPlusEps_4_Ref.bin'
    file2='deltaVolPointMatPlusEps_4.bin'
    maxDiff,maxRelD=evalMatDiff(file1,file2)
    if maxDiff>1e-12:
        print('%s and %s have different values!'%(file1,file2))
        output['deltaVolumeMatPass']=0
    if MPI.COMM_WORLD.rank == 0:
        print('Eval Functions:')
        reg_write_dict(output, 1e-10, 1e-10)
    
    if MPI.COMM_WORLD.rank == 0:
        os.system('cp deltaVolPointMatPlusEps_4_Ref.bin deltaVolPointMatPlusEps_4.bin')
        os.remove('deltaVolPointMatPlusEps_4_Ref.bin')
    MPI.COMM_WORLD.Barrier()
    
    resetCase()

if __name__ == '__main__':

    # run the airfoil case
    runTests(defOpts)

