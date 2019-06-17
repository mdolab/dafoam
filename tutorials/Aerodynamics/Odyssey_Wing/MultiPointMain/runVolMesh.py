from pyhyp import pyHyp
fileName = 'surfaceMesh.cgns'

options= {
    # ---------------------------
    #        Input Parameters
    # ---------------------------
    'inputFile':fileName,
    'fileType':'cgns',
    'unattachedEdgesAreSymmetry':True,
    'outerFaceBC':'farField',
    'autoConnect':True,
    'BC':{},
    'families':'wall',
    
    # ---------------------------
    #        Grid Parameters
    # ---------------------------
    'N': 25,
    's0':1.0e-3,  
    'marchDist':20*0.5334,
    #'nConstantStart':1,
    
    # ---------------------------
    #   Pseudo Grid Parameters
    # ---------------------------
    'ps0':-1,
    'pGridRatio':-1,
    'cMax':0.2,

    # ---------------------------
    #   Smoothing parameters
    # ---------------------------
    'epsE': 1.0,
    'epsI': 2.0,
    'theta': 3.0,
    'volCoef': .25,
    'volBlend': 0.0005,
    'volSmoothIter': 30,
    #'kspreltol':1e-4,
}

hyp = pyHyp(options=options)
hyp.run()
#hyp.writeCGNS('Odyssey_500K.cgns')
hyp.writePlot3D('volMesh.xyz')


