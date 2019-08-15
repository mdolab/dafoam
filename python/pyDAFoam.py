#!/usr/bin/env python
"""

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    The Python interface to DAFoam.
    It controls the adjoint solvers and external modules for design optimization

"""

# =============================================================================
# Imports
# =============================================================================
import os,shutil
import re
import time
import copy
import gzip
import sys,subprocess
import numpy as np
from mpi4py import MPI
from baseclasses import AeroSolver, AeroProblem
from openfoammeshreader import of_mesh_utils as ofm
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
try:
    from collections import OrderedDict
except ImportError:
    try:
        from ordereddict import OrderedDict
    except ImportError:
        print('Could not find any OrderedDict class. For 2.6 and earlier, \
use:\n pip install ordereddict')




class Error(Exception):
    """
    Format the error message in a box to make it clear this
    was a expliclty raised exception.
    """
    def __init__(self, message):
        msg = '\n+'+'-'*78+'+'+'\n' + '| pyDAFoam Error: '
        i = 19
        for word in message.split():
            if len(word) + i + 1 > 78: # Finish line and start new one
                msg += ' '*(78-i)+'|\n| ' + word + ' '
                i = 1 + len(word)+1
            else:
                msg += word + ' '
                i += len(word)+1
        msg += ' '*(78-i) + '|\n' + '+'+'-'*78+'+'+'\n'
        print(msg)
        Exception.__init__(self)


class PYDAFOAM(AeroSolver):
    """
    Create an instance of pyDAFoam to work with. 

    Parameters
    ----------   

    comm : mpi4py communicator
        An optional argument to pass in an external communicator.
        Note that this option has not been tested with openfoam.

    options : dictionary
        The list of options to use with pyDAFoam.

    debug : Boolean
        An optional flag to enable debugging options

    """ 

    def __init__(self, comm=None, options=None, debug=False): 

        # Information for base class:
        name = 'PYDAFOAM'
        category = 'Three Dimensional CFD'
        informs = {}

        # If 'options' is not None, go through and make sure all keys
        # are lower case:
        if options is not None:
            for key in options.keys():
                options[key.lower()] = options.pop(key)
        else:
            raise Error('The \'options\' keyword argument must be passed \
            pyDAFoam. The options dictionary must contain (at least) the gridFile \
            entry for the grid')

        # Load all the option/objective/DV information:
        defOpts = self._getDefOptions() 
        self.imOptions = self._getImmutableOptions()

        # This is the real solver so dtype is 'd'
        self.dtype = 'd'

        # Next set the MPI Communicators and associated info
        if comm is None:
            comm = MPI.COMM_WORLD
        # end

        self.comm = comm

        # Initialize the inherited aerosolver
        AeroSolver.__init__(self, name, category, defOpts, informs, options=options)

        # now we handle the special key updatedefaultdicts and update default dictionaries
        updateDefaultDicts = self.getOption('updatedefaultdicts')
        if not len(updateDefaultDicts)==0:
            for key in updateDefaultDicts.keys():
                defaultDict = self.getOption(key)
                defaultDict.update(updateDefaultDicts[key])
                self.setOption(key,defaultDict)

        # Remind the user of all the DAFoam options:
        if self.getOption('printalloptions'):
            self.printCurrentOptions()

        # Since we are using an OpenFOAM solver, the problem is assumed
        # from the run directory
        
        # check whether we are running in parallel
        nProc = self.comm.size
        
        self.parallel = False
        if nProc > 1:
            self.parallel = True
        # end
                
        # setup for restartopt  
        if self.getOption('restartopt'):            
            self.skipFlowAndAdjointRuns = True
        else:
            self.skipFlowAndAdjointRuns = False
            
        # Save the rank and number of processors
        self.rank = self.comm.rank
        self.nProcs = self.comm.size

        argv = sys.argv
        if self.parallel:
            argv.append('-parallel')
       
        # get current directory
        dirName = os.getcwd()

        # Initialize the case data that will be filled in readGrid function
        self.fileNames = {}
        self.x0 = None
        self.x = None
        self.faces = None
        self.boundaries = None
        self.owners = None
        self.neighbours = None
        self.nCells = None

        # Setup a fail flag for flow solver
        self.meshQualityFailure=False

        # set the startFrom variable
        self.solveFrom = 'startTime'
        
        # set the solveAdjoint value to None to start
        self.solveAdjoint = None
        
        # write a basic control and fvAdjoint file
        self._writeControlDictFile()
        self._writeAdjointDictFile()
         
        # Misc setup, e.g., copy points to points_orig for the first run
        # decompose the domain if running in parallel, running genFaceCenters and genWallFaces etc.
        self.miscSetup()
    
        # read the openfoam case mesh information
        self._readGrid(dirName)        
    
        # get the reduced point connectivities for the base patches in the mesh
        self._computeBasicFamilyInfo()

        # Add a couple of special families. 
        self.allFamilies = 'allsurfaces'
        self.addFamilyGroup(self.allFamilies, self.basicFamilies)
    
        self.allWallsGroup = 'all'
        self.addFamilyGroup(self.allWallsGroup, self.wallList)
    
        # Set the design families if given, otherwise default to all
        # walls
        self.designFamilyGroup = self.getOption('designSurfaceFamily')
        if self.designFamilyGroup is None:
            self.designFamilyGroup = self.allWallsGroup
    
        # Set the mesh families if given, otherwise default to all
        # walls
        self.meshFamilyGroup = self.getOption('meshSurfaceFamily')
        if self.meshFamilyGroup is None:
            self.meshFamilyGroup= self.allWallsGroup
    
        self.coords0 = self.getSurfaceCoordinates(self.allFamilies)
       
       
        # By Default we don't have an external mesh object or a
        # geometric manipulation object
        self.mesh = None
        self.DVGeo = None

        # this allows us to print mesh in tecplot format
        self.printMesh = False

        # Create a name to var mapping for cost function
        self.possibleObjectives = {'CD':'CD',
                                   'CL':'CL',
                                   'CMX':'CMX',
                                   'CMY':'CMY',
                                   'CMZ':'CMZ',
                                   'CPL':'CPL',
                                   'NUS':'NUS',
                                   'AVGV':'AVGV',
                                   'VARV':'VARV',
                                   'AVGS':'AVGS',
                                   'VMS':'VMS',
                                   'TPR':'TPR',
                                   'TTR':'TTR',
                                   'MFR':'MFR'
                                   }

        # output file counter
        self.flowRunsCounter = 0
        self.adjointRunsCounter = 0
        
        # For multipoint indexing
        self.multiPointFCIndex = None
        
        if self.comm.rank == 0:
            print('Done Init.')

        return

        
    # ------------------------
    # Default pyDAFoam Options
    # ------------------------

    #**************************************** NOTE ***************************************#
    #** whenever you add a new input parameter, remember to add its documentation here! **#
    #**************************************** NOTE ***************************************#
    def aCompleteInputParameterSet(self):
        """
        All input parameters

        Parameters
        ----------
        updatedefaultdicts: dict
            A special key that allows to add/modify the default dictionaries, which avoids explicitly writing out all dict values.
            e.g., updatedefaultdicts: {'divschemes':{'div(phi,U)':'Gauss upwind'}} will modify the 'div(phi,U)' value to 'Gauss upwind' for the 'divschemes' option while keeping all the other default keys unchanged.

        casename : str
            Name for the case to save intermediate optimization results
        
        outputdirectory : str
            Output directory for saving intermediate optimization results. Do NOT set this to the current path

        writesolution : bool
            Whether to write intermediate results to disk

        printalloptions : bool
            Whether to print all parameters before starting an optimization

        maxflowiters : int
            Max non-linear iterations for the flow solver

        writeinterval : int
            Field output interval for the flow solver, this typically equals to maxflowiters

        writecompress : str
            Whether to compress the results using .gz format
        
        writeformat : str
            IO format: can be either ascii or binary
        
        residualcontrol : float
            Relative residual convergence tolerance (non-linear iteration tolerance) for flow solvers

        nonorthogonalcorrectors : int
            Number of non-orthogonal corrections. Typically is 0 for steady-state solvers

        fvsolvers: dict
            Dictionary for solvers in fvSolution
        
        consistent : bool
            Whether to use coupled p and U solution technique for flow solvers
        
        divschemes : dict
            divScheme in fvSchemes for flow and adjoint

        gradschemes : dict
            gradScheme in fvSchemes for flow and adjoint
            Note: to limit grad use: grad(U) cellLimited Gauss linear;
        
        sngradschemes : dict
            snGradScheme in fvSchemes for flow and adjoint
        
        interpolationschemes : dict
            interpolationScheme in fvSchemes for flow and adjoint
        
        laplacianschemes : dict
            laplacianScheme in fvSchemes for flow and adjoint
        
        ddtschemes : dict
            ddtScheme in fvSchemes for flow and adjoint
        
        d2dt2schemes : dict
            d2dt2Scheme in fvSchemes for flow and adjoint
        
        fluxrequired : list
            fluxRequired in fvSchemes for flow and adjoint
        
        divuadj : str
            div(phi,U) scheme for adjoint only. 
            If divuadj is non-empty, run adjoint using divuadj for div(phi,U) instead. 
            This allows us to use 2nd order scheme to run the flow while using 1st order scheme to solve adjoint. 
            This is sometimes useful when the flow converges poorly.

        pMinFactor (pMaxFactor) : float
            Min and max factor to constrain pressure

        rhoMin (rhoMax) : float
            Min and max factor to constrain density

        fvrelaxfactors : dict
            relaxation factors for fvSolution
        
        divdev2 : bool
            Whether to use the dev2 scheme. NOTE: dev2 scheme is used after OpenFOAM-2.4.x
        
        walldistmethod : str
            Methods to compute wall distance
        
        transproperties : dict
            Values for transportProperties, e.g., {'nu':1.5e-5,'TRef':300.0}
        
        thermoproperties : dict
            Values for thermophysicalProperties, e.g., {'Pr':0.7,'molWeight':28.0}
        
        thermotype : dict
            Entires for thermoType in thermophysicalProperties, e.g., {'type':'hePsiThermo','energy':'sensibleInternalEnergy'}
        
        radmodel : str
            RASModel, now supports SpalartAllmaras, SpalartAllmarasFv3, kOmegaSST, kEpsilon, LaunderSharmaKE
        
        rasmodelparameters: dict
            Optional turbulence parameters defined in turbulenceProperties-RAS
        
        actuatoractive : int
            Whether to add actuator source terms to mimic the impact of propellers
        
        actuatoradjustthrust : int
            Whether to automatically adjust the total thrust of the source terms, and let it equals to the drag of design surface
        
        actuatorvolumenames : list
            List of user-defined volumes for the actuator propeller. The source terms will be then added into these volumes. Set the volume geometric information in "userdefinedvolumeinfo"
        
        actuatorthrustcoeff : list
            If not adjust thrust, we can prescribe the thrust of the source terms.
        
        actuatorpoverd : list
            The pitch-to-diameter of the actuator propellers. This will be used to related thrust and torque
        
        actuatorrotataiondir : list
            Rotation direction of the propellers. right: the propeller is rotating towards the starboard side. left: the propeller is rotating towards the port side.
        
        adjointsolver : str
            Name of the discrete adjoint solver. Currently support: adjointSolverSimpleFoam, adjointSolverBuoyantBoussinesqSimpleFoam, adjointSolverRhoSimpleFoam
        
        flowcondition : str
            Flow condition. Options are: Incompressible, Compressible  
        
        usecoloring : bool
            Whether to use graph coloring. Set it to false for debugging only.

        adjdvtypes : list
            List of adjoint design variable types. Options are UIn (inlet velocity), Xv (volume coordinates), FFD (FFD coordinates), and Vis (molecular viscosity)
        
        epsderiv : float
            Finite-differene step size for partial derivative wrt state variables
        
        epsderivffd : float
            Finite-differene step size for partial derivative wrt FFD coordinates
        
        epsderivxv: float
            Finite-differene step size for partial derivative wrt volume coordinates
        
        epsderivuin : float
            Finite-differene step size for partial derivative wrt velocity inlet
        
        epsderivvis : float
            Finite-differene step size for partial derivative wrt molecular viscosity
        
        jacmatordering : str
            State variable ordering strategy, can be either cell: cell-by-cell or state: state-by-state ordering
        
        jacmatreordering : str
            Reordering strategy for the state Jacobian matrices. Reordering can save PC fill-in memory usage. Options are: natural: no-reordering, nd: Nested Dissection, rcm: Reverse Cuthill-McKee
        
        mintoljac : float
            A lower bound to limit the min value in the Jacobian matrices

        maxtoljac : float
            A upper bound to limit the max value in the Jacobian matrices
        
        mintolpc : float
            A lower bound to limit the min value in the PC Jacobian matrices

        maxtolpc : float
            A upper bound to limit the max value in the PC Jacobian matrices
        
        stateresettol : float
            Residual L2 diff tolerance for checking if the state perturbation is properly reset. Default is 1e-6. 
            Normally users don't need to change the default.
            But if you keep getting erroring saying L2 norm not same, and you are sure you implementation is correct. Increase this number. 
        
        tractionBCMaxIter : label
            maximal iteration number to iteratively correct D and gradD. Default is 20.
            Normally users don't need to change the default.
            If dRdWT becomes very slow, decrease this number.
        
        correctwalldist : bool
            Whether to update wall distance when volume coordinates are changed
        
        reducerescon4jacmat : bool
            Whether to reduce the max connectivity level for state Jacobian matrices. When set it to True, it will reduce the con level according to the values in maxresconlv4jacpcmat. One should not set it to true unless you have severe flow convergence issue. This treatment will significantly reduce the adjoint derivative accuracy.
        
        maxresconlv4jacpcmat : dict
            Upper bound for the max connectivity level in PC state Jacobian matrix dRdWTPC. Reducing the max con level will reduce the memory usage when doing the ILU fill-in of PC state Jacobian, it also increases the conditioning of the PC matrix and improves convergence. Typically, we need to reduce the max con level for p to 2 and max con level for phi to 1. NOTE: when reducerescon4jacmat is set to True, we will also reduce the max con level for state Jacobian matrix dRdWT

        delturbprod4pcmat : bool
            Whether to delete contribution of the turbulence production terms for the PC state Jacobian matrix
        
        calcpcmat : bool
            Whether to compute the PC state Jacobian matrix dRdWPC. If set it to False, we will use the state Jacobian matrix dRdWT for preconditioning
        
        fastpcmat : bool
           Whether to use the OpenFoam built-in fvMatrix to assemble the PC mat, it will be fast
        
        writematrices : bool
            Whether to write all partial derivative matrices and vectors to disk. This is for debugging only

        adjgmrescalceigen: bool
            whether to compute the preconditioned eigenvalues during the adjoint GMRES solution
        
        adjgmresrestart : int
            GMRES solver restart number. Typically, we set it to adjgmresmaxiters; we never restart
        
        adjglobalpciters : int
            Global Richardson iteration number. Typically, we set it to 0. Setting it to non-zero might improve convergence, but the speed becomes very slow
        
        adjlocalpciters : int
            Local Richardson iteration number. Similarly, we set it to 1
        
        adjasmoverlap : int
            ASM global PC overlap level. 1 is usually good. Set it to a larger value may improve convergence, but requires more memory
        
        adjpcfilllevel : int
            Fill-in level of the ILU local PC. This is a critical parameter for adjoint convergence. Most of the case, 1 is enough. But if the flow converges poorly, we may need to set it to 2. The memory usage significantly increases when increasing this parameter. Don't set it to larger than 2
        
        adjgmresmaxiters : int
            Max iteration number for the adjoint linear equation solution
        
        adjgmresabstol : float
            Absolute tolerance of the adjoint linear equation solution
        
        adjgmresreltol : float
            Relative tolerance of the adjoint linear equation solution
        
        normalizeresiduals : list
            List of residual names to normalize, e.g, {'URes','pRes'}
            Note: each residual will be normalized by their cell volume or face area, if their names are found
            in the list
        
        normalizestates : list
            List of state names to normalize, e.g., {'U','p'}
        
        statescaling : dict
            When we need to normalize states, set the scaling factors here
        
        residualscaling : dict
            This is to further scale the residual using the prescribed values here. This is to basically ensure
            that all residuals have similar magnitudes

        nffdpoints : int
            Number of FFD design variables. This will be used to read the eltaVolPointMatPlusEps matrix in the OpenFOAM layer
        
        setflowbcs : bool
            Whether to set flow boundary conditions
        
        flowbcs : dict
            When setflowbcs is set to True, give the flow boundary values here.
            The useWallFunction key denotes whether to apply appropriate wall boundary conditions to turbulence variables
            'flowbcs':  {'bc0':{'patch':'inlet','variable':'U','value':[20.0,0.0,0.0]},
                         'bc1':{'patch':'outlet','variable':'p','value':[0.0]},
                         'bc2':{'patch':'inlet','variable':'k','value':[0.06]},
                         'bc3':{'patch':'inlet','variable':'omega','value':[400.0]},
                         'bc4':{'patch':'inlet','variable':'epsilon','value':[2.16]},
                         'bc5':{'patch':'inlet','variable':'nuTilda','value':[1.5e-4]},
                         'bc6':{'patch':'inlet','variable':'T','value':[300.0]},
                         'useWallFunction':'true'}
        
        inletpatches : list
            List of inlet patch names
        
        outletpatches : list
            List of outlet patch names

        derivuininfo : dict
            Information of user-defined patches for UIn derivative. 
            Example of dF/dUxInlet
                example: {
                            'stateName':'U',
                            'component':0,
                            'patchNames':['inlet']
                          }
        
        userdefinedpatchinfo : dict
            Information of user-defined patches. Currently support:
            A patch defined within a box:
                example: {'userDefinedPatch0':{
                                               'type':'box',
                                               'stateName':'U',
                                               'component':0,
                                               'scale':1.0,
                                               'centerX':0.5,
                                               'centerY':0.5,
                                               'centerZ':0.5,
                                               'sizeX':0.1,
                                               'sizeY':0.1,
                                               'sizeZ':0.1}}
            An existing patch in the mesh:
                example: {'userDefinedPatch0':{
                                               'type':'patch',
                                               'patchName':'inlet',
                                               'stateName':'U',
                                               'component':0,
                                               'scale':1.0}}
        
        userdefinedvolumeinfo : dict
            Information for user-defined volumes. Currently support:
            A volume defined within a cylinder:
            example: {'userDefinedPatch0':{
                                           'type':'cylinder',
                                           'stateName':'U',
                                           'component':0,
                                           'scale':1.0,
                                           'centerX':0.5,
                                           'centerY':0.5,
                                           'centerZ':0.5,
                                           'width':0.1,
                                           'radius':0.3,
                                           'axis':'x'}}
            A volume defined within a annulus:
                example: {'userDefinedPatch0':{
                                               'type':'annulus',
                                               'stateName':'U',
                                               'component':0,
                                               'scale':1.0,
                                               'centerX':0.5,
                                               'centerY':0.5,
                                               'centerZ':0.5,
                                               'width':0.1,
                                               'radiusInner':0.1,
                                               'radiusOuter':0.5,
                                               'axis':'x'}}
        
        referencevalues: dict
            Reference values for objective function computation. Fail to set appropriate reference values will get Seg Fault. Typical entities are:
            magURef : Referene velocity magnitude used in CD, CL, CMX, CMY, CMZ, CPL. 
            rhoRef : Reference density. For incompressible flow, set it to 1.0. For compressible flow, set it to far field density
            pRef : Reference pressure. Always set it to 0.0
            ARef : Reference area used in CD, CL, CMX, CMY, CMZ
            LRef : Reference length used in CMX, CMY, CMZ, NUS
        
        usenksolver : bool
            Whether to use NK solver

        nksegregatedturb : bool
            Whether to segregate turbulence var in NK
        
        usenksolver : bool
            Whether to segregate phi in NK

        nkewrtol0 : float
            The initial tolerance for EW
        
        nkewrtolmax : float
            The max tolerance for EW

        nkpclag : int
            How frequent to update the pc mat for NK

        nkgmresrestart : int
            Restart number for the NK GMRES linear solver

        nkasmoverlap : int
            ASM PC overlap for the NK GMRES linear solver

        nkpcfilllevel : int
            ILU PC fill-in level for the NK GMRES linear solver
        
        nkgmresmaxiters : int
            Max iteration number for the NK GMRES linear solver
                
        nkjacmatreordering : word
            Reordering strategy for the NK state Jacobian matrices. Reordering can save PC fill-in memory usage. Options are: natural: no-reordering, nd: Nested Dissection, rcm: Reverse Cuthill-McKee
            
        nkreltol : float
            Relative tolerance for the NK nonlinear solution

        nkabstol : float
            Absolute tolerance for the NK nonlinear solution

        nkstol : float
            S tolerance (difference between two nonliner steps) for the NK nonlinear solution 

        nkmaxiters : int
            Max iteration number for the NK nonlinear solution

        nkmaxfuncevals : int
            Max function evaluations for the NK nonlinear solution
        
        objfuncgeoinfo : list
            ListList of the objective function geometric information. This can be the list of design surface names or names of user-defined volumes/patches for AVGV and VARV objs

        objfuncs : list
            List of objective functions for adjoint. Currently support:
            CD: drag coefficient
            CL: lift coefficient
            CMX: moment in x
            CMY: moment in y
            CMZ: moment in z
            CPL: total pressure loss coefficient for duct flows
            NUS: Nusselt number
            AVGV: averaged state variable values in a specified user-defined volume
            VARV: variance of state variable values in a specified user-defined volume 
            AVGS: averaged state variable values over specified internal or boundary patches

        dragdir : list
            Vector of drag direction
        
        liftdir : list
            Vector of lift direction
        
        cofr : list
            Vector of center of rotation
        
        rotrad : list
            rotational speed vector in rad/s

        meshmaxaspectratio : float
            Max tolerant mesh aspect ratio when doing checkMesh. Note we may want to relax the OpenFOAM-default tolerance a bit especially for low y+ mesh
        
        meshminfacearea : float
            Min tolerant mesh face area when doing checkMesh
        
        meshminvol : float
            Max tolerant mesh volume when doing checkMesh
        
        meshmaxnonortho: float
            Max tolerant mesh non-orthogonal angle when doing checkMesh
        
        meshmaxskewness : float
            Max tolerant mesh skewness when doing checkMesh
        
        meshsurfacefamily : object
            Names of the mesh surface family (group)
        
        designsurfacefamily : object
            Names of the design surface family (group)
        
        designsurfaces : list
            List of design surface names
        
        dummy : str
            A dummy parameter
        
        objstdthres : float
            A tolerance for the standard deviation of objective function. If larger than this tol, we consider the flow not converging
        
        postprocessingdir : str
            Name for postProcessing dir
        
        filechecktime : float
            Time interval to check whether the flow/adjoint simulation is done. This is useful when mpispawnrun is False
        
        multipointopt : bool
            Whether to run multipoint optimization
        
        restartopt : bool
            Whether to restart the adjoint. If truee, it will read the logs from the outputdirectory. NOTE: always restart an optimization fromt the flow solution step. If the optimization dies at the adjoint solution step. Delete all the adjoint solution logs and the previous flow solution logs, then restart.
        
        avgobjfuncs : bool
            Whether to average the objective function. This is only needed for poor flow convergence

        avgobjfuncsstart : int
            From which step to average the objfuncs?
        
        mpispwanrun : bool
            Whether to use the spawnrun option for parallel runs. For a small number of CPUs, e.g., 4, set it to True. When running on HPC, set it to False, and use the foamRun.sh script in the tutorials.
        
        runpotentialfoam : bool
            Whether to run a potentialFoam before running the flow solvers. This can prevent nan solution because of the bad initial field
        
        updatemesh : bool
            Whether to update mesh. If you run an optimization, set it to True. If you just want to run the flow and adjoint, and your surface and volume mesh never change, set it to False. When testing the regression, we need to set it to False to ensure consistent mesh over different computer planforms.
        
        preservepatches : list
            Patches name for preservePatches which is needed for cycic boundary conditions in parallel
        
        singleprocessorfacesets : list
            Face set name for singleprocessorfacesets which is needed for cycicAMI boundary conditions in parallel

        decomposepardict: dict
            Information for decomposeParDict

        """

        # this is a dummy function for sphinx to pick up the documentation

        return

    #**************************************** NOTE ***************************************#
    # whenever you add a new input parameter,                                             #
    # remember to add its documentation to the above aCompleteInputParameterSet function! #
    #**************************************** NOTE ***************************************#
    def _getDefOptions(self):
        """
        There are many options for PYDAFOAM. These technically belong in
        the __init__ function but it gets far too long so we split
        them out. 
        
        Note: all defOpts must be defined lower case.
        """
        defOpts = { # a special key to add/modify default dictionary without explicitly write them all 
                    'updatedefaultdicts':[dict,{}],
                    
                    #Output options
                    'casename':[str,'pyDAFoamCase'],
                    'outputdirectory':[str,'../'],
                    'writesolution':[bool,False],
                    'writelinesearch':[bool,False],
                    'printalloptions':[bool,True],
         
                    # controlDict
                    'maxflowiters':[int,1000],
                    'writeinterval':[int,200],
                    'writecompress':[str,'on'],
                    'writeformat':[str,'ascii'],
                     
                    # fvSolution
                    'residualcontrol':[float,1.0e-20],
                    'simplecontrol':[dict,{'nNonOrthogonalCorrectors':'0',
                                           #'pRefCell':'0',
                                           #'pRefValue':'0',
                                           #'rhoMax':'2',
                                           #'rhoMin':'0.5',
                                           #'pMinFactor':'0.5',
                                           #'pMaxFactor':'1.5',
                                           'rhoLowerBound':'0.2',
                                           'rhoUpperBound':'10.0',
                                           'pLowerBound':'20000',
                                           'pUpperBound':'1000000',
                                           'ULowerBound':'-1000',
                                           'UUpperBound':'1000',
                                           'eLowerBound':'50000',
                                           'eUpperBound':'800000',
                                           'hLowerBound':'50000',
                                           'hUpperBound':'800000',
                                           'consistent':'false',
                                           'transonic':'false'}],
                    'pisocontrol':[dict,{'nNonOrthogonalCorrectors':'0',
                                         'nCorrectors':'2',
                                         'pRefCell':'0',
                                         'pRefValue':'0',
                                         'maxCo':'0.5',
                                         'rDeltaTSmoothingCoeff':'0.5',
                                         'maxDeltaT':'0.5'}],
                    'fvsolvers':[dict,{'"(p|p_rgh|G)"':{'solver':'GAMG',
                                                    'tolerance':'1e-20',
                                                    'relTol':'0.1',
                                                    'smoother':'GaussSeidel',
                                                    'cacheAgglomeration':'true',
                                                    'agglomerator':'faceAreaPair',
                                                    'nCellsInCoarsestLevel':'10',
                                                    'nPreSweeps':'0',
                                                    'nPostSweeps':'2',
                                                    'mergeLevels':'1',
                                                    'maxIter':'10'},
                                       '"(p|p_rgh|G)Final"':{'solver':'GAMG',
                                                    'tolerance':'1e-06',
                                                    'relTol':'0',
                                                    'smoother':'GaussSeidel',
                                                    'cacheAgglomeration':'true',
                                                    'agglomerator':'faceAreaPair',
                                                    'nCellsInCoarsestLevel':'10',
                                                    'nPreSweeps':'0',
                                                    'nPostSweeps':'2',
                                                    'mergeLevels':'1',
                                                    'maxIter':'500'},
                                       '"(U|T|e|h|nuTilda|k|omega|epsilon|ReThetat|gammaInt)"':
                                                    {'solver':'smoothSolver',
                                                     'tolerance':'1e-20',
                                                     'relTol':'0.1',
                                                     'smoother':'GaussSeidel',
                                                     'nSweeps':'1',
                                                     'maxIter':'10'},
                                        '"(U|T|e|h|nuTilda|k|omega|epsilon|ReThetat|gammaInt)Final"':
                                                    {'solver':'smoothSolver',
                                                     'tolerance':'1e-06',
                                                     'relTol':'0',
                                                     'smoother':'GaussSeidel',
                                                     'nSweeps':'1',
                                                     'maxIter':'500'},
                                       'Phi':{'solver':'GAMG',
                                              'tolerance':'1e-6',
                                              'relTol':'1e-2',
                                              'smoother':'DIC',
                                              'maxIter':'50',
                                              'cacheAgglomeration':'true',
                                              'agglomerator':'faceAreaPair',
                                              'nCellsInCoarsestLevel':'10',
                                              'mergeLevels':'1',}}],
                    'fvrelaxfactors':[dict,{'fields':{'p':0.3,
                                                      'rho':0.3,
                                                      'p_rgh':0.3},
                                            'equations':{'U':0.7,
                                                         'nuTilda':0.7,
                                                         'k':0.7,
                                                         'epsilon':0.7,
                                                         'omega':0.7,
                                                         'ReThetat':0.7,
                                                         'gammaInt':0.7,
                                                         'h':0.7,
                                                         'T':0.7,
                                                         'e':0.7,
                                                         'G':0.7,}}],

                    
                    # fvSchemes
                    'divschemes':[dict,{'default':'none',
                                        'div(phi,U)':'bounded Gauss linearUpwindV grad(U)',
                                        'div(phi,T)':'bounded Gauss upwind',
                                        'div(phi,e)':'bounded Gauss upwind',
                                        'div(phi,h)':'bounded Gauss upwind',
                                        'div(phi,nuTilda)':'bounded Gauss upwind',
                                        'div(phi,k)':'bounded Gauss upwind',
                                        'div(phi,omega)':'bounded Gauss upwind',
                                        'div(phi,epsilon)':'bounded Gauss upwind',
                                        'div(phi,ReThetat)':'bounded Gauss upwind',
                                        'div(phi,gammaInt)':'bounded Gauss upwind',
                                        'div(phi,Ekp)':'bounded Gauss upwind',
                                        'div(phid,p)':'Gauss upwind',
                                        'div(phi,K)':'bounded Gauss upwind',
                                        'div(pc)':'bounded Gauss upwind',
                                        'div((nuEff*dev2(T(grad(U)))))':'Gauss linear',
                                        'div((p*(U-URel)))': 'Gauss linear',
                                        'div((-devRhoReff.T()&U))':'Gauss linear',
                                        'div(sigmaD)':'Gauss linear',
                                        'div(((rho*nuEff)*dev2(T(grad(U)))))':'Gauss linear',
                                        'div((phi|interpolate(rho)),p)':'Gauss upwind',
                                        }],
                    'gradschemes':[dict,{'default':'Gauss linear'}],
                    'interpolationschemes':[dict,{'default':'linear'}],
                    'ddtschemes':[dict,{'default':'steadyState'}],
                    'd2dt2schemes':[dict,{'default':'steadyState'}],
                    'laplacianschemes':[dict,{'default':'Gauss linear corrected'}],
                    'sngradschemes':[dict,{'default':'corrected'}],
                    'fluxrequired':[list,['p','p_rgh','Phi']],
                    'divuadj':[str,''], # if divuadj is non-empty, run adjoint using divuadj instead 
                    'divdev2':[bool,True],
                    'walldistmethod':[str,'meshWave'],

                    # transportProperties
                    'transproperties':[dict,{'nu':1.5e-5,
                                             'TRef':273.15,
                                             'beta':3e-3,
                                             'Pr':0.7,
                                             'Prt':0.85,
                                             'DT':4e-5,
                                             'g':[0.0,0.0,0.0],
                                             'rhoRef':1.0,
                                             'CpRef':1005.0}],
                    
                    'radiationproperties':[dict,{'absorptivity':0.5,
                                                 'emissivity':0.5,
                                                 'E':0.0,
                                                 'radiation':'off',
                                                 'radiationModel':'none',
                                                 'solverFreq':1}],
                    
                    'thermoproperties':[dict,{'molWeight':28.97,
                                              'Cp':1005.0,
                                              'Hf':0.0,
                                              'mu':1.8e-5,
                                              'Pr':0.7,
                                              'TRef':300.0,
                                              'Prt':1.0,}],
                    
                    'thermalproperties':[dict,{'C':434.0,
                                               'k':60.5,
                                               'alpha':1.1e-5,
                                               'thermalStress':'false'}],
                    
                    'mechanicalproperties':[dict,{'rho':7854.0,
                                                  'nu':0.0,
                                                  'E':2e11}],

                    'mrfproperties':[dict,{'active':'false',
                                           'selectionmode':'cellZone',
                                           'cellzone':'region0',
                                           'nonrotatingpatches':[],
                                           'axis':[0,0,1],
                                           'origin':[0,0,0],
                                           'omega':0}],
                    
                    'thermotype':[dict,{'type':'hePsiThermo',
                                        'mixture':'pureMixture',
                                        'thermo':'hConst',
                                        'transport':'const',
                                        'equationOfState':'perfectGas',
                                        'specie':'specie',
                                        'energy':'sensibleInternalEnergy'}],
                    
                    # turbulenceProperties
                    'rasmodel':[str,'SpalartAllmaras'],
                    'rasmodelparameters':[dict,{'Cv2':'5.0',
                                                'lambdaErr':'1e-6',
                                                'maxLambdaIter':'10',
                                                'kMin':'1e-16',
                                                'omegaMin':'1e-16',
                                                'epsilonMin':'1e-16',
                                                'nuTildaMin':'1e-16',
                                                'decayControl':'no',
                                                'kInf':'1.0',
                                                'omegaInf':'1.0'}],

                    # actuator properties
                    'actuatoractive':[int,0],
                    'actuatoradjustthrust':[int,0],
                    'actuatorvolumenames':[list,[]],
                    'actuatorthrustcoeff':[list,[]],
                    'actuatorpoverd':[list,[]],
                    'actuatorrotationdir':[list,[]],
                    
                    # Adjoint Options
                    'adjointsolver':[str,'adjointSolverSimpleFoam'],
                    'usecoloring':[bool,True],
                    'flowcondition':[str,'Incompressible'],
                    'adjdvtypes':[list,['UIn']],
                    'epsderiv':[float,1e-5],
                    'epsderivffd':[float,1e-3],
                    'epsderivxv':[float,1e-7],
                    'epsderivuin':[float,1e-5],
                    'epsderivvis':[float,1e-8],
                    'adjjacmatordering':[str,'state'],
                    'adjjacmatreordering':[str,'rcm'],
                    'mintoljac':[float,1e-14],
                    'maxtoljac':[float,1e14],
                    'mintolpc':[float,1e-14],
                    'maxtolpc':[float,1e14],
                    'stateresettol':[float,1e-6],
                    'tractionbcmaxiter':[int,20],
                    'correctwalldist':[bool,True],
                    'reducerescon4jacmat':[bool,False],
                    'delturbprod4pcmat':[bool,False],
                    'calcpcmat':[bool,True],
                    'fastpcmat':[bool,False],
                    'writematrices':[bool,False],
                    'adjgmrescalceigen':[bool,False],
                    'adjgmresrestart':[int,200],
                    'adjglobalpciters':[int,0],
                    'adjlocalpciters':[int,1],
                    'adjasmoverlap':[int,1],
                    'adjpcfilllevel':[int,0],
                    'adjgmresmaxiters':[int,1000],
                    'adjgmresabstol':[float,1.0e-16],
                    'adjgmresreltol':[float,1.0e-6],
                    'normalizeresiduals':[list,['URes','pRes','p_rghRes','phiRes','TRes','nuTildaRes','kRes','omegaRes','epsilonRes','ReThetatRes','gammaIntRes','GRes','DRes']],
                    'normalizestates':[list,['U','p','p_rgh','phi','T','nuTilda','k','omega','epsilon','ReThetat','gammaInt','G','D']],
                    'maxresconlv4jacpcmat':[dict,{'URes':2,'pRes':2,'p_rghRes':2,'phiRes':1,'nuTildaRes':2,'kRes':2,'omegaRes':2,'epsilonRes':2,'ReThetatRes':2,'gammaIntRes':2,'TRes':2,'eRes':2,'GRes':2,'DRes':2}],
                    'statescaling':[dict,{}],
                    'residualscaling':[dict,{}],
                    'nffdpoints':[int,0],
                    
                    # Flow options
                    'setflowbcs':[bool,False],
                    'flowbcs':[dict,{}],
                    'inletpatches':[list,['inlet']],
                    'outletpatches':[list,['outlet']],
                    'derivuininfo':[dict,{'stateName':'U','component':0,'type':'fixedValue','patchNames':['inlet']}],
                    'userdefinedpatchinfo':[dict,{}],
                    'userdefinedvolumeinfo':[dict,{}],
                    'referencevalues':[dict,{'magURef':1.0,'rhoRef':1.0,'pRef':0.0,'ARef':1.0,'LRef':1.0}],
                    'usenksolver':[bool,False],
                    'nksegregatedturb':[bool,False],
                    'nksegregatedphi':[bool,False],
                    'nkewrtol0':[float,0.3],
                    'nkewrtolmax':[float,0.7],
                    'nkpclag':[int,1],
                    'nkgmresrestart':[int,500],
                    'nkjacmatreordering':[str,'rcm'],
                    'nkasmoverlap':[int,1],
                    'nkglobalpciters':[int,0],
                    'nklocalpciters':[int,1],
                    'nkpcfilllevel':[int,0],
                    'nkgmresmaxiters':[int,500],
                    'nkreltol':[float,1e-8],
                    'nkabstol':[float,1e-12],
                    'nkstol':[float,1e-8],
                    'nkmaxiters':[int,100],
                    'nkmaxfuncevals':[int,10000],

                    
                    # objectiveFunctionOptions
                    'objfuncgeoinfo':[list,[['wall'],['wall']]], # objfuncgeoinfo is a listlist
                    'objfuncs':[list,['CD','CL']],
                    'dragdir':[list,[1.0,0.0,0.0]],
                    'liftdir':[list,[0.0,0.0,1.0]],
                    'cofr':[list,[0.0,0.0,0.0]],
                    'rotrad':[list,[0.0,0.0,0.0]],

                    # Mesh quality
                    'meshmaxaspectratio':[float,1000.0],
                    'meshminfacearea':[float,-1.0],
                    'meshminvol':[float,1.0e-16],
                    'meshmaxnonortho':[float,70.0],
                    'meshmaxskewness':[float,4.0],
                    
                    # Surface definition parameters:
                    'meshsurfacefamily':[object, None],
                    'designsurfacefamily':[object, None],
                    'designsurfaces':[list,['body']],
                                        
                    # misc
                    'dummy':[str,'test'],
                    'objstdthres':[float,1.0e8],
                    'postprocessingdir':[str,'postProcessing'],
                    'filechecktime':[float,1.0],
                    'multipointopt':[bool,False],
                    'restartopt':[bool,False],
                    'avgobjfuncs':[bool,False],
                    'avgobjfuncsstart': [int,1000],
                    'mpispawnrun': [bool,True],
                    'runpotentialfoam':[bool,False],
                    'updatemesh':[bool,True],
                    'rerunopt2':[int,-9999],
                    'preservepatches':[list,[]],
                    'singleprocessorfacesets':[list,[]],
                    'decomposepardict':[dict,{'method':'scotch','simpleCoeffs':{'n':'(2 2 1)','delta':'0.001'}}]
        }


        return defOpts

    def _getImmutableOptions(self):
        """We define the list of options that *cannot* be changed after the
        object is created. pyDAFoam will raise an error if a user tries to
        change these. The strings for these options are placed in a set"""

        return ('meshSurfaceFamily', 'designSurfaceFamily')
    
        
    def setMultiPointFCIndex(self,index):
        self.multiPointFCIndex = index
        return

    def getSolverMeshIndices(self):
        '''
        Get the list of indices to pass to the mesh object for the
        volume mesh mapping
        '''
        # Setup External Warping
        nCoords = len(self.x0.flatten())
       
        nCoords = self.comm.allgather(nCoords)
        offset =0
        for i in xrange(self.comm.rank):
            offset+=nCoords[i]

        meshInd = np.arange(nCoords[self.comm.rank])+offset
        
        return meshInd

    def getPointSetName(self,apName):
        """
        Take the apName and return the mangled point set name.

        """
        return 'openFoamCoords'

    def setDesignVars(self,x):
        '''
        Set the internal design variables. 
        At the moment we don't have any internal DVs to set.
        ''' 
        pass
        
        return
    
    def addVariablesPyOpt(self,optProb):
        '''
        Add the internal variables to the opt problem.
        At the moment we don't have any internal DVs to set.
        '''
        
        return

    def __call__(self):
        '''
        Solve the flow
        '''
        
        if self.getOption('multipointopt') and self.multiPointFCIndex==None:
            raise Error('multipointopt is true while multiPointFCIndex is not set')
            
        # For restart runs, check the logs and data in the outputdirectory, if anything is 
        # missing, set skipFlowAndAdjointRuns to False 
        if self.skipFlowAndAdjointRuns:
            self._checkRestartLogs(logOpt=1) # check flow 
            self._checkRestartLogs(logOpt=3) # check checkMesh

        # Update the internal run options
        self.solveFrom = 'startTime'
        self.solveAdjoint = False

        # remove the old solution directory
        self._cleanSolution()

        # Update the current run control files
        self._writeAdjointDictFile()
        if self.getOption('flowcondition') == "Incompressible":
            self._writeTransportPropertiesFile()
        elif self.getOption('flowcondition') == "Compressible":
            self._writeThermophysicalPropertiesFile()
        self._writeTurbulencePropertiesFile()
        self._writeGFile()
        self._writeRadiationPropertiesFile()
        self._writeThermalPropertiesFile()
        self._writeMRFPropertiesFile()
        self._writeMechanicalPropertiesFile()
        self._writeControlDictFile()
        self._writeFvSchemesFile()
        self._writeFvSolutionFile()
        
        self.comm.Barrier()

        # update the CFD Coordinates
        
        # add point set and update the mesh based on the DV valurs
        self.ptSetName = self.getPointSetName('dummy')
        ptSetName = self.ptSetName
        if (self.DVGeo is not None) and (self.getOption('updatemesh')):
            if not ptSetName in self.DVGeo.points:
                coords0 = self.mapVector(self.coords0, self.allFamilies, 
                                         self.designFamilyGroup)
                
                self.DVGeo.addPointSet(coords0, self.ptSetName)
                self.pointsSet = True
            
            # set the surface coords
            if self.comm.rank == 0:
                print ('DVGeo PointSet UpToDate: '+str(self.DVGeo.pointSetUpToDate(ptSetName)))
            if not self.DVGeo.pointSetUpToDate(ptSetName):
                if self.comm.rank == 0:
                    print ('Updating DVGeo PointSet....')
                coords = self.DVGeo.update(ptSetName,config=None)
                self.setSurfaceCoordinates(coords, self.designFamilyGroup)
                if self.comm.rank == 0:
                    print ('DVGeo PointSet UpToDate: '+str(self.DVGeo.pointSetUpToDate(ptSetName)))
                
                # warp the mesh
                if self.comm.rank == 0:
                    print ('Warping the volume mesh....')
                self.mesh.warpMesh()

                # write the new volume coords to a file
                if self.comm.rank == 0:
                    print ('Writting the updated volume mesh....')
                newGrid = self.mesh.getSolverGrid()
                ofm._writeOpenFOAMVolumePoints(self.fileNames,newGrid)

        # remove the old post processing results if they exist
        self._cleanPostprocessingDir()

        if self.printMesh and self.mesh is not None:
            meshName = os.path.join(os.getcwd(),"caseMesh.dat")
            self.mesh.writeOFGridTecplot(meshName)
        # end


        # For restart runs, we can directly read checkMeshLog 
        # so no need to run checkMesh until self.skipFlowAndAdjointRuns==False
        if not self.skipFlowAndAdjointRuns: 
            # check the mesh quality
            self.runCheckMeshQuality()
            self._copyLogs(logOpt=3) # copy for checkMesh
        
        # check if mesh q failed
        outputDir = self.getOption('outputdirectory')
        if not self.getOption('multipointopt'):
            logFileName=os.path.join(outputDir,'checkMeshLog_%3.3d'%self.flowRunsCounter)    
        else:
            logFileName=os.path.join(outputDir,'checkMeshLog_FC%d_%3.3d'%(self.multiPointFCIndex,self.flowRunsCounter))    
   
        self.meshQualityFailure=self.checkMeshLog(logFileName)

        if self.meshQualityFailure==True:
            if self.comm.rank==0: 
                print("Checking Mesh Quality. Failed!")
        else:
            if self.comm.rank==0:
                print("Checking Mesh Quality. Passed!")
                
        if self.comm.rank==0:
            print('Calling Flow Solver %03d'%self.flowRunsCounter)   
             
        # For restart runs, we can directly read objFuncs.dat 
        # so no need to run OF solver or write logFiles until self.skipFlowAndAdjointRuns==False    
        if not self.skipFlowAndAdjointRuns:
        
            logFileName = 'flowLog' 
            # we don't need to solve OF if mesh quality fails
            if self.meshQualityFailure==False:
                # call the actual openfoam executable
                self._callOpenFoamSolver(logFileName)
                # check if we need to calculate the averaged obj funcs  
                if self.getOption('avgobjfuncs'):
                    self._calcAveragedObjFuncs()

            # copy flow to outputdirectory
            outputDir = self.getOption('outputdirectory')
            if self.getOption('writelinesearch'):
                # figure out the output file options
                caseName = self.getOption('casename')+'_flow_%3.3d'%self.flowRunsCounter
                # outputdirectory and casename to get full output path
                newDir = os.path.join(outputDir, caseName)
                if not self.getOption('multipointopt'):
                    origDir = './'
                else:
                    origDir = '../FlowConfig%d/'%self.multiPointFCIndex
                self._copyResults(origDir,newDir)

            self._copyLogs(logOpt=1) # copy for flow 

        self.flowRunsCounter+=1

        return 

    def solveADjoint(self):
        '''
        Solve the adjoint
        '''
        
        if self.getOption('multipointopt') and self.multiPointFCIndex==None:
            raise Error('multipointopt is true while multiPointFCIndex is not set')
            
        # For restart runs, check the logs and data in the outputdirectory, if anything is 
        # missing, set skipFlowAndAdjointRuns to False 
        if self.skipFlowAndAdjointRuns:
            self._checkRestartLogs(logOpt=2) # check adjoint 

        self.solveFrom = 'latestTime'
        self.solveAdjoint = True
        self.setOption('setflowbcs',False)

        divUAdj=self.getOption('divuadj')
        divSchemes=self.getOption('divschemes')
        if len(divUAdj)!=0: 
            divUFlow=divSchemes['div(phi,U)']
  
        # Update the current run control files
        self._writeAdjointDictFile()
        if self.getOption('flowcondition') == "Incompressible":
            self._writeTransportPropertiesFile()
        elif self.getOption('flowcondition') == "Compressible":
            self._writeThermophysicalPropertiesFile()
        self._writeTurbulencePropertiesFile()
        self._writeGFile()
        self._writeRadiationPropertiesFile()
        self._writeThermalPropertiesFile()
        self._writeMRFPropertiesFile()
        self._writeMechanicalPropertiesFile()
        self._writeControlDictFile()
        self._writeFvSchemesFile()
        self._writeFvSolutionFile()
        self.comm.Barrier()
        
        logFileName = 'adjointLog'

        if self.comm.rank==0:
            print('Calling Adjoint Solver %03d'%self.adjointRunsCounter)

        if self.getOption('rerunopt2')>0:
            if self.getOption('rerunopt2')>=self.adjointRunsCounter:
                # copy adjoint result to outputdirectory
                self.solveAdjoint = False
                self.setOption('setflowbcs',True)
                self._writeAdjointDictFile()
                self._callOpenFoamSolver(logFileName)
                caseName = self.getOption('casename')+'_adjoint_%3.3d'%self.adjointRunsCounter
                outputDir = self.getOption('outputdirectory')
                newDir = os.path.join(outputDir, caseName)
                origDir = './'
                self._copyResults(origDir,newDir)  
            else:
                exit()      

        # For restart runs, we can directly read objFuncsSens.dat 
        # so no need to run adjoint solver or write logFiles until self.skipFlowAndAdjointRuns==False
        if not self.skipFlowAndAdjointRuns:

            # write the current design variable values to files
            self._writeDVs()
        
            # write delta vol points
            adjDVTypes = self.getOption('adjdvtypes')
            if ("FFD" in adjDVTypes) and (self.getOption('updatemesh')):
                # only need to write deltaVol mat once for multipoint
                if self.multiPointFCIndex==None or self.multiPointFCIndex == 0:
                    self._writeDeltaVolPointMat()

            # if divuadj is set, we will use a different divu scheme for adjoint
            # so we need to first re-compute the flow using the 1st order scheme
            # before calling the adjoint
            if len(divUAdj)!=0: 
                if self.comm.rank==0:
                    print('divuadj is set. Re-computing the flow using div(U): %s'%divUAdj)
                # rerun the flow
                # clean previous solution
                self._cleanSolution()
                # setup the parameters
                divSchemes['div(phi,U)']=divUAdj
                self.setOption('divschemes',divSchemes)
                self.solveAdjoint = False
                self._writeAdjointDictFile()
                self._writeFvSchemesFile()
                self.comm.Barrier()
                # call the actual openfoam executable
                self._callOpenFoamSolver('flowLog')

                # now turn solveAdjoint on and run the adjoint solution
                self.solveAdjoint = True
                self._writeAdjointDictFile()
                self.comm.Barrier()
                # call the actual openfoam executable
                self._callOpenFoamSolver(logFileName)
                
                # resume the paramters
                if self.comm.rank==0:
                    print('Completed! Reset div(U) to %s'%divUFlow)
                divSchemes['div(phi,U)']=divUFlow
                self.setOption('divschemes',divSchemes)
                self._writeFvSchemesFile()
                self.comm.Barrier()
            else:
                # call the actual openfoam executable
                self._callOpenFoamSolver(logFileName)
    
            # copy adjoint result to outputdirectory
            outputDir = self.getOption('outputdirectory')
            if self.getOption('writesolution'):
                # figure out the output file options
                caseName = self.getOption('casename')+'_adjoint_%3.3d'%self.adjointRunsCounter
                # outputdirectory and casename to get full output path
                newDir = os.path.join(outputDir, caseName)
                if not self.getOption('multipointopt'):
                    origDir = './'
                else:
                    origDir = '../FlowConfig%d/'%self.multiPointFCIndex
                self._copyResults(origDir,newDir)
            
            self._copyLogs(logOpt=2) 
         
        self.adjointRunsCounter+=1

        return

    def runPotentialFoam(self):
                    
        logName='potentialFoamLog'
        
        mpiSpawnRun=self.getOption('mpispawnrun')
        
        if self.parallel and mpiSpawnRun:
        
            # Setup the indicator file to tell the python layer that potentialFoam
            # is finished running
            finishFile = 'potentialFoamFinished.txt' 

            # Remove the indicator file if it exists
            if os.path.isfile(finishFile):
                if self.comm.rank==0:
                    try:
                        os.remove(finishFile)
                    except:
                        raise Error('pyDAFoam: Unable to remove %s'%finishFile)
                        sys.exit(0)
                    # end
                # end
            # end
        
            # write the bash script to run the solver
            scriptName = 'runPotentialFoam.sh'
            
            self.writeParallelPotentialFoamScript(logName,scriptName)
            if self.comm.rank==0:
                print("Running potentialFoam.")
                
            # check if the run script exists
            while not os.path.exists(scriptName):
                time.sleep(1)
            self.comm.Barrier()
                
            # Run potentialFoam through a bash script
            child = self.comm.Spawn('sh', [scriptName],self.comm.size,
                                    MPI.INFO_NULL, root=0)
    
            # Now wait for potentialFoam to finish. The bash script will create
            # the empty finishFile when it completes
            counter = 0
            checkTime = self.getOption('filechecktime')
            while not os.path.exists(finishFile):
                time.sleep(checkTime)
                counter+=1
            # end
   
            # Wait for all of the processors to get here
            self.comm.Barrier()
            
        elif not self.parallel and mpiSpawnRun: 

            print("Running potentialFoam.")
            f=open(logName,'w')
            subprocess.call("potentialFoam",stdout=f,stderr=subprocess.STDOUT, shell=False)
            f.close()

        elif not mpiSpawnRun:
        
            finishFile='jobFinished'
            runFile = 'runPotentialFoam'
        
            if self.comm.rank==0:
                # Remove the log file if it exists
                if os.path.isfile(logName):
                    try:
                        os.remove(logName)
                    except:
                        raise Error('pyDAFoam: status %d: Unable to remove %s'%logName)
 
                # Remove the finish file
                if os.path.isfile(finishFile):
                    try:
                        os.remove(finishFile)
                    except:
                        raise Error('pyDAFoam: status %d: Unable to remove %s'%finishFile)
  
                # now create the run File
                fTouch=open(runFile,'w')
                fTouch.close()
                  
            self.comm.Barrier()
            
            if self.comm.rank==0:
                print("Running potentialFoam.")

            # check if the job finishes
            checkTime = self.getOption('filechecktime')
            while not os.path.isfile(finishFile):
                time.sleep(checkTime)
            # end
            self.comm.Barrier()  

        else:

            raise Error('Parallel and MPISpawn not valid!')

        return

    def runCheckMeshQuality(self):
                
        #self.meshQualityFailure=False
    
        logName='checkMeshLog'
        
        mpiSpawnRun=self.getOption('mpispawnrun')
        
        if self.parallel and mpiSpawnRun:
        
            # Setup the indicator file to tell the python layer that checkMesh
            # is finished running
            finishFile = 'checkMeshFinished.txt' 

            # Remove the indicator file if it exists
            if os.path.isfile(finishFile):
                if self.comm.rank==0:
                    try:
                        os.remove(finishFile)
                    except:
                        raise Error('pyDAFoam: Unable to remove %s'%finishFile)
                        sys.exit(0)
                    # end
                # end
            # end
        
            # write the bash script to run the solver
            scriptName = 'runCheckMesh.sh'
            
            self.writeParallelCheckMeshScript(logName,scriptName)
            if self.comm.rank==0:
                print("Checking Mesh Quality.")
                
            # check if the run script exists
            while not os.path.exists(scriptName):
                time.sleep(1)
            self.comm.Barrier()
                
            # Run checkMesh through a bash script
            child = self.comm.Spawn('sh', [scriptName],self.comm.size,
                                    MPI.INFO_NULL, root=0)
    
            # Now wait for checkMesh to finish. The bash script will create
            # the empty finishFile when it completes
            counter = 0
            checkTime = self.getOption('filechecktime')
            while not os.path.exists(finishFile):
                time.sleep(checkTime)
                counter+=1
            # end
   
            # Wait for all of the processors to get here
            self.comm.Barrier()
            
        elif not self.parallel and mpiSpawnRun:

            f=open(logName,'w')
            subprocess.call("checkMesh",stdout=f,stderr=subprocess.STDOUT, shell=False)
            f.close()

        elif not mpiSpawnRun:
        
            finishFile='jobFinished'
            runFile = 'runCheckMesh'
        
            if self.comm.rank==0:
                # Remove the log file if it exists
                if os.path.isfile(logName):
                    try:
                        os.remove(logName)
                    except:
                        raise Error('pyDAFoam: status %d: Unable to remove %s'%logName)
 
                # Remove the finish file
                if os.path.isfile(finishFile):
                    try:
                        os.remove(finishFile)
                    except:
                        raise Error('pyDAFoam: status %d: Unable to remove %s'%finishFile)
  
                # now create the run File
                fTouch=open(runFile,'w')
                fTouch.close()
                  
            self.comm.Barrier()
            
            if self.comm.rank==0:
                print("Checking Mesh Quality.")

            # check if the job finishes
            checkTime = self.getOption('filechecktime')
            while not os.path.isfile(finishFile):
                time.sleep(checkTime)
            # end
            self.comm.Barrier()
            
        else:

            raise Error('Parallel and MPISpawn not valid!')

        return
    
    
    def checkMeshLog(self,logName):
        
        f=open(logName,'r')
        lines=f.readlines()
        f.close()
        
        meshminfacearea = -1.0
        meshmaxaspectratio = 1.0e6
        meshminvol = -1.0
        meshmaxnonortho = 1.0e6
        meshmaxskewness = 1.0e6
        
        val = '[-+]?[0-9]*\.?[0-9]*[e]?[-+]?[0-9]*'
        reMinSurfaceArea = re.compile(r'.*Minimum\s*face\s*area\s*[:=]?\s*(%s)\.*'%val)
        reAspectRatio = re.compile(r'.*Max\s*aspect\s*ratio\s*[:=]?\s*(%s).*'%val)
        reMinVol = re.compile(r'.*Min[imum]*?\s*[negative]*?\s*volume\s*[:=]?\s*(%s).*'%val)
        reMaxNonOrth = re.compile(r'.*Mesh\snon-orthogonality\sMax:\s(%s).*'%val)
        reMaxSkewness = re.compile(r'.*Max\sskewness\s=\s(%s).*'%val)
        
        meshOK = False
        facePyramidsOK = False
        for line in lines:
        
            res1 = reMinSurfaceArea.match(line)
            res2 = reAspectRatio.match(line)
            res3 = reMinVol.match(line)
            res4 = reMaxNonOrth.match(line)
            res5 = reMaxSkewness.match(line)
            
            if 'Face pyramids OK.' in line:
                facePyramidsOK = True

            if res1:
                meshminfacearea = float(res1.group(1))
                #print meshminfacearea
            if res2:
                meshmaxaspectratio = float(res2.group(1))
                #print meshmaxaspectratio
            if res3:
                meshminvol = float(res3.group(1))
                #print meshminvol
            if res4:
                meshmaxnonortho = float(res4.group(1))
                #print meshmaxnonortho
            if res5:
                meshmaxskewness = float(res5.group(1))
                #print meshmaxskewness
                
            if 'Mesh OK' in line:
                meshOK = True
        
        if self.comm.rank == 0 and meshOK == False:
            print ("************* Warning! Mesh Quality Not Perfect! ************")
        
        if ((meshmaxaspectratio >= self.getOption('meshmaxaspectratio')) or
            (meshminfacearea <= self.getOption('meshminfacearea'))  or
            (meshminvol <= self.getOption('meshminvol'))  or
            (meshmaxnonortho >= self.getOption('meshmaxnonortho'))  or
            (meshmaxskewness >= self.getOption('meshmaxskewness')) or 
            (facePyramidsOK == False) ) :
            return True
        else:
            return False
         
    
    def _callOpenFoamSolver(self,logFileName):
        '''
        Call the OpenFOAM executables.
        
        Parameters
        ----------   
        
        logFileName : str
            Name of the log file
        '''

        # we dont need to solve openfoam if mesh quality does not pass
        if self.meshQualityFailure:
            return
        
        # for multipoint optimization
        if self.getOption('multipointopt'):
        
            if self.multiPointFCIndex==None:
                raise Error('multiPointFCIndex not set!')
            
            finishFile='jobFinished_FC%d'%self.multiPointFCIndex
            
            if self.comm.rank==0:
                # create the run files
                if 'flowLog' in logFileName:
                    fTouch=open('runFlowSolver_FC%d'%self.multiPointFCIndex,'w')
                    fTouch.close()
                elif 'adjointLog' in logFileName:
                    fTouch=open('runAdjointSolver_FC%d'%self.multiPointFCIndex,'w')
                    fTouch.close()
                else:
                    raise Error('pyDAFoam: logFileName error!')
                        
                print("Simulation Started. Check the %s file for the progress."%logFileName)
                
            self.comm.Barrier()
            
            # check if the job finishes
            checkTime = self.getOption('filechecktime')
            while not os.path.isfile(finishFile):
                time.sleep(checkTime)
            
            self.comm.Barrier()
            
            if self.comm.rank==0:    
                # remove finish file        
                if os.path.isfile(finishFile):
                    try:
                        os.remove(finishFile)
                    except:
                        raise Error('pyDAFoam: status %d: Unable to remove %s'%finishFile)
     
            self.comm.Barrier()
            
            if self.comm.rank==0:
                print("Simulation Finished!")
                
            return
        
        # if it is not multipoint optimization, do the following    
        mpiSpawnRun = self.getOption('mpispawnrun')
        
        if self.parallel and mpiSpawnRun:
            # Setup the indicator file to tell the python layer that simpleFoam 
            # is finished running
            finishFile = 'foamFinished.txt' 

            # Remove the indicator file if it exists
            if os.path.isfile(finishFile):
                if self.comm.rank==0:
                    try:
                        os.remove(finishFile)
                    except:
                        raise Error('pyDAFoam: Unable to remove %s'%finishFile)
                        sys.exit(0)                    
                    # end
                # end
            # end
            self.comm.Barrier()

            # write the bash script to run the solver
            scriptName = 'runAdjointSolver.sh'
            # we could now set the solver as an option!!!
            self.writeParallelRunScript(logFileName,scriptName)

            if self.comm.rank==0:
                print("Simulation Started. Check the %s file for the progress."%logFileName)
            self.comm.Barrier()
            
            # check if the run script exists
            while not os.path.exists(scriptName):
                time.sleep(1)
            self.comm.Barrier()

            # Run simplefoam through a bash script
            child = self.comm.Spawn('sh', [scriptName],self.comm.size,
                                    MPI.INFO_NULL, root=0)
            
            # Now wait for simpleFoam to finish. The bash script will create
            # the empty finishFile when it completes
            counter = 0
            checkTime = self.getOption('filechecktime')
            while not os.path.exists(finishFile):
                time.sleep(checkTime)
                counter+=1
            # end
         
            # Wait for all of the processors to get here
            self.comm.Barrier()

            if self.comm.rank==0:
                print("Simulation Finished!")

        elif not self.parallel and mpiSpawnRun:

            f = open(logFileName,'w')
            adjointSolver = self.getOption("adjointsolver")
            status = subprocess.call(adjointSolver,stdout=f,stderr=subprocess.STDOUT, shell=False)
            f.close()
            if status != 0:
                raise Error('pyDAFoam: status %d: Unable to run %s'%(status,adjointSolver))

        elif not mpiSpawnRun:
            
            finishFile='jobFinished'
            
            if self.comm.rank==0:
                # Remove the log file if it exists
                if os.path.isfile(logFileName):
                    try:
                        os.remove(logFileName)
                    except:
                        raise Error('pyDAFoam: status %d: Unable to remove %s'%logFileName)

                # remove finish file
                if os.path.isfile(finishFile):
                    try:
                        os.remove(finishFile)
                    except:
                        raise Error('pyDAFoam: status %d: Unable to remove %s'%finishFile)
                        
                # create the run files
                if 'flowLog' in logFileName:
                    fTouch=open('runFlowSolver','w')
                    fTouch.close()
                elif 'adjointLog' in logFileName:
                    fTouch=open('runAdjointSolver','w')
                    fTouch.close()
                else:
                    raise Error('pyDAFoam: logFileName error!')          
                    
            self.comm.Barrier()
            
            if self.comm.rank==0:
                print("Simulation Started. Check the %s file for the progress."%logFileName)
                        
            # check if the job finishes
            checkTime = self.getOption('filechecktime')
            while not os.path.isfile(finishFile):
                time.sleep(checkTime)
            self.comm.Barrier()
            if self.comm.rank==0:
                print("Simulation Finished!")

        else:
            
            raise Error('Parallel and MPISpawn not valid!')
        # end
        
        return

    def miscSetup(self):

        # first check if mesh files are properly compressed. This can happen if writecompress is on, but the mesh files
        # have not been compressed. In this case, we will manually compress all the mesh files (except for boundary),
        # and delete the uncompressed ones.
        ofm.checkMeshCompression()

        # we need to decompose the domain if running in parallel
        if self.comm.rank == 0:
            print("Checking if we need to decompose the domain")
        
        if self.parallel:
            if self.comm.rank == 0:
                for i in range(self.comm.size):
                    # if any processor* is missing, we need to do the decomposePar
                    if not os.path.exists('processor%d'%i): 
                        # delete all processor folders
                        for j in range(self.comm.size):
                            if os.path.exists('processor%d'%j):
                                try:
                                    shutil.rmtree("processor%d"%j)
                                except:
                                    raise Error('pyDAFoam: Unable to remove processor%d directory'%j)
                        # decompose domains
                        self._writeDecomposeParDictFile()
                        f = open('decomposeParLog','w')
                        status = subprocess.call("decomposePar",stdout=f,stderr=subprocess.STDOUT,shell=False)
                        f.close()
                        if status != 0:
                            raise Error('pyDAFoam: status %d: Unable to run decomposePar'%status)
                        break   
        self.comm.Barrier()
    
        # run potentialFoam to generate a nice initial field. For y+=1 mesh, not running potentialFoam will 
        # have convergence issue
        if self.getOption('runpotentialfoam'):
            self.runPotentialFoam()
    
        # we need to check if points_orig exists, if not, we need to copy it from points
        # this only needs to be done once
        if self.comm.rank == 0:
            print("Checking if we need to copy points to points_orig")

        if self.getOption('writecompress')=='on':
            pointName='points.gz'
            pointNameOrig='points_orig.gz'
        elif self.getOption('writecompress')=='off':
            pointName='points'
            pointNameOrig='points_orig'
        else:
            raise Error('writecompress not valid')
            
        if self.parallel:
            if self.comm.rank == 0:
                for i in range(self.comm.size):
                    if not os.path.exists('processor%d/constant/polyMesh/%s'%(i,pointNameOrig)) :
                        pointsOrig='processor%d/constant/polyMesh/%s'%(i,pointName)
                        pointsCopy='processor%d/constant/polyMesh/%s'%(i,pointNameOrig)
                        try:
                            shutil.copyfile(pointsOrig,pointsCopy)
                        except:
                            raise Error('pyDAFoam: Unable to copy %s to %s.'%(pointsOrig,pointsCopy))
                            sys.exit(0)
        else:
            if not os.path.exists('constant/polyMesh/%s'%pointNameOrig) :
                pointsOrig='constant/polyMesh/%s'%pointName
                pointsCopy='constant/polyMesh/%s'%pointNameOrig
                try:
                    shutil.copyfile(pointsOrig,pointsCopy)
                except:
                    raise Error('pyDAFoam: Unable to copy %s to %s.'%(pointsOrig,pointsCopy))
                    sys.exit(0)
        # wait for rank 0 
        self.comm.Barrier()

        return


    def _coloringComputationRequired(self):
        '''
        check whether any of the required colorings are missing, if so
        recompute.
        '''
        missingColorings = False
        
        if( self.getOption('usecoloring')):
            # We need colorings, check if they exist
            requiredColorings=[]

            requiredColorings.append('dRdWColoring_%d.bin'%self.nProcs)
            for objFunc in self.getOption('objFuncs'):
                requiredColorings.append('dFdWColoring_%s_%d.bin'%(objFunc,self.nProcs))

            if 'Xv' in self.getOption('adjdvtypes'):
                requiredColorings.append('dRdXvColoring_%d.bin'%self.nProcs)
                for objFunc in self.getOption('objFuncs'):
                    requiredColorings.append('dFdXvColoring_%s_%d.bin'%(objFunc,self.nProcs))
            
            #now check for the require coloring
            for coloring in requiredColorings:
                if(not os.path.exists(coloring)):
                    missingColorings = True
                    break;

        return missingColorings

    def computeAdjointColoring(self):
        '''
        Run the routines to compute the coloring for the various adjoint 
        partial derivatives.

        This only needs to be run once per case.
        '''

        # Update the internal run options
        self.solveFrom = 'startTime'
        self.solveAdjoint = True
        
        logFileName='coloringLog'
        
        mpiSpawnRun=self.getOption('mpispawnrun')

        # check if a coloring computation is necessary. if not return without
        # doing anything
        if(not self._coloringComputationRequired()):
            return
            
        if self.rank==0:
            print(' Colorings not found, running coloringSolver%s, check the coloringLog file for the progress.'%self.getOption('flowcondition'))

        # Update the current adjoint options
        self._writeAdjointDictFile()
        if self.getOption('flowcondition') == "Incompressible":
            self._writeTransportPropertiesFile()
        elif self.getOption('flowcondition') == "Compressible":
            self._writeThermophysicalPropertiesFile()
        self._writeTurbulencePropertiesFile()
        self._writeGFile()
        self._writeRadiationPropertiesFile()
        self._writeThermalPropertiesFile()
        self._writeMRFPropertiesFile()
        self._writeMechanicalPropertiesFile()
        self._writeControlDictFile()
        self._writeFvSchemesFile()
        self._writeFvSolutionFile()
        self.comm.Barrier()

        if  self.parallel and mpiSpawnRun:
            # Setup the indicator file to tell the python layer that coloring 
            # is finished running
            finishFile = 'coloringFinished.txt' 

            # Remove the indicator file if it exists
            if self.comm.rank==0:
                if os.path.isfile(finishFile):
                    try:
                        os.remove(finishFile)
                    except:
                        raise Error('pyDAFoam: Unable to remove %s'%finishFile)  
            self.comm.Barrier()
        
            # write the bash script to run the coloring
            scriptName = 'runColoring.sh'
            self.writeParallelColoringScript('coloringLog',scriptName)
            
            # check if the run script exists
            while not os.path.exists(scriptName):
                time.sleep(1)
            self.comm.Barrier()

            # Run coloring through a bash script
            child = self.comm.Spawn('sh', [scriptName],self.comm.size,
                                    MPI.INFO_NULL, root=0)
                                    
            # Now wait for coloring to finish. The bash script will create
            # the empty finishFile when it completes
            checkTime = self.getOption('filechecktime')
            while not os.path.exists(finishFile):
                time.sleep(checkTime)
            # end

            # Wait for all of the processors to get here
            self.comm.Barrier()
        
        elif not self.parallel and mpiSpawnRun:

            f = open('coloringLog','w')
            status = subprocess.call('coloringSolver%s'%self.getOption('flowcondition'),stdout=f,stderr=subprocess.STDOUT, shell=False)
            f.close()
            if status != 0:
                raise Error('pyDAFoam: status %d: Unable to run coloring'%status)
            
        elif not mpiSpawnRun:
        
            runFile  = 'runColoring'
            finishFile = 'jobFinished'
            
            if self.comm.rank==0:
                # Remove the finish file if it exists
                if os.path.isfile(finishFile):
                    try:
                        os.remove(finishFile)
                    except:
                        raise Error('pyDAFoam: Unable to remove %s'%finishFile)
            
                # Remove the log file if it exists
                if os.path.isfile(logFileName):
                    try:
                        os.remove(logFileName)
                    except:
                        raise Error('pyDAFoam: Unable to remove %s'%logFileName)
            
                # touch the run file
                fTouch=open(runFile,'w')
                fTouch.close()
                
            self.comm.Barrier()
                                    
            # check if the job finishes
            checkTime = self.getOption('filechecktime')
            while not os.path.isfile(finishFile):
                time.sleep(checkTime)
            self.comm.Barrier()
            
        else:
            
            raise Error('Parallel and MPISpawn not valid!')

        return

    def evalFunctions(self, funcs, evalFuncs=None, ignoreMissing=False):
        """
        Evaluate the desired functions given in iterable object,
        'evalFuncs' and add them to the dictionary 'funcs'. The keys
        in the funcs dictioary will be have an _<ap.name> appended to
        them. Additionally, information regarding whether or not the
        last analysis with the aeroProblem was sucessful is
        included. This information is included as "funcs['fail']". If
        the 'fail' entry already exits in the dictionary the following
        operation is performed:

        funcs['fail'] = funcs['fail'] or <did this problem fail>

        In other words, if any one problem fails, the funcs['fail']
        entry will be False. This information can then be used
        directly in the pyOptSparse. 

        Parameters
        ----------
        funcs : dict
            Dictionary into which the functions are saved.

        evalFuncs : iterable object containing strings
          If not None, use these functions to evaluate.

        ignoreMissing : bool
            Flag to supress checking for a valid function. Please use
            this option with caution.

        Examples
        --------
        >>> funcs = {}
        >>> CFDsolver()
        >>> CFDsolver.evalFunctions(funcs, ['CD', 'CL'])
        >>> funcs
        >>> # Result will look like:
        >>> # {'CD':0.501, 'CL':0.02750}
        """

        res = self._getSolution()
        
        if self.comm.rank == 0:
            print('keys',res.keys())
        if evalFuncs is None:
            print 'evalFuncs not set, exiting...'
            sys.exit(0)

        # Just call the regular _getSolution() command and extract the
        # ones we need:
        
        
        for f in evalFuncs:
            fname = self.possibleObjectives[f]
            if f in res.keys():
                key = f
                funcs[key] = res[f]
            elif fname in res.keys():
                key = fname
                funcs[key] = res[fname]
            else:
                if not ignoreMissing:                    
                    raise Error('Reqested function: %s is not known to pyDAFoam.'%f)
                # end
            # end
        # end
             
        # we need to call checkMeshQuality and the flow convergence before evalFunction
        # so that we can tell if the flow solver fails
        if self.meshQualityFailure or self._checkFlowFailure(evalFuncs):
            funcs['fail'] = True
        else:
            funcs['fail'] = False

        return

    def _checkSolutionFailure(self, aeroProblem, funcs):
        """
        Take in a an aeroProblem and check for failure. Then append the fail
        flag in funcs.
        NOTE: this function is deprecated
    
        Parameters
        ----------
        aeroProblem : pyAero_problem class
            The aerodynamic problem to to get the solution for

        funcs : dict
            Dictionary into which the functions are saved.
        """
        #self.setAeroProblem(aeroProblem)
        # We also add the fail flag into the funcs dictionary. If fail
        # is already there, we just logically 'or' what was
        # there. Otherwise we add a new entry. 
        failFlag = False#self.curAP.solveFailed or self.curAP.fatalFail
        if 'fail' in funcs:
            funcs['fail'] = funcs['fail'] or failFlag
        else:
            funcs['fail'] = failFlag
    
    def _calcAveragedObjFuncs(self):
        '''
        Read the flowLog file and calculate the averaged value for the specified obj function.
        The averaging starts from "avgStart"
        '''
        
        avgStart = self.getOption('avgobjfuncsstart')
 
        maxiter = self.getOption('maxflowiters')
        if maxiter <= avgStart:
            raise Error("avgstart larger than maxflowiters!")
        
        if self.comm.rank==0:
        
            # get evalFuncs
            evalFuncs = self.getOption('objfuncs')

            if not self.getOption('multipointopt'):
                logFileName = 'flowLog'
            else:
                logFileName = '../FlowConfig%d/flowLog'%self.multiPointFCIndex
            
            if not os.path.isfile(logFileName):
                raise Error("%s file not found!"%logFileName)
                
            f=open(logFileName,'r')
            lines=f.readlines()
            f.close()
            
            objFuncPrintName ={}
            for key in self.possibleObjectives.keys():
                objFuncPrintName[key]= self.possibleObjectives[key]+":"
            
            if not self.getOption('multipointopt'):
                avgObjFileName = 'objFuncsMean.dat'
            else:
                avgObjFileName = '../FlowConfig%d/objFuncsMean.dat'%self.multiPointFCIndex
                                
            f=open(avgObjFileName,'w')
            for funcName in evalFuncs:
            
                objFunc=[]
                for line in lines:
                    if objFuncPrintName[funcName] in line:
                        col = line.split()
                        objFunc.append(float(col[1]))
                stepTotal = len(objFunc)
    
                objFuncMean = 0.0
                nCount = 0
                for i in range(stepTotal):
                    if i >= avgStart:
                        objFuncMean += objFunc[i]
                        nCount += 1
                objFuncMean /= nCount
                
                f.write('%s %20.16f\n'%(funcName,objFuncMean))

            f.close()
        
        self.comm.Barrier()
         
        return
         
    def _checkFlowFailure(self,evalFuncs):
        '''
        Read the flowLog file and check if flow converges well. This is done by calculating
        the standard deviation of the objective function. If it is above a threshold, return true
        
        Parameters
        ----------
        evalFuncs : list
            List containing the names of the objective functions
        '''
        
        returnFlag = False
        
        outputDir = self.getOption('outputdirectory')
        if not self.getOption('multipointopt'):
            logFileName = os.path.join(outputDir,'flowLog_%3.3d'%(self.flowRunsCounter-1))    
        else:
            logFileName = os.path.join(outputDir,'flowLog_FC%d_%3.3d'%(self.multiPointFCIndex,(self.flowRunsCounter-1)))   
        
        if not os.path.isfile(logFileName):
            return True
            
        f=open(logFileName,'r')
        lines=f.readlines()
        f.close()
        
        objFuncPrintName ={}
        for key in self.possibleObjectives.keys():
            objFuncPrintName[key]= self.possibleObjectives[key]+":"
        
        for funcName in evalFuncs:
        
            objFunc=[]
            step = 0
            for line in lines:
                if objFuncPrintName[funcName] in line:
                    col = line.split()
                    objFunc.append(col[1])
                    step +=1
            
            stepStart = step*0.8 # average using the last 80% of the time steps
            
            objFuncMean = 0.0
            nCount = 0
            for i in range(len(objFunc)):
                if i >= stepStart:
                    objFuncMean += float(objFunc[i])
                    nCount += 1
            objFuncMean /= nCount
            
            objFuncStd = 0.0
            nCount = 0
            for i in range(len(objFunc)):
                if i >= stepStart:
                    objFuncStd += (float(objFunc[i])-float(objFuncMean))**2.0
                    nCount += 1
            objFuncStd /= nCount
            objFuncStd = objFuncStd**0.5
            
            if self.comm.rank==0:
                print(funcName+" std: "+str(objFuncStd))
           
            if objFuncStd > self.getOption('objstdthres'):
                returnFlag = True 
        
        return returnFlag
            
    def _checkAdjointFailure(self):
        """
        Read the adjointLog file and check if adjoint converges
        """
        
        outputDir = self.getOption('outputdirectory')
        if not self.getOption('multipointopt'):
            logFileName = os.path.join(outputDir, 'adjointLog_%3.3d'%(self.adjointRunsCounter-1)) 
        else:
            logFileName = os.path.join(outputDir, 'adjointLog_FC%d_%3.3d'%(self.multiPointFCIndex,(self.adjointRunsCounter-1))) 
            
        if not os.path.isfile(logFileName):
            if self.comm.rank==0:
                print("adjointLog not found")
            return True

        f=open(logFileName,'r')
        lines=f.readlines()

        adjointResDict={}
        adjointIterDict={}

        parsing = False
        for line in lines:
        
            if 'Solving Adjoint' in line:
                parsing = True
                columns = line.split() 
                cfKey=columns[2]
                adjointRes=np.empty([0])
                adjointIter=np.empty([0])
                
            if 'Total iterations' in line:
                parsing = False
                adjointResDict.update({cfKey:adjointRes})
                adjointIterDict.update({cfKey:adjointIter})
                
            if parsing  == True:
                if 'Main iteration' in line:
                    line = line.strip()  # get rid of \n
                    columns = line.split() # split all the columns in one line
                    adjointRes=np.append(adjointRes,columns[6])
                    adjointIter=np.append(adjointIter,columns[2])
            
        f.close()
        
        if not adjointResDict: # if adjointResDict is empty return True
            return True

        relResTarget=self.getOption('adjgmresreltol')*100.0 # here we loose the criterion a bit
        absResTarget=self.getOption('adjgmresabstol')*100.0 
        
        returnFlag=False
        for key in adjointResDict:
            resRatio=float(adjointResDict[key][-1]) / float(adjointResDict[key][0])
            absRes=float(adjointResDict[key][-1])
            if resRatio>relResTarget and absRes>absResTarget:
                if self.comm.rank==0:
                    print("adjoint convergence tolerance not satisfied")
                returnFlag=True
                break
        
        return returnFlag
        

    def evalFunctionsSens(self, funcsSens, evalFuncs=None):
        """
        Evaluate the sensitivity of the desired functions given in
        iterable object,'evalFuncs' and add them to the dictionary
        'funcSens'.

        Parameters
        ----------
        funcSens : dict
        Dictionary into which the function derivatives are saved.

        evalFuncs : iterable object containing strings
            The functions the user wants the derivatives of


        Examples
        --------
        >>> funcSens = {}
        >>> CFDsolver.evalFunctionsSens(funcSens, ['CD', 'CL'])
        """
        
        # Check for evalFuncs, this should eventually have some kind of default
        if evalFuncs is None:
            print 'evalFuncs not set, exiting...'
            sys.exit(0)

        adjDVTypes = self.getOption('adjdvtypes')
            
        # Do the functions one at a time:
        for f in evalFuncs:
            if self.comm.rank == 0:
                print('Reading adjoint: %s'%f)

            key ='%s'%f

            # Set dict structure for this derivative
            funcsSens[key] = {}

            if "Xv" in adjDVTypes:
                ptSetName = self.ptSetName

                # Geometric derivatives
                dIdXv = self._readXvSensitivity(f)

                if self.DVGeo is not None and self.DVGeo.getNDV() > 0:
                    # Now get total derivative wrt surface cordinates
                    self.mesh.warpDeriv(dIdXv[:])
                    dIdXs = self.mesh.getdXs()#self.groupName)# should this be groupname?.
                    dIdXs = self.mapVector(dIdXs, self.meshFamilyGroup, 
                                           self.designFamilyGroup)
                    dIdx = self.DVGeo.totalSensitivity(dIdXs, ptSetName=ptSetName, comm=self.comm)
                else:
                    dIdx = {'vCoords':dIdXv}

                # add to dict
                funcsSens[key].update(dIdx)
                
            if "FFD" in adjDVTypes:
                # FFD derivatives
                dIdx = self._readFFDSensitivity(f)
                # add to dict
                funcsSens[key].update(dIdx)
                
            if "UIn" in adjDVTypes:
                funcsSens[key]['UIn'] = self._readUInSensitivity(f)

            if "Vis" in adjDVTypes:
                funcsSens[key]['Vis'] = self._readVisSensitivity(f)
                
        # we need to check if adjoint converges, if not, return fail=true
        funcsSens['fail'] = self._checkAdjointFailure()

       
        return
        
    def updateVolumePoints(self):
        '''
        Update the vol mesh point coordinates based on the current values of design variables
        '''

        # update the CFD Coordinates
        self.ptSetName = self.getPointSetName('dummy')
        ptSetName = self.ptSetName
        if (self.DVGeo is not None) and (self.getOption('updatemesh')):
            if not ptSetName in self.DVGeo.points:
                coords0 = self.mapVector(self.coords0, self.allFamilies, self.designFamilyGroup)
                self.DVGeo.addPointSet(coords0, self.ptSetName)
                self.pointsSet = True
                
            # set the surface coords
            #if self.comm.rank == 0:
            #    print ('DVGeo PointSet UpToDate: '+str(self.DVGeo.pointSetUpToDate(ptSetName)))
            if not self.DVGeo.pointSetUpToDate(ptSetName):
                #if self.comm.rank == 0:
                #    print 'Updating DVGeo PointSet....'
                coords = self.DVGeo.update(ptSetName,config=None)
                self.setSurfaceCoordinates(coords, self.designFamilyGroup)
                #if self.comm.rank == 0:
                #    print ('DVGeo PointSet UpToDate: '+str(self.DVGeo.pointSetUpToDate(ptSetName)))
        
            # warp the mesh
            self.mesh.warpMesh()

        return
        
    def writeUpdatedVolumePoints(self):
        '''
        write the vol mesh point coordinates based on the current values of design variables
        '''

        # update the CFD Coordinates
        self.updateVolumePoints()
        # write the new volume coords to a file
        newGrid = self.mesh.getSolverGrid()
        #print newGrid
        ofm._writeOpenFOAMVolumePoints(self.fileNames,newGrid)

        return

    def updateDVSurfacePoints(self):
        '''
        Update and write the surface points based on the current values of design variables
        NOTE: this function is depricated since it does not work in parallel. Use writeUpdatedVolumePoints instead
        '''
               
        # update the CFD Coordinates
        self.ptSetName = self.getPointSetName('dummy')#self.curAP.name)
        ptSetName = self.ptSetName
        if self.DVGeo is not None:
            if not ptSetName in self.DVGeo.points:
                self.DVGeo.addPointSet(self.coords0, self.ptSetName)
                self.pointsSet = True
            
            # set the surface coords
            if self.comm.rank == 0:
                print ('DVGeo PointSet UpToDate: '+str(self.DVGeo.pointSetUpToDate(ptSetName)))
            if not self.DVGeo.pointSetUpToDate(ptSetName):
                if self.comm.rank == 0:
                    print 'Updating DVGeo PointSet....'
                coords = self.DVGeo.update(ptSetName,config=None)
                #self.setSurfaceCoordinates(coords, self.designFamilyGroup)
                if self.comm.rank == 0:
                    print ('DVGeo PointSet UpToDate: '+str(self.DVGeo.pointSetUpToDate(ptSetName)))
        
        ############### write points
        if self.getOption('writecompress'):
            fPoints = gzip.open('constant/polyMesh/points.gz' 'wb')
        else:
            fPoints = open('constant/polyMesh/points', 'w')
        # write the file header
        fPoints.write('/*--------------------------------*- C++ -*---------------------------------*\ \n')
        fPoints.write('| ========                 |                                                 | \n')
        fPoints.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n')
        fPoints.write('|  \\    /   O peration     | Version:  v1812                                 | \n')
        fPoints.write('|   \\  /    A nd           | Web:      www.OpenFOAM.com                      | \n')
        fPoints.write('|    \\/     M anipulation  |                                                 | \n')
        fPoints.write('\*--------------------------------------------------------------------------*/ \n')
        fPoints.write('FoamFile\n')
        fPoints.write('{\n')
        fPoints.write('    version     2.0;\n')
        fPoints.write('    format      ascii;\n')
        fPoints.write('    class       dictionary;\n')
        fPoints.write('    location    "constant/polyMesh";\n')
        fPoints.write('    object      points;\n')
        fPoints.write('}\n')
        fPoints.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        fPoints.write('\n')
        
        fPoints.write('%d\n'%len(coords[:,0]))
        fPoints.write('(\n')
        
        for i in range(len(coords[:,0])):
            fPoints.write('(%f %f %f)\n'%(coords[i,0],coords[i,1],coords[i,2]))
        fPoints.write(')\n')
        
        fPoints.close()

        return 
        
    def _writeDeltaVolPointMatNew(self,deltaVPointThreshold=1.0E-16):
        '''
        Perturb each design variable (xDvDot) and save the delta volume point coordinates (xVDot)
        to a mat, this will be used to calculate dRdFFD and dJdFFD in the OpenFOAM layer.
        Specifically, it computes the following:
        Given a xDvDot, calculate xVDot:
            xSDot = \\frac{dX_{S}}{dX_{DV}}\\xDvDot
            xVDot = \\frac{dX_{V}}{dX_{S}}\\xSDot
            
        ***************************    
        It is a known issue that the warpDerivFwd function only "works" for very large perturbation 
        (e.g., 1e6) for a given xSDot. So to make it works for our purpose, we need to manually scale
        the user-input epsFFD value up to 1e6 and then scale the result (xVDot) back for the final output 
        NOTE: This function does NOT work very well at this moment, need to figure out a way to use 
        small perturbation
        ***************************
        
        
        Parameters
        ----------
        deltaVPointThreshold: float
            A threshold, any delta volume coordinates (xVDot) smaller than this value will be ignored
            
        '''
        
        if self.DVGeo is None:
            raise Error("DVGeo not set!")
            exit(1)
        
        # Get the FFD size
        nDVs= self.getOption('nffdpoints')
        getNDVs=0
        for key in self.DVGeo.getValues().keys():
            getNDVs += len( self.DVGeo.getValues()[key] )
        #getNDVs = self.DVGeo.getNDV()
        if not nDVs == getNDVs:
            raise Error("nffdpoints="+str(nDVs)+" while nDVs="+str(getNDVs))
            exit(1)

        if self.comm.rank == 0:
            print "Delete Old DeltaVolPointMat ... "
            fileName1= 'deltaVolPointMatPlusEps_%s.bin'%self.comm.size
            if os.path.isfile(fileName1):
                os.remove(fileName1)
            fileName2= 'deltaVolPointMatPlus2Eps_%s.bin'%self.comm.size
            if os.path.isfile(fileName2):
                os.remove(fileName2)
        self.comm.Barrier()
        
        # get epsFFD
        epsFFD = self.getOption('epsderivffd')
        if self.comm.rank == 0:
            print ("epsFFD: "+str(epsFFD))
        
        # get the old vol points coords, this is a single list including all the xyz coords
        oldVolPoints = self.getVolumePoints()
        nXPts = len(oldVolPoints)
        
        # create the delta point mat
        plusEpsVolPointMat = PETSc.Mat().create(PETSc.COMM_WORLD)
        plusEpsVolPointMat.setSizes( ((nXPts,None),(None,nDVs)) )
        plusEpsVolPointMat.setFromOptions()
        plusEpsVolPointMat.setPreallocationNNZ((nDVs,nDVs))
        plusEpsVolPointMat.setUp()
        Istart, Iend = plusEpsVolPointMat.getOwnershipRange()
         
        # create xDVDot vec and initialize it with zeros
        # need to sort the key to get an unique key sequence, so that the evalFuncsSens can
        # read the sensitivity in the exact same order
        xDV = self.DVGeo.getValues()
        xDvDot = {}
        for key in sorted(xDV.keys()):
            # get the length of DV for this key
            try:
                lenDVs = len( xDV[key] )
            except: # if xDV[key] is just a number
                lenDVs = 1
            xDvDot[key] = np.zeros(lenDVs,self.dtype)
        
        # get the original surf coords    
        ptSetName = self.getPointSetName('dummy')
        xSDot0 = np.zeros_like(self.coords0,self.dtype)
        xSDot0 = self.mapVector(xSDot0, self.allFamilies,self.designFamilyGroup)
        
        # for each DV, perturb epsFFD and save the delta vol point coordinates
        dvCol=0
        for key in sorted(xDV.keys()):
            # get the length of DV for this key
            try:
                lenDVs = len( xDV[key] )
            except: # if xDV[key] is just a number
                lenDVs = 1
            # now perturb each xDv and get xVDot
            for i in range(lenDVs):
                # perturb xDvDot
                xDvDot[key][i] += 1.0e6 # note here we will need to perturb a very large value (1.0e6)!
                # get xSDot
                xSDot = self.DVGeo.totalSensitivityProd(xDvDot,ptSetName=ptSetName,comm=self.comm).reshape(xSDot0.shape)
                # get xVDot
                xVDot = self.mesh.warpDerivFwd(xSDot)
                # reset xDvDot
                xDvDot[key][i] -= 1.0e6 # reset the perturbation
                # assign the delta vol coords to the mat
                for idx in xrange(Istart, Iend):
                    idxRel = idx-Istart
                    deltaVal = xVDot[idxRel] * epsFFD / 1.0e6  # scale the result back
                    if abs(deltaVal) > deltaVPointThreshold: # a threshold
                        plusEpsVolPointMat[idx,dvCol] = deltaVal
                # increment the mat col number to the next DV
                dvCol+=1
                
        # assemble   
        plusEpsVolPointMat.assemblyBegin()
        plusEpsVolPointMat.assemblyEnd()
        # save
        viewer = PETSc.Viewer().createBinary('deltaVolPointMatPlusEps_%s.bin'%self.comm.size, 'w')
        viewer(plusEpsVolPointMat)

        return
        
    def _writeDeltaVolPointMat(self,deltaVPointThreshold=1.0E-16):
        '''
        Perturb each design variable and save the delta volume point coordinates
        to a mat, this will be used to calculate dRdFFD and dJdFFD in the OpenFOAM layer.
        NOTE: This is the finite-difference version of writeDeltaVolPointMatNew.
        
        Parameters
        ----------
        deltaVPointThreshold: float
            A threshold, any delta volume coordinates smaller than this value will be ignored
            
        '''
        
        # Get the FFD size
        nDVs= self.getOption('nffdpoints')
        getNDVs=0
        for key in self.DVGeo.getValues().keys():
            getNDVs += len( self.DVGeo.getValues()[key] )
        #getNDVs = self.DVGeo.getNDV() # this will have issues for child FFDs
        if not nDVs == getNDVs:
            raise Error("nffdpoints="+str(nDVs)+" while nDVs="+str(getNDVs))
            exit(1)

        if self.comm.rank == 0:
            print "Delete Old DeltaVolPointMat ... "
            fileName1= 'deltaVolPointMatPlusEps_%s.bin'%self.comm.size
            if os.path.isfile(fileName1):
                os.remove(fileName1)
        self.comm.Barrier()
    
        # get the old vol points coords, this is a single list including all the xyz coords
        oldVolPoints = self.getVolumePoints()
        nXPts = len(oldVolPoints)
        # get eps
        epsFFD = self.getOption('epsderivffd')
        if self.comm.rank == 0:
            print ("epsFFD: "+str(epsFFD))
           
        # ************** perturb +epsFFD ****************
        plusEpsVolPointMat = PETSc.Mat().create(PETSc.COMM_WORLD)
        plusEpsVolPointMat.setSizes( ((nXPts,None),(None,nDVs)) )
        plusEpsVolPointMat.setFromOptions()
        plusEpsVolPointMat.setPreallocationNNZ((nDVs,nDVs))
        plusEpsVolPointMat.setUp()
        Istart, Iend = plusEpsVolPointMat.getOwnershipRange()
        # for each DV, perturb epsFFD and save the delta vol point coordinates
        dvCol=0
        # need to sort the key to get an unique key sequence, so that the evalFuncsSens can
        # read the sensitivity in the exact same order
        xDV = self.DVGeo.getValues()
        for key in sorted(xDV.keys()):
            if self.comm.rank == 0:
                print ("Writting deltaVolPointMatPlusEps for "+key)
            # get the length of DV for this key
            try:
                lenDVs = len( xDV[key] )
            except: # if xDV[key] is just a number
                lenDVs = 1
            # loop over all the dvs in this key and perturb
            for i in range(lenDVs):
                # perturb
                try: 
                    xDV[key][i]+=epsFFD
                except:
                    xDV[key]+=epsFFD
                self.DVGeo.setDesignVars(xDV)
                # update the vol points according to the new DV values
                self.updateVolumePoints()
                # get the new vol points
                newVolPoints = self.getVolumePoints()
                # assign the delta vol coords to the mat
                for idx in xrange(Istart, Iend):
                    idxRel = idx-Istart
                    deltaVal = newVolPoints[idxRel] - oldVolPoints[idxRel]
                    if abs(deltaVal) > deltaVPointThreshold: # a threshold
                        plusEpsVolPointMat[idx,dvCol] = deltaVal
                try: 
                    xDV[key][i]-=epsFFD
                except:
                    xDV[key]-=epsFFD
                dvCol+=1
        # assemble   
        plusEpsVolPointMat.assemblyBegin()
        plusEpsVolPointMat.assemblyEnd()
        # save
        viewer = PETSc.Viewer().createBinary('deltaVolPointMatPlusEps_%s.bin'%self.comm.size, 'w')
        viewer(plusEpsVolPointMat)
    
        return

    def getVolumePoints(self):
        '''
        Get the volume point coordinates from mesh
        '''

        volPoints = self.mesh.getSolverGrid()

        return volPoints
                    
    def _readXvSensitivity(self,obj,readCurrent=False):
        '''
        Read the coordinate sensitivity from a file.

        Parameters
        ----------
        
        obj : str
            the name of the objective for which the sensitivity is
            required.
        
        readCurrent : boolean
            whether to read the xv sens file from the current dir. if not, read it from the output dir
        '''
        # get file name
        if self.getOption('multipointopt'):
            filePrefix = 'FC%d_'%self.multiPointFCIndex
        else:
            filePrefix = ''

        if readCurrent:
            readDir = os.getcwd()
            objFile = 'objFuncsSens_d%sdXv.bin'%(self.possibleObjectives[obj])
        else:
            readDir = self.getOption('outputdirectory')
            objFile = 'objFuncsSens_d%sdXv_%s%3.3d.bin'%(self.possibleObjectives[obj],
                       filePrefix,self.adjointRunsCounter-1)

        fileName = os.path.join(readDir,objFile) 
        
        if self.comm.rank == 0:
            print("Reading %s"%fileName)  

        # Get the current nodes from the mesh so we know the size
        nodes = self.mesh.getCommonGrid()
        nLocalXv = len(nodes.flatten())
        tJtX = np.zeros([nLocalXv],self.mesh.dtype)

        # read the Xv sens
        sensXv = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        sensXv.setSizes( (nLocalXv,PETSc.DECIDE),bsize=1)
        sensXv.setFromOptions()
        Istart, Iend = sensXv.getOwnershipRange()

        viewer = PETSc.Viewer().createBinary(fileName,comm=PETSc.COMM_WORLD)
        sensXv.load(viewer)

        for idx in xrange(Istart, Iend):
            idxRel = idx-Istart
            tJtX[idxRel]=sensXv[idx]
        # end

        return tJtX

    def _readFFDSensitivity(self,obj):
        '''
        Read the FFD sensitivity from a file. And convert it to a dict for pyOptSparse

        Parameters
        ----------
        
        obj : str
            the name of the objective for which the sensitivity is
            required.
        '''
        # Get the FFD size
        nDVs= self.getOption('nffdpoints')
        getNDVs=0
        for key in self.DVGeo.getValues().keys():
            getNDVs += len( self.DVGeo.getValues()[key] )
        #getNDVs = self.DVGeo.getNDV()
        if not nDVs == getNDVs:
            print ("Error! nffdpoints="+str(nDVs)+" while nDVs="+str(getNDVs))
            exit(1)
        
        # get file name
        outputDir = self.getOption('outputdirectory')
        
        if self.getOption('multipointopt'):
            filePrefix = 'FC%d_'%self.multiPointFCIndex
        else:
            filePrefix = ''
        
         # note we need to read self.adjointRunsCounter-1
        objFile = "objFuncsSens_d%sdFFD_%s%3.3d.dat"%(self.possibleObjectives[obj],filePrefix,self.adjointRunsCounter-1)
        fileName = os.path.join(outputDir,objFile)
        
        if self.comm.rank == 0:
            print("Reading %s"%fileName) 
 
        #read in the sensitivity file (petsc vec with matlab output format)
        f = open(fileName, 'r')
        lines = f.readlines()
        f.close()

        ffdIn=[]
        lineCounter = 0
        for line in lines:
            # don't read the header and last line
            if lineCounter > 2 and lineCounter < nDVs+3:
                vals = line.split()
                ffdIn.append(float(vals[0]))
            lineCounter+=1
        # end

        # assign tJtFFD
        tJtFFD = {}
        
        xDVs = self.DVGeo.getValues()
        
        # now loop over all the keys in sorted order and read the sens
        ffdCounterI=0
        # NOTE: we need to first sort the keys for all the local and global DVs
        # so that the sequence is consistent with the writeDeltaVolPointMat function
        for key in sorted(xDVs.keys()):
            if self.comm.rank == 0:
                print ("Reading Function Sens for "+key)
            
            # get the length of DV for this key
            lenDVs = len(xDVs[key])
            # loop over all the DVs for this key and assign
            tJtFFD[key] = np.zeros([lenDVs],self.mesh.dtype)
            for i in range(lenDVs):
                tJtFFD[key][i] = ffdIn[ffdCounterI]
                ffdCounterI+=1

        # check if something is missing
        if not ffdCounterI == nDVs:
            raise Error("nDVs is %d while ffdCounterI is %d!"%(nDVs,ffdCounterI))
            exit(1)
                    
        return tJtFFD

    def _readUInSensitivity(self,obj):
        '''
        Read the inlet sensitivity from a file.

        Parameters
        ----------
        
        obj : str
            the name of the objective for which the sensitivity is
            required.
        '''
        # get file name
        outputDir = self.getOption('outputdirectory')
        
        if self.getOption('multipointopt'):
            filePrefix = 'FC%d_'%self.multiPointFCIndex
        else:
            filePrefix = ''

        objFile = 'objFuncsSens_d%sdUIn_%s%3.3d.dat'%(self.possibleObjectives[obj],filePrefix,self.adjointRunsCounter-1)
        fileName = os.path.join(outputDir,objFile) 
        
        if self.comm.rank == 0:
            print("Reading %s"%fileName)   

        #read in the sensitivity file
        f = open(fileName, 'r')
        lines = f.readlines()
        f.close()
        
        if self.mesh == None:
            dtype ='d'
        else:
            dtype = self.mesh.dtype
        
        dJdInlet = np.zeros([3],dtype)

        nCounter=0
        lineCounter=0
        for line in lines:
            # don't read the header and last line
            if lineCounter > 2 and lineCounter < 6:
                vals = line.split()
                val = float(vals[0])
                dJdInlet[nCounter]=val
                nCounter=nCounter+1
            lineCounter +=1
        # end

        return dJdInlet


    def _readVisSensitivity(self,obj):
        '''
        Read the viscosity sensitivity from a file.

        Parameters
        ----------
        
        obj : str
            the name of the objective for which the sensitivity is
            required.
        '''
        # get file name
        outputDir = self.getOption('outputdirectory')
        
        if self.getOption('multipointopt'):
            filePrefix = 'FC%d_'%self.multiPointFCIndex
        else:
            filePrefix = ''

        objFile = 'objFuncsSens_d%sdVis_%s%3.3d.dat'%(self.possibleObjectives[obj],filePrefix,self.adjointRunsCounter-1)
        fileName = os.path.join(outputDir,objFile) 
        
        if self.comm.rank == 0:
            print("Reading %s"%fileName)   

        #read in the sensitivity file
        f = open(fileName, 'r')
        lines = f.readlines()
        f.close()
        
        if self.mesh == None:
            dtype ='d'
        else:
            dtype = self.mesh.dtype
        
        dJdVis = np.zeros([1],dtype)

        lineCounter=0
        for line in lines:
            # don't read the header and last line
            if lineCounter ==3:
                vals = line.split()
                val = float(vals[0])
                dJdVis[0]=val
            lineCounter +=1
        # end

        return dJdVis

    def getSurfaceCoordinates(self, groupName=None):
        """Return the coordinates for the surfaces defined by groupName. 

        Parameters
        ----------
        groupName : str
            Group identifier to get only coordinates cooresponding to
            the desired group. The group must be a family or a
            user-supplied group of families. The default is None which
            corresponds to all wall-type surfaces.
        """
        
        if groupName is None:
            groupName = self.allWallsGroup

        # Get the required size
        npts, ncell = self._getSurfaceSize(groupName)
        pts = np.zeros((npts, 3), self.dtype)

        # loop over the families in this group and populate the surface
        famInd = self.families[groupName]
        counter = 0
        for Ind in famInd:
            name = self.basicFamilies[Ind]
            bc = self.boundaries[name]
            for ptInd in bc['indicesRed']:
                pts[counter,:] = self.x[ptInd]
                counter +=1

        return pts

    def getSurfaceConnectivity(self, groupName=None):
        """Return the connectivity of the coordinates at which the forces (or tractions) are
        defined. This is the complement of getForces() which returns
        the forces at the locations returned in this routine. 

        Parameters
        ----------
        groupName : str
            Group identifier to get only forces cooresponding to the
            desired group. The group must be a family or a user-supplied 
            group of families. The default is None which corresponds to 
            all wall-type surfaces.
        """
        
        if groupName is None:
            groupName = self.allWallsGroup

        # loop over the families in this group and populate the connectivity
        famInd = self.families[groupName]
        conn = []
        faceSizes = []

        pointOffset = 0
        for Ind in famInd:
            # select the face from the basic families
            name = self.basicFamilies[Ind]

            # get the size of this
            bc = self.boundaries[name]
            nPts = len(bc['indicesRed'])            

            # get the number of reduced faces associated with this boundary
            nFace = len(bc['facesRed'])

            # check that this isn't an empty boundary
            if nFace > 0:
                # loop over the faces and add them to the connectivity and faceSizes array
                for iFace in xrange(nFace):
                    face = copy.copy(bc['facesRed'][iFace])
                    for i in xrange(len(face)):
                        face[i] +=pointOffset
                    conn.extend(face)
                    faceSizes.append(len(face))
                    
                pointOffset+=nPts

        return conn,faceSizes

    def updateGeometryInfo(self):

        pass

 
    def _getSolution(self):
        '''
        Get all of the solution values that are available in 
        objFuncs.dat or objFuncsMean.dat
        '''
        sol = {}
        sol.update(self._readObjectiveFunctions()) 
        
        return sol
        

# --------------------------
# Private Utility functions
# --------------------------
    def _getSurfaceSize(self, groupName):
        """Internal routine to return the size of a particular surface. This
        does *NOT* set the actual family group"""
        if groupName is None:
            groupName = self.allFamilies

        if groupName.lower() not in self.families:
            raise Error("'%s' is not a family in the OpenFoam Case or has not been added"
                        " as a combination of families"%groupName)

        # loop over the basic surfaces in the family group and sum up the number of
        # faces and nodes
        
        famInd = self.families[groupName]
        nPts = 0
        nCells = 0
        for Ind in famInd:
            name = self.basicFamilies[Ind]
            bc = self.boundaries[name]
            nCells += len(bc['facesRed'])
            nPts += len(bc['indicesRed'])

        return nPts, nCells
        
    def mapVector(self, vec1, groupName1, groupName2, vec2=None):
        """This is the main workhorse routine of everything that deals with
        families in pyDAFoam. The purpose of this routine is to convert a
        vector 'vec1' (of size Nx3) that was evaluated with
        'groupName1' and expand or contract it (and adjust the
        ordering) to produce 'vec2' evaluated on groupName2.

        A little ascii art might help. Consider the following "mesh"
        . Family 'fam1' has 9 points, 'fam2' has 10 pts and 'fam3' has
        5 points.  Consider that we have also also added two
        additional groups: 'f12' containing 'fam1' and 'fma2' and a
        group 'f23' that contains families 'fam2' and 'fam3'. The vector
        we want to map is 'vec1'. It is length 9+10. All the 'x's are
        significant values.

        The call: mapVector(vec1, 'f12', 'f23')
        
        will produce the "returned vec" array, containing the
        significant values from 'fam2', where the two groups overlap,
        and the new values from 'fam3' set to zero. The values from
        fam1 are lost. The returned vec has size 15. 

            fam1     fam2      fam3
        |---------+----------+------|

        |xxxxxxxxx xxxxxxxxxx|        <- vec1
                  |xxxxxxxxxx 000000| <- returned vec (vec2)


        Parameters
        ----------
        vec1 : Numpy array
            Array of size Nx3 that will be mapped to a different family set.

        groupName1 : str
            The family group where the vector vec1 is currently defined

        groupName2 : str
            The family group where we want to the vector to mapped into

        vec2 : Numpy array or None
            Array containing existing values in the output vector we want to keep. 
            If this vector is not given, the values will be filled with zeros. 

        Returns
        -------
        vec2 : Numpy array
            The input vector maped to the families defined in groupName2.
        """
        if groupName1 not in self.families or groupName2 not in self.families:
            raise Error("'%s' or '%s' is not a family in the CGNS file or has not been added"
                        " as a combination of families"%(groupName1, groupName2))
       
        # Shortcut:
        if groupName1 == groupName2:
            return vec1

        if vec2 is None:
            npts, ncell = self._getSurfaceSize(groupName2)
            vec2 = np.zeros((npts, 3), self.dtype)
            
        famList1 = self.families[groupName1]
        famList2 = self.families[groupName2]

        '''
        This functionality is predicated on the surfaces being traversed in the 
        same order every time. Loop over the allfamilies list, keeping track of sizes
        as we go and if the family is in both famLists, copy the values from vec1 to vec2.

        '''

        vec1counter = 0
        vec2counter = 0

        for ind in self.families[self.allFamilies]:
            npts, ncell = self._getSurfaceSize(self.basicFamilies[ind])

            if  ind in famList1 and ind in famList2:
                vec2[vec2counter:npts+vec2counter] =  vec1[vec1counter:npts+vec1counter] 

            if ind in famList1:
                vec1counter+=npts                

            if ind in famList2:
                vec2counter+=npts

        return vec2


    # *******************************************
    # File handling routines for IO based control
    # *******************************************

    # base case files
    def _readGrid(self, caseDir):
        """
        Read in the mesh information we need to run the case.

        Parameters
        ----------
        caseDir : str
            The directory containing the openFOAM Mesh files
        """

        # generate the file names
        self.fileNames = ofm.getFileNames(caseDir,comm=self.comm)

        # Copy the reference points file to points to ensure
        # consistant starting point
        if self.getOption('writecompress')=='on':
            pointName='points.gz'
            pointNameOrig='points_orig.gz'
        elif self.getOption('writecompress')=='off':
            pointName='points'
            pointNameOrig='points_orig'
        else:
            raise Error('writecompress not valid')

        if self.comm.rank == 0:
            print("Copying points_orig to points")
        if self.parallel:
            if self.comm.rank == 0:
                for i in range(self.comm.size):
                    pointsOrig=os.path.join(caseDir, 'processor%d/constant/polyMesh/%s'%(i,pointNameOrig))
                    pointsCopy=os.path.join(caseDir, 'processor%d/constant/polyMesh/%s'%(i,pointName))
                    try:
                        shutil.copyfile(pointsOrig,pointsCopy)
                    except:
                        raise Error('pyDAFoam: Unable to copy %s to %s.'%(pointsOrig,pointsCopy))
                        sys.exit(0)
        else:
            pointsOrig=os.path.join(caseDir, 'constant/polyMesh/%s'%pointNameOrig)
            pointsCopy=os.path.join(caseDir, 'constant/polyMesh/%s'%pointName)
            try:
                shutil.copyfile(pointsOrig,pointsCopy)
            except:
                raise Error('pyDAFoam: Unable to copy %s to %s.'%(pointsOrig,pointsCopy))
                sys.exit(0)
        
        # wait for rank 0 
        self.comm.Barrier()

        if self.comm.rank == 0:
            print("Reading points")

        # Read in the volume points
        self.x0 = ofm.readVolumeMeshPoints(self.fileNames)
        self.x = copy.copy(self.x0)
        
        # Read the face info for the mesh
        self.faces = ofm.readFaceInfo(self.fileNames)

        # Read the boundary info
        self.boundaries = ofm.readBoundaryInfo(self.fileNames,self.faces)

        # Read the cell info for the mesh
        self.owners,self.neighbours = ofm.readCellInfo(self.fileNames)

        self.nCells= self._countCells()


    def _computeBasicFamilyInfo(self):
        '''
        Loop over the boundary data and compute necessary family information for the basic patches

        '''
        # get the list of basic families
        self.basicFamilies = sorted(self.boundaries.keys())
        
        # save and return a list of the wall boundaries
        self.wallList = []
        counter = 0
        # for each boundary, figure out the unique list of volume node indices it uses
        for name in self.basicFamilies:
            # setup the basic families dictionary
            self.families[name]=[counter]
            counter+=1
            
            # Create a handle for this boundary
            bc = self.boundaries[name]

            # get the number of faces associated with this boundary
            nFace = len(bc['faces'])

            # create the index list
            indices = []

            # check that this isn't an empty boundary
            if nFace > 0:
                for iFace in bc['faces']:
                    # get the node information for the current face
                    face = self.faces[iFace]
                    indices.extend(face)
            
            # Get the unique entries 
            indices = np.unique(indices)
            
            # now create the reverse dictionary to connect the reduced set with the original
            inverseInd = {}
            for i in xrange(len(indices)):
                inverseInd[indices[i]]=i

            # Now loop back over the faces and store the connectivity in terms of the reduces index set
            facesRed = []
            for iFace in bc['faces']:
                # get the node information for the current face
                face = self.faces[iFace]
                nNodes = len(face)
                # Generate the reduced connectivity.
                faceReduced = []
                for j in range(nNodes):
                    indOrig = face[j]
                    indRed = inverseInd[indOrig]
                    faceReduced.append(indRed)
                facesRed.append(faceReduced)
                
            #Check that the length of faces and facesRed are equal
            if not (len(bc['faces'])==len(facesRed)):
                raise Error("Connectivity for faces on reduced index set is not the same length as original.")

            # put the reduced faces and index list in the boundary dict
            bc['facesRed'] = facesRed
            bc['indicesRed'] = list(indices)

            # now check for walls
            if bc['type']=='wall' or bc['type']=='slip' or bc['type']=='cyclic':
                self.wallList.append(name)
                    
        return
   

# ---------------------------------------------------------------
# Additional files defined for the adjoint solver and optimization
# ---------------------------------------------------------------

    def _readObjectiveFunctions(self):
        '''
        read in the costFunctions from objFuncs.dat or objFuncsMean.dat
        '''
        
        outputDir = self.getOption('outputdirectory')
        
        if self.getOption('multipointopt'):
            filePrefix = 'FC%d_'%self.multiPointFCIndex
        else:
            filePrefix = ''
            
        if self.getOption('avgobjfuncs'):
            costFuncsFileName = os.path.join(outputDir,"objFuncsMean_%s%3.3d.dat"%(filePrefix,self.flowRunsCounter-1)) # note we need to read self.flowRunsCounter-1
        else:
            costFuncsFileName = os.path.join(outputDir,"objFuncs_%s%3.3d.dat"%(filePrefix,self.flowRunsCounter-1)) # note we need to read self.flowRunsCounter-1
        
        if self.comm.rank == 0:
            print("Reading %s"%costFuncsFileName)
            
        f = open(costFuncsFileName,'r')
        
        tmpSol = {}
        sol = {}

        for line in f.readlines():
            vals = line.split()
            if '#'in vals:
                #print('comment:',vals)
                # This is a comment, ignore unless time is in the list
                pass
            else:
                #print('active:',vals)
                key = vals[0]
                val = float(vals[1])
                #print 'power',key,val,type(key)
                tmpSol[key] = val
            # end
        #end
        #print ('tmpsol',tmpSol.keys())
        for key in tmpSol.keys():
            sol[key]=tmpSol[key]
        # end
        
        f.close()
        
        return sol

# Solution cleaning

    def _cleanPostprocessingDir(self):
        '''
        Delete the files in the postprocessing directory from the previous 
        iteration.
        '''
        postProcessingDir = os.path.join(os.getcwd(),self.getOption('postprocessingdir'))
        #print('direxists',os.path.isdir(postProcessingDir))
        isDir = os.path.isdir(postProcessingDir)
        if isDir:
            if self.comm.rank==0:
                print("Deleting the postProcessing Folder")
                try:
                    shutil.rmtree("postProcessing")
                except:
                    raise Error('pyDAFoam: Unable to remove postprocessing directory')
                    sys.exit(0)
                # end
            # end
        # end
        
        self.comm.Barrier()

        return 

    def _cleanSolution(self):
        '''
        Delete the files from the previous solution.
        '''
        
        if self.comm.rank==0:
            print("Deleting Previous Solution Files") 
         
        solutionDir = os.getcwd()
        if self.parallel:
            # loop over the number of proccessors, deleating each solution directory along the
            #way
            for i in range(self.comm.size):
                solutionDirPar = os.path.join(solutionDir,'processor%d'%i)
                self._deleteDir(solutionDirPar)
        else:
            self._deleteDir(solutionDir)
 
        return 
        
    def _deleteDir(self,solutionDir):
        '''
        delete a single directory only on the root process
        '''
        if self.comm.rank==0:
            isDir = os.path.isdir(solutionDir)
            if isDir:
                for f in os.listdir(solutionDir):
                    # Create a regular expression for all of the non-zero numeric directories
                    dirNum = re.compile(r'([1-9]+)([0-9]*)')
                    dirNum1 = re.compile(r'^\d+\.\d+$') # for folder such as 0.01
 
                    # If this is one of those directories, delete it
                    res = dirNum.match(f)
                    res1 = dirNum1.match(f)
                    #print res
                    if res or res1:
                        
                        delSolutionDir=solutionDir+'/%s'%f
                        try:
                            shutil.rmtree(delSolutionDir)
                        except:
                            raise Error('pyDAFoam: Unable to remove solution directory: %s'%f)
                            sys.exit(0)
                            
                        # end
                    # end
                # end
            # end
            
        # Wait for all of the processors to get here
        self.comm.Barrier()
        
        return 

    def _copyResults(self,origDir,newDir):
        """
        Copy the orig directory to the requested new directory. Used to save results
        during a run.
        """
        if self.comm.rank==0:
            print("Removing previous solutions at %s"%newDir)
            if os.path.exists(newDir):
                try:
                    shutil.rmtree(newDir)
                except:
                    print('pyDAFoam: Unable to remove existing directory %s.'%newDir)
            # end
            
            print("Copying solutions %s to %s"%(origDir,newDir))
            IGNORE_PATTERNS = ('*.bin','*.info','*.swp','*.clr')
            try:
                shutil.copytree(origDir,newDir,ignore=shutil.ignore_patterns(*IGNORE_PATTERNS))
            except:
                print('pyDAFoam: Unable to copy directory to %s.'%newDir)
            # end 

            if self.getOption('writecompress')=='on':
                pointNameOrig='points_orig.gz'
            elif self.getOption('writecompress')=='off':
                pointNameOrig='points_orig'
            else:
                raise Error('writecompress not valid')

            if not self.parallel:
                # Now remove the original point file so that the geometry is plotted
                # correctly in tecplot
                pointsFileName = 'constant/polyMesh/%s'%pointNameOrig
                pointsFile = os.path.join(newDir,pointsFileName)
                if os.path.exists(pointsFile):
                    try:
                        os.remove(pointsFile)
                    except:
                        print('pyDAFoam: Unable to remove file %s.'%pointsFile)
                    # end

            #repeat for parallel directories
            if self.parallel:
                for i in range(self.comm.size):
                    # this is a parallel case remove points file from processor dirs 
                    pointsFileName = 'processor%d/constant/polyMesh/%s'%(i,pointNameOrig)
                    pointsFile = os.path.join(newDir,pointsFileName)
                    if os.path.exists(pointsFile):
                        try:
                            os.remove(pointsFile)
                        except:
                            print('pyDAFoam: Unable to remove file %s.'%pointsFile)
            
        # end
        self.comm.Barrier()
        
        return
    
    def _checkRestartLogs(self,logOpt):
        """
        Check if the logs and data exist for restart runs, if not, set self.skipFlowAndAdjointRuns == False
        """
        
        if self.getOption('multipointopt') and self.multiPointFCIndex==None:
            raise Error('multipointopt is true while multiPointFCIndex is not set')
        
        if not self.skipFlowAndAdjointRuns:
            return
        
        outputDir = self.getOption('outputdirectory')
        
        if logOpt == 1:
            # check flowLog
            if not self.getOption('multipointopt'):
                fileName = os.path.join(outputDir,'flowLog_%3.3d'%self.flowRunsCounter)
            else:
                fileName = os.path.join(outputDir,'flowLog_FC%d_%3.3d'%(self.multiPointFCIndex,self.flowRunsCounter))

            if not os.path.isfile(fileName):
                self.skipFlowAndAdjointRuns = False
                return
                
            # check objFuncs.dat 
            if not self.getOption('multipointopt'):
                fileName = os.path.join(outputDir,'objFuncs_%3.3d.dat'%self.flowRunsCounter)
            else:
                fileName = os.path.join(outputDir,'objFuncs_FC%d_%3.3d.dat'%(self.multiPointFCIndex,self.flowRunsCounter))
                
            if not os.path.isfile(fileName):
                self.skipFlowAndAdjointRuns = False
                return
            
            # check objFuncsMean.dat 
            if self.getOption('avgobjfuncs'):
                if not self.getOption('multipointopt'):
                    fileName = os.path.join(outputDir,'objFuncsMean_%3.3d.dat'%self.flowRunsCounter)
                else:
                    fileName = os.path.join(outputDir,'objFuncsMean_FC%d_%3.3d.dat'%(self.multiPointFCIndex,self.flowRunsCounter))
                    
                if not os.path.isfile(fileName):
                    self.skipFlowAndAdjointRuns = False
                    return
                    
        elif logOpt == 2:
            # check adjointLog 
            if not self.getOption('multipointopt'):
                fileName = os.path.join(outputDir,'adjointLog_%3.3d'%self.adjointRunsCounter)
            else:
                fileName = os.path.join(outputDir,'adjointLog_FC%d_%3.3d'%(self.multiPointFCIndex,self.adjointRunsCounter))

            if not os.path.isfile(fileName):
                self.skipFlowAndAdjointRuns = False
                return

            # check designVariables.dat 
            if not self.getOption('multipointopt'):
                fileName = os.path.join(outputDir,'designVariables_%3.3d.dat'%self.adjointRunsCounter)
            else:
                fileName = os.path.join(outputDir,'designVariables_FC%d_%3.3d.dat'%(self.multiPointFCIndex,self.adjointRunsCounter))

            if not os.path.isfile(fileName):
                self.skipFlowAndAdjointRuns = False
                return
           
            # check sensitivities
            if not self.getOption('multipointopt'):        
                workingDir= os.getcwd()
            else:
                workingDir= os.path.join(os.getcwd(),'../FlowConfig%d'%(self.multiPointFCIndex))
            allFiles = os.listdir(workingDir)
            for file1 in allFiles:
                if 'objFuncsSens' in file1:
                    sensFileName = os.path.splitext(file1)[0]
                    sensFileExt = os.path.splitext(file1)[1]
                    if sensFileExt == ".info":
                        continue
                    if not self.getOption('multipointopt'):
                        fileName=os.path.join(outputDir,sensFileName+'_%3.3d'%self.adjointRunsCounter+sensFileExt)
                    else:
                        fileName=os.path.join(outputDir,sensFileName+'_FC%d_%3.3d'%(self.multiPointFCIndex,self.adjointRunsCounter)+sensFileExt)
                        
                    if not os.path.isfile(fileName):
                        self.skipFlowAndAdjointRuns = False
                        return
                    
        elif logOpt == 3:
            # check checkMeshLog
            if not self.getOption('multipointopt'):
                fileName = os.path.join(outputDir,'checkMeshLog_%3.3d'%self.flowRunsCounter)
            else:
                fileName = os.path.join(outputDir,'checkMeshLog_FC%d_%3.3d'%(self.multiPointFCIndex,self.flowRunsCounter))

            if not os.path.isfile(fileName):
                self.skipFlowAndAdjointRuns = False
                return
 
        else:
            raise Error('logOpt not valid')
        
        return         
   

    def _writeDVs(self):
        '''
        Write the crrent DV values to files
        '''

        if self.DVGeo is None:
            return
        else:
            dvs=self.DVGeo.getValues()
            if self.comm.rank==0:
                f=open('designVariables.dat','w')
                for key in dvs.keys():
                    f.write('%s '%key)
                    for val in dvs[key]:
                        f.write('%.15e '%val)
                    f.write('\n')
                f.close()
            self.comm.Barrier()
             
      
    def _copyLogs(self,logOpt):
        """
        Number the flowLog, adjointLog, checkMeshLog, objFuncs.dat, objFuncsMean.dat, objFuncsSens.dat and 
        copy them to the outputdirectory. The evalFunctions and evalFunctionsSens functions will then read the
        objective functions and their sensitvities from the outputdirectory. If restartopt=true, The evalFunctions 
        and evalFunctionsSens functions will directly read information from the outputdirectory without running
        any flow or adjoint until the optimization reach the restart point. This will reproduce exact optimization
        process.
        
        NOTE: for checkMeshLog, we always copy from the current directory
        
        Parameters
        ----------
        logOpt : int
            1: flow; 2: adjoint; 3: checkMesh
        """
        
        if self.getOption('multipointopt') and self.multiPointFCIndex==None:
            raise Error('multipointopt is true while multiPointFCIndex is not set')
        
        outputDir = self.getOption('outputdirectory')
        
        if logOpt == 1:
            # copy flowLog to the outputdirectory
            if not self.getOption('multipointopt'):
                fileOrig = 'flowLog'
                fileCopy = os.path.join(outputDir,'flowLog_%3.3d'%self.flowRunsCounter)
            else:
                fileOrig = '../FlowConfig%d/flowLog'%(self.multiPointFCIndex)
                fileCopy = os.path.join(outputDir,'flowLog_FC%d_%3.3d'%(self.multiPointFCIndex,self.flowRunsCounter))
                
            if self.comm.rank == 0:
                print("Copying %s to %s"%(fileOrig,fileCopy))
                shutil.copyfile(fileOrig,fileCopy)
                
            # copy objFuncs.dat to the outputdirectory    
            if not self.getOption('multipointopt'):
                fileOrig = 'objFuncs.dat'
                fileCopy = os.path.join(outputDir,'objFuncs_%3.3d.dat'%self.flowRunsCounter)
            else:
                fileOrig = '../FlowConfig%d/objFuncs.dat'%(self.multiPointFCIndex)
                fileCopy = os.path.join(outputDir,'objFuncs_FC%d_%3.3d.dat'%(self.multiPointFCIndex,self.flowRunsCounter))
                
            if self.comm.rank == 0:
                print("Copying %s to %s"%(fileOrig,fileCopy))
                shutil.copyfile(fileOrig,fileCopy)
            
            # copy objFuncsMean.dat to the outputdirectory    
            if self.getOption('avgobjfuncs'):
                if not self.getOption('multipointopt'):
                    fileOrig = 'objFuncsMean.dat'
                    fileCopy = os.path.join(outputDir,'objFuncsMean_%3.3d.dat'%self.flowRunsCounter)
                else:
                    fileOrig = '../FlowConfig%d/objFuncsMean.dat'%(self.multiPointFCIndex)
                    fileCopy = os.path.join(outputDir,'objFuncsMean_FC%d_%3.3d.dat'%(self.multiPointFCIndex,self.flowRunsCounter))
                    
                if self.comm.rank == 0:
                    print("Copying %s to %s"%(fileOrig,fileCopy))
                    shutil.copyfile(fileOrig,fileCopy)
                        
        elif logOpt == 2:
            # copy adjointLog to the outputdirectory
            if not self.getOption('multipointopt'):
                fileOrig = 'adjointLog'
                fileCopy = os.path.join(outputDir,'adjointLog_%3.3d'%self.adjointRunsCounter)
            else:
                fileOrig = '../FlowConfig%d/adjointLog'%(self.multiPointFCIndex)
                fileCopy = os.path.join(outputDir,'adjointLog_FC%d_%3.3d'%(self.multiPointFCIndex,self.adjointRunsCounter))

            if self.comm.rank == 0:
                print("Copying %s to %s"%(fileOrig,fileCopy))
                shutil.copyfile(fileOrig,fileCopy)

            # copy designVariables.dat to the outputdirectory
            if not self.getOption('multipointopt'):
                fileOrig = 'designVariables.dat'
                fileCopy = os.path.join(outputDir,'designVariables_%3.3d.dat'%self.adjointRunsCounter)
            else:
                fileOrig = '../FlowConfig%d/designVariables.dat'%(self.multiPointFCIndex)
                fileCopy = os.path.join(outputDir,'designVariables_FC%d_%3.3d.dat'%(self.multiPointFCIndex,self.adjointRunsCounter))

            if self.comm.rank == 0:
                print("Copying %s to %s"%(fileOrig,fileCopy))
                try:
                    shutil.copyfile(fileOrig,fileCopy)
                except:
                    print("Cannot copy %s to %s!!!"%(fileOrig,fileCopy))
           
            # copy sensitivities
            if not self.getOption('multipointopt'):        
                workingDir= os.getcwd()
            else:
                workingDir= os.path.join(os.getcwd(),'../FlowConfig%d'%(self.multiPointFCIndex))
            allFiles = os.listdir(workingDir)
            for file1 in allFiles:
                if 'objFuncsSens' in file1:
                    sensFileName = os.path.splitext(file1)[0]
                    sensFileExt = os.path.splitext(file1)[1]
                    if sensFileExt == ".info":
                        continue
                    if not self.getOption('multipointopt'):
                        fileOrig = file1
                        fileCopy=os.path.join(outputDir,sensFileName+'_%3.3d'%self.adjointRunsCounter+sensFileExt)
                    else:
                        fileOrig = os.path.join('../FlowConfig%d'%self.multiPointFCIndex,file1)
                        fileCopy=os.path.join(outputDir,sensFileName+'_FC%d_%3.3d'%(self.multiPointFCIndex,self.adjointRunsCounter)+sensFileExt)
                        
                    if self.comm.rank == 0:
                        print("Copying %s to %s"%(fileOrig,fileCopy))
                        shutil.copyfile(fileOrig,fileCopy)
                    
        elif logOpt == 3:
            # copy checkMeshLog to outputdirectory
            if not self.getOption('multipointopt'):
                fileOrig = 'checkMeshLog'
                fileCopy = os.path.join(outputDir,'checkMeshLog_%3.3d'%self.flowRunsCounter)
            else:
                fileOrig = 'checkMeshLog' # NOTE: we always copy checkMeshLog from the current dir
                fileCopy = os.path.join(outputDir,'checkMeshLog_FC%d_%3.3d'%(self.multiPointFCIndex,self.flowRunsCounter))

            if self.comm.rank == 0:
                print("Copying %s to %s"%(fileOrig,fileCopy))
                shutil.copyfile(fileOrig,fileCopy)
 
        else:
            raise Error('logOpt not valid')
               
        self.comm.Barrier()   
        
        return
        

# parallel run scripts

    def writeParallelRunScript(self,logFileName,scriptName):

        if self.comm.rank==0:
            f = open(scriptName,'w')
            
            f.write('#!/bin/bash \n')

            f.write('echo "Running Flow Solver..."\n')
            adjointSolver = self.getOption('adjointsolver')
            f.write('%s -parallel > %s\n'%(adjointSolver,logFileName))
            f.write('echo "Finished, writing end file"\n')
            f.write('touch foamFinished.txt\n')
            f.write('# End of the script file\n')
            
            f.close()

        self.comm.Barrier()

        return

    def writeParallelColoringScript(self,logFileName,scriptName):

        if self.comm.rank==0:
            f = open(scriptName,'w')
            
            f.write('#!/bin/bash \n')
            
            f.write('echo "Running Coloring Solver..."\n')
            f.write('coloringSolver%s -parallel > %s\n'%(self.getOption('flowcondition'),logFileName))
            f.write('echo "Coloring finished"\n')
            f.write('touch coloringFinished.txt\n')
            f.write('# End of the script file\n')
            
            f.close()
        return

    def writeParallelPotentialFoamScript(self,logFileName,scriptName):

        if self.comm.rank==0:
            f = open(scriptName,'w')
            
            f.write('#!/bin/bash \n')
            f.write('potentialFoam -parallel > %s\n'%logFileName)
            f.write('touch potentialFoamFinished.txt\n')
            f.write('# End of the script file\n')
            
            f.close()
            
        self.comm.Barrier()
        
        return
        
    def writeParallelCheckMeshScript(self,logFileName,scriptName):

        if self.comm.rank==0:
            f = open(scriptName,'w')
            
            f.write('#!/bin/bash \n')
            f.write('checkMesh -parallel > %s\n'%logFileName)
            f.write('touch checkMeshFinished.txt\n')
            f.write('# End of the script file\n')
            
            f.close()
            
        self.comm.Barrier()
        
        return
        
                
    def writeSurfaceSensitivityMap(self,evalFuncs=None,groupName='all',fileType='openfoam'):
        """
        Write the surface sensitivities.
        It will read the sensitivity, e.g., objFuncsSens_dCDdXv.bin from the current directory.
        So make sure you have solve the adjoint and the objFuncsSens_dCDdXv.bin file is up-to-date.
 
        Parameters
        ----------
        
        evalFuncs : iterable object containing strings
            The functions the user wants the derivatives of
 
        groupName : str
            The group the user wants the derivatives of
            
        fileType: str
            The format for the sensitivyt map, options are: tecplot or openfoam

        Examples
        --------
        >>> CFDsolver.writeSurfaceSensitivityMap( ['CD','CL'], 'designSurfaces', 'openfoam')
        """

        if self.comm.rank==0:
            print("Writing Surface Sensitivity Map.")
        
        if self.mesh is None:      
            raise Error('Cannot write the sensitivity map without a mesh object')
            
        # Check for evalFuncs, this should eventually have some kind of default
        if evalFuncs is None:
            print 'evalFuncs not set, exiting...'
            sys.exit(0)

        if 'Xv' not in self.getOption('adjdvtypes'):
            raise Error('Xv not in adjDVType!') 
            
        # generate the folders to save the sensitivty files for openfoam output format
        if fileType == 'openfoam':
            if self.comm.size == 1:
                meshDir = '99999/polyMesh/'
                sensDir = '99999/'          
                if not os.path.exists(meshDir):
                    os.makedirs(meshDir)
            else:
                meshDir = 'processor%d/99999/polyMesh/'%self.comm.rank
                sensDir = 'processor%d/99999/'%self.comm.rank
                if self.comm.rank ==0:
                    for i in range(self.comm.size):
                        dir1 = 'processor%d/99999/polyMesh/'%i
                        if not os.path.exists(dir1):
                            os.makedirs(dir1)  
                self.comm.Barrier()
        
        # warp the mesh   
        if self.DVGeo is not None:
            self.mesh.warpMesh()
            
        funcsSens ={}
 
        # Do the functions one at a time:
        for func in evalFuncs:
            
            if self.comm.rank == 0:
                print('Reading adjoint: %s'%func)
 
            key ='%s'% func
 
            # Set dict structure for this derivative
            funcsSens[key] = {}
             
            ptSetName=self.getPointSetName('dummy')
               
            # Geometric derivatives
            dIdXv = self._readXvSensitivity(func,readCurrent=True)
 
            # Now get total derivative wrt surface cordinates
            self.mesh.warpDeriv(dIdXv[:])
            # warp dIdx on surface and map it to groupName
            dIdx1 = self.mesh.getdXs()
            dIdx1 = self.mapVector(dIdx1, self.meshFamilyGroup,groupName) 
            dIdx = {ptSetName:dIdx1}
                
            # add to dict
            funcsSens[key].update(dIdx)
            
            # Obtain the points and connectivity for the specified groupName
            pts = self.getSurfaceCoordinates(groupName)
            conn, faceSizes = self.getSurfaceConnectivity(groupName)
            conn = np.array(conn).flatten()
            
            if fileType == 'tecplot':
                # Now triangulate the surface
              
                # Triangle info...point and two vectors
                p0 = []
                v1 = []
                v2 = []
                sens=[]
                
                # loop over the faces
                connCounter=0
                for iFace in xrange(len(faceSizes)):
                    # Get the number of nodes on this face
                    faceSize = faceSizes[iFace]
                    faceNodes = conn[connCounter:connCounter+faceSize]
                    
                    # Start by getting the centerpoint and the sens at the centerpoint
                    ptSum= [0, 0, 0]
                    sensSum=[0, 0, 0]
                    for i in xrange(faceSize):
                        idx = faceNodes[i]
                        ptSum+=pts[idx]
                        # just get the sum of dIdx at the centerpoint
                        sensSum+=dIdx[ptSetName][idx]
        
                    avgPt = ptSum/faceSize
                    avgSens = sensSum/faceSize
                    # Now go around the face and add a triangle for each adjacent pair
                    # of points. This assumes an ordered connectivity from the
                    # meshwarping
                    for i in xrange(faceSize):
                        idx = faceNodes[i]
                        p0.append(avgPt)
                        v1.append(pts[idx]-avgPt)
                        if(i<(faceSize-1)):
                            idxp1 = faceNodes[i+1]
                            v2.append(pts[idxp1]-avgPt)
                            # calculate the cell centered sensitivity for each triangulated face
                            sensSumTriSurf=avgSens+dIdx[ptSetName][idx]+dIdx[ptSetName][idxp1]
                            sens.append(sensSumTriSurf/3.0)
                            
                        else:
                            # wrap back to the first point for the last element
                            idx0 = faceNodes[0]
                            v2.append(pts[idx0]-avgPt)
                            # calculate the cell centered sensitivity for each triangulated face
                            sensSumTriSurf=avgSens+dIdx[ptSetName][idx]+dIdx[ptSetName][idx0]
                            sens.append(sensSumTriSurf/3.0)
                            
                    # Now increment the connectivity
                    connCounter+=faceSize
                
                # calculate the face normal unit vector
                triSurfUnitNormal=np.zeros(shape=(len(v1),3))
                for indx in range(len(v1)):
                    triSurfUnitNormal[indx]=np.cross(v2[indx],v1[indx]) # v2 X v1 is the outward dir
                    magNorm=np.linalg.norm(triSurfUnitNormal[indx])
                    triSurfUnitNormal[indx]=triSurfUnitNormal[indx]/magNorm
                
                # write the surface sensivitity to tecplot
                fout = open('sensMap_%s.dat'%func, 'w')
                fout.write("TITLE = Sensitivity Map for %s \n"%func)
                fout.write("VARIABLES = \"CoordinateX\" \"CoordinateY\" \"CoordinateZ\" \"SensitivityX\" \"SensitivityY\" \"SensitivityZ\" \"SensitivityNorm\"  \n")
                fout.write('Zone T=%s\n'%('surf'))
                fout.write('Nodes = %d, Elements = %d ZONETYPE=FETRIANGLE\n'% (len(p0)*3, len(p0)))
                fout.write('DATAPACKING=POINT\n')
                for i in range(len(p0)):
                    points = []
                    points.append(p0[i])
                    points.append(p0[i]+v1[i])
                    points.append(p0[i]+v2[i])
                    val=np.dot(sens[i], triSurfUnitNormal[i])
                    for j in range(len(points)):
                        fout.write('%f %f %f %f %f %f %f\n'% (points[j][0], points[j][1],points[j][2],sens[i][0],sens[i][1],sens[i][2],val))
                for i in range(len(p0)):
                    fout.write('%d %d %d\n'% (3*i+1, 3*i+2,3*i+3))
                fout.close()
                
                
            elif fileType == 'openfoam':
     
                ############### write points
                fPoints = open(os.path.join(meshDir,'points'),'w')
                # write the file header
                fPoints.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
                fPoints.write('| =========                 |                                                 |\n')
                fPoints.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
                fPoints.write('|  \\\\    /   O peration     | Version:  plus                                  |\n')
                fPoints.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
                fPoints.write('|    \\\\/     M anipulation  |                                                 |\n')
                fPoints.write('\*---------------------------------------------------------------------------*/\n')
                fPoints.write('FoamFile\n')
                fPoints.write('{\n')
                fPoints.write('    version     2.0;\n')
                fPoints.write('    format      ascii;\n')
                fPoints.write('    class       vectorField;\n')
                fPoints.write('    location    "%s";\n'%meshDir)
                fPoints.write('    object      points;\n')
                fPoints.write('}\n')
                fPoints.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
                fPoints.write('\n')
                
                fPoints.write('%d\n'%len(pts))
                fPoints.write('(\n')
                for i in range(len(pts)):
                    fPoints.write('(%f %f %f)\n'%(float(pts[i][0]),float(pts[i][1]),float(pts[i][2])))
                fPoints.write(')\n')
                fPoints.close()
                
                ################ write faces
                fFaces = open(os.path.join(meshDir,'faces'),'w')
                # write the file header
                fFaces.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
                fFaces.write('| =========                 |                                                 |\n')
                fFaces.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
                fFaces.write('|  \\\\    /   O peration     | Version:  plus                                  |\n')
                fFaces.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
                fFaces.write('|    \\\\/     M anipulation  |                                                 |\n')
                fFaces.write('\*---------------------------------------------------------------------------*/\n')
                fFaces.write('FoamFile\n')
                fFaces.write('{\n')
                fFaces.write('    version     2.0;\n')
                fFaces.write('    format      ascii;\n')
                fFaces.write('    class       faceList;\n')
                fFaces.write('    location    "%s";\n'%meshDir)
                fFaces.write('    object      faces;\n')
                fFaces.write('}\n')
                fFaces.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
                fFaces.write('\n')
                
                counterI = 0
                fFaces.write('%d\n'%len(faceSizes))
                fFaces.write('(\n')
                for i in range(len(faceSizes)):
                    fFaces.write('%d('%faceSizes[i])
                    for j in range(faceSizes[i]):
                        fFaces.write(' %d '%conn[counterI])
                        counterI+=1
                    fFaces.write(')\n')
                fFaces.write(')\n')
                fFaces.close()
                
                ################ write owner
                # note we don't actually need owner information for the surface mesh, so we simply assign zeros here
                fOwner = open(os.path.join(meshDir,'owner'),'w')
                # write the file header
                fOwner.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
                fOwner.write('| =========                 |                                                 |\n')
                fOwner.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
                fOwner.write('|  \\\\    /   O peration     | Version:  plus                                  |\n')
                fOwner.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
                fOwner.write('|    \\\\/     M anipulation  |                                                 |\n')
                fOwner.write('\*---------------------------------------------------------------------------*/\n')
                fOwner.write('FoamFile\n')
                fOwner.write('{\n')
                fOwner.write('    version     2.0;\n')
                fOwner.write('    format      ascii;\n')
                fOwner.write('    class       labelList;\n')
                fOwner.write('    location    "%s";\n'%meshDir)
                fOwner.write('    object      owner;\n')
                fOwner.write('}\n')
                fOwner.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
                fOwner.write('\n')
                
                fOwner.write('%d\n'%len(faceSizes))
                fOwner.write('(\n')
                for i in range(len(faceSizes)):
                    fOwner.write('0\n')
                fOwner.write(')\n')        
                fOwner.close()
                
                
                ################ write neighbour
                # note we don't actually need neighbour information for the surface mesh, so we simply assign zeros here
                fNeighbour = open(os.path.join(meshDir,'neighbour'),'w')
                # write the file header
                fNeighbour.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
                fNeighbour.write('| =========                 |                                                 |\n')
                fNeighbour.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
                fNeighbour.write('|  \\\\    /   O peration     | Version:  plus                                  |\n')
                fNeighbour.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
                fNeighbour.write('|    \\\\/     M anipulation  |                                                 |\n')
                fNeighbour.write('\*---------------------------------------------------------------------------*/\n')
                fNeighbour.write('FoamFile\n')
                fNeighbour.write('{\n')
                fNeighbour.write('    version     2.0;\n')
                fNeighbour.write('    format      ascii;\n')
                fNeighbour.write('    class       labelList;\n')
                fNeighbour.write('    location    "%s";\n'%meshDir)
                fNeighbour.write('    object      neighbour;\n')
                fNeighbour.write('}\n')
                fNeighbour.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
                fNeighbour.write('\n')
            
                fNeighbour.write('%d\n'%len(faceSizes))
                fNeighbour.write('(\n')
                for i in range(len(faceSizes)):
                    fNeighbour.write('0\n')
                fNeighbour.write(')\n')       
                fNeighbour.close()
                
                
                ################ write boundary
                fBoundary = open(os.path.join(meshDir,'boundary'),'w')
                # write the file header
                fBoundary.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
                fBoundary.write('| =========                 |                                                 |\n')
                fBoundary.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
                fBoundary.write('|  \\\\    /   O peration     | Version:  plus                                  |\n')
                fBoundary.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
                fBoundary.write('|    \\\\/     M anipulation  |                                                 |\n')
                fBoundary.write('\*---------------------------------------------------------------------------*/\n')
                fBoundary.write('FoamFile\n')
                fBoundary.write('{\n')
                fBoundary.write('    version     2.0;\n')
                fBoundary.write('    format      ascii;\n')
                fBoundary.write('    class       polyBoundaryMesh;\n')
                fBoundary.write('    location    "%s";\n'%meshDir)
                fBoundary.write('    object      boundary;\n')
                fBoundary.write('}\n')
                fBoundary.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
                fBoundary.write('\n')
                
                fBoundary.write('1\n')
                fBoundary.write('(\n')
                fBoundary.write('    %s\n'%groupName)
                fBoundary.write('    {\n')
                fBoundary.write('        type       wall;\n')
                fBoundary.write('        nFaces     %d;\n'%len(faceSizes))
                fBoundary.write('        startFace  0;\n')
                fBoundary.write('    }\n')
                fBoundary.write(')\n')
                
                fBoundary.close()
                
                
                ################ write sensitivity
                fSens = open(os.path.join(sensDir,'sensitivity4%s'%func),'w')
                # write the file header
                fSens.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
                fSens.write('| =========                 |                                                 |\n')
                fSens.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
                fSens.write('|  \\\\    /   O peration     | Version:  plus                                  |\n')
                fSens.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
                fSens.write('|    \\\\/     M anipulation  |                                                 |\n')
                fSens.write('\*---------------------------------------------------------------------------*/\n')
                fSens.write('FoamFile\n')
                fSens.write('{\n')
                fSens.write('    version     2.0;\n')
                fSens.write('    format      ascii;\n')
                fSens.write('    class       volVectorField;\n')
                fSens.write('    location    "%s";\n'%sensDir)
                fSens.write('    object      sensitivity4%s;\n'%func)
                fSens.write('}\n')
                fSens.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
                fSens.write('\n')
                
                fSens.write('dimensions      [0 0 0 0 0 0 0];\n')
                fSens.write('internalField   uniform (0 0 0);\n')
                
                counterI = 0
                fSens.write('boundaryField\n')
                fSens.write('{\n')
                fSens.write('    %s\n'%groupName)
                fSens.write('    {\n')
                fSens.write('        type  wall;\n')
                fSens.write('        value nonuniform List<vector>\n')
                fSens.write('%d\n'%len(faceSizes))
                fSens.write('(\n')
                counterI = 0
                for i in range(len(faceSizes)):
                    sensXMean=0.0
                    sensYMean=0.0
                    sensZMean=0.0
                    for j in range(faceSizes[i]):
                        idxI = conn[counterI]
                        sensXMean += dIdx[ptSetName][idxI][0]
                        sensYMean += dIdx[ptSetName][idxI][1]
                        sensZMean += dIdx[ptSetName][idxI][2]
                        counterI+=1
                    sensXMean /= faceSizes[i]
                    sensYMean /= faceSizes[i]
                    sensZMean /= faceSizes[i]
                    fSens.write('(%f %f %f)\n'%(sensXMean,sensYMean,sensZMean))   
                fSens.write(')\n')
                fSens.write(';\n')
                fSens.write('    }\n')
                fSens.write('}\n')
                fSens.close()
            
            else:
                raise Error('fileType not valid! Options are: tecplot or openfoam') 

       
        return
        
    def writeSurfaceSensitivityMapFFD(self,evalFuncs,groupName=None):
        """
        Write the surface sensitivities to a OpenFOAM file.
        Parameters
        ----------
        
        evalFuncs : iterable object containing strings
            The functions the user wants the derivatives of
 
        groupName : str
            The group the user wants the derivatives of
 
        Examples
        --------
        >>> CFDsolver.writeSurfaceSensitivityMap( ['CD','CL'])
        """
                
        if self.comm.rank==0:
            print("Writing Surface Sensitivity Map.")
        
        if self.mesh is None:      
            raise Error('Cannot write the sensitivity map without a mesh object')
            
        # Check for evalFuncs, this should eventually have some kind of default
        if evalFuncs is None:
            print 'evalFuncs not set, exiting...'
            sys.exit(0)
            
        if groupName == None:
            groupName = self.designFamilyGroup
        
        # setup ptSetName and add pointSet
        self.ptSetName = self.getPointSetName('dummy')
        ptSetName = self.ptSetName
        if self.DVGeo is not None:
            if not ptSetName in self.DVGeo.points:
                coords0 = self.mapVector(self.coords0, self.allFamilies,groupName)
                self.DVGeo.addPointSet(coords0,ptSetName)
                self.pointsSet = True

        dIdXs0 = np.zeros_like(self.coords0,self.dtype)
        dIdXs0 = self.mapVector(dIdXs0,self.allFamilies,groupName)
        
        # generate the folder to save the sensitivty files
        if self.comm.size == 1:
            meshDir = '99999/polyMesh/'
            sensDir = '99999/'          
            if not os.path.exists(meshDir):
                os.makedirs(meshDir)
        else:
            meshDir = 'processor%d/99999/polyMesh/'%self.comm.rank
            sensDir = 'processor%d/99999/'%self.comm.rank
            if self.comm.rank ==0:
                for i in range(self.comm.size):
                    dir1 = 'processor%d/99999/polyMesh/'%i
                    if not os.path.exists(dir1):
                        os.makedirs(dir1)  
            self.comm.Barrier()      
             
        # Do the functions one at a time:
        for func in evalFuncs:
            if self.comm.rank == 0:
                print('Reading adjoint: %s'%func)
 
            key ='%s'% func
            
            # calc the surface sens dIdXs based on the dIdXFFD
            if self.getOption('computeffdderivs'):
                dIdXFFD = self._readFFDSensitivity(func)
                dIdXs = self.DVGeo.totalSensitivityProd(dIdXFFD,ptSetName=ptSetName,comm=self.comm).reshape(dIdXs0.shape) 
            else:
                raise Error('computeffdderivs not set') 
                
            # Obtain the points and connectivity
            pts = self.getSurfaceCoordinates(groupName)
            conn, faceSizes = self.getSurfaceConnectivity(groupName)
            conn = np.array(conn).flatten()
            sens = np.zeros((len(faceSizes),3))
                
            ############### write points
            fPoints = open(os.path.join(meshDir,'points'),'w')
            # write the file header
            fPoints.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
            fPoints.write('| =========                 |                                                 |\n')
            fPoints.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            fPoints.write('|  \\\\    /   O peration     | Version:  plus                                  |\n')
            fPoints.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
            fPoints.write('|    \\\\/     M anipulation  |                                                 |\n')
            fPoints.write('\*---------------------------------------------------------------------------*/\n')
            fPoints.write('FoamFile\n')
            fPoints.write('{\n')
            fPoints.write('    version     2.0;\n')
            fPoints.write('    format      ascii;\n')
            fPoints.write('    class       vectorField;\n')
            fPoints.write('    location    "%s";\n'%meshDir)
            fPoints.write('    object      points;\n')
            fPoints.write('}\n')
            fPoints.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            fPoints.write('\n')
            
            fPoints.write('%d\n'%len(pts))
            fPoints.write('(\n')
            for i in range(len(pts)):
                fPoints.write('(%f %f %f)\n'%(float(pts[i][0]),float(pts[i][1]),float(pts[i][2])))
            fPoints.write(')\n')
            fPoints.close()
            
            ################ write faces
            fFaces = open(os.path.join(meshDir,'faces'),'w')
            # write the file header
            fFaces.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
            fFaces.write('| =========                 |                                                 |\n')
            fFaces.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            fFaces.write('|  \\\\    /   O peration     | Version:  plus                                  |\n')
            fFaces.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
            fFaces.write('|    \\\\/     M anipulation  |                                                 |\n')
            fFaces.write('\*---------------------------------------------------------------------------*/\n')
            fFaces.write('FoamFile\n')
            fFaces.write('{\n')
            fFaces.write('    version     2.0;\n')
            fFaces.write('    format      ascii;\n')
            fFaces.write('    class       faceList;\n')
            fFaces.write('    location    "%s";\n'%meshDir)
            fFaces.write('    object      faces;\n')
            fFaces.write('}\n')
            fFaces.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            fFaces.write('\n')
            
            counterI = 0
            fFaces.write('%d\n'%len(faceSizes))
            fFaces.write('(\n')
            for i in range(len(faceSizes)):
                fFaces.write('%d('%faceSizes[i])
                for j in range(faceSizes[i]):
                    fFaces.write(' %d '%conn[counterI])
                    counterI+=1
                fFaces.write(')\n')
            fFaces.write(')\n')
            fFaces.close()
            
            ################ write owner
            # note we don't actually need owner information for the surface mesh, so we simply assign zeros here
            fOwner = open(os.path.join(meshDir,'owner'),'w')
            # write the file header
            fOwner.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
            fOwner.write('| =========                 |                                                 |\n')
            fOwner.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            fOwner.write('|  \\\\    /   O peration     | Version:  plus                                  |\n')
            fOwner.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
            fOwner.write('|    \\\\/     M anipulation  |                                                 |\n')
            fOwner.write('\*---------------------------------------------------------------------------*/\n')
            fOwner.write('FoamFile\n')
            fOwner.write('{\n')
            fOwner.write('    version     2.0;\n')
            fOwner.write('    format      ascii;\n')
            fOwner.write('    class       labelList;\n')
            fOwner.write('    location    "%s";\n'%meshDir)
            fOwner.write('    object      owner;\n')
            fOwner.write('}\n')
            fOwner.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            fOwner.write('\n')
            
            fOwner.write('%d\n'%len(faceSizes))
            fOwner.write('(\n')
            for i in range(len(faceSizes)):
                fOwner.write('0\n')
            fOwner.write(')\n')        
            fOwner.close()
            
            
            ################ write neighbour
            # note we don't actually need neighbour information for the surface mesh, so we simply assign zeros here
            fNeighbour = open(os.path.join(meshDir,'neighbour'),'w')
            # write the file header
            fNeighbour.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
            fNeighbour.write('| =========                 |                                                 |\n')
            fNeighbour.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            fNeighbour.write('|  \\\\    /   O peration     | Version:  plus                                  |\n')
            fNeighbour.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
            fNeighbour.write('|    \\\\/     M anipulation  |                                                 |\n')
            fNeighbour.write('\*---------------------------------------------------------------------------*/\n')
            fNeighbour.write('FoamFile\n')
            fNeighbour.write('{\n')
            fNeighbour.write('    version     2.0;\n')
            fNeighbour.write('    format      ascii;\n')
            fNeighbour.write('    class       labelList;\n')
            fNeighbour.write('    location    "%s";\n'%meshDir)
            fNeighbour.write('    object      neighbour;\n')
            fNeighbour.write('}\n')
            fNeighbour.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            fNeighbour.write('\n')
        
            fNeighbour.write('%d\n'%len(faceSizes))
            fNeighbour.write('(\n')
            for i in range(len(faceSizes)):
                fNeighbour.write('0\n')
            fNeighbour.write(')\n')       
            fNeighbour.close()
            
            
            ################ write boundary
            fBoundary = open(os.path.join(meshDir,'boundary'),'w')
            # write the file header
            fBoundary.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
            fBoundary.write('| =========                 |                                                 |\n')
            fBoundary.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            fBoundary.write('|  \\\\    /   O peration     | Version:  plus                                  |\n')
            fBoundary.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
            fBoundary.write('|    \\\\/     M anipulation  |                                                 |\n')
            fBoundary.write('\*---------------------------------------------------------------------------*/\n')
            fBoundary.write('FoamFile\n')
            fBoundary.write('{\n')
            fBoundary.write('    version     2.0;\n')
            fBoundary.write('    format      ascii;\n')
            fBoundary.write('    class       polyBoundaryMesh;\n')
            fBoundary.write('    location    "%s";\n'%meshDir)
            fBoundary.write('    object      boundary;\n')
            fBoundary.write('}\n')
            fBoundary.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            fBoundary.write('\n')
            
            fBoundary.write('1\n')
            fBoundary.write('(\n')
            fBoundary.write('    %s\n'%groupName)
            fBoundary.write('    {\n')
            fBoundary.write('        type       wall;\n')
            fBoundary.write('        nFaces     %d;\n'%len(faceSizes))
            fBoundary.write('        startFace  0;\n')
            fBoundary.write('    }\n')
            fBoundary.write(')\n')
            
            fBoundary.close()
            
            
            ################ write sensitivity
            fSens = open(os.path.join(sensDir,'sensitivity4%s'%func),'w')
            # write the file header
            fSens.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
            fSens.write('| =========                 |                                                 |\n')
            fSens.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
            fSens.write('|  \\\\    /   O peration     | Version:  plus                                  |\n')
            fSens.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
            fSens.write('|    \\\\/     M anipulation  |                                                 |\n')
            fSens.write('\*---------------------------------------------------------------------------*/\n')
            fSens.write('FoamFile\n')
            fSens.write('{\n')
            fSens.write('    version     2.0;\n')
            fSens.write('    format      ascii;\n')
            fSens.write('    class       volVectorField;\n')
            fSens.write('    location    "%s";\n'%sensDir)
            fSens.write('    object      sensitivity4%s;\n'%func)
            fSens.write('}\n')
            fSens.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            fSens.write('\n')
            
            fSens.write('dimensions      [0 0 0 0 0 0 0];\n')
            fSens.write('internalField   uniform (0 0 0);\n')
            
            counterI = 0
            fSens.write('boundaryField\n')
            fSens.write('{\n')
            fSens.write('    %s\n'%groupName)
            fSens.write('    {\n')
            fSens.write('        type  wall;\n')
            fSens.write('        value nonuniform List<vector>\n')
            fSens.write('%d\n'%len(faceSizes))
            fSens.write('(\n')
            counterI = 0
            for i in range(len(faceSizes)):
                sensXMean=0.0
                sensYMean=0.0
                sensZMean=0.0
                for j in range(faceSizes[i]):
                    idxI = conn[counterI]
                    sensXMean += dIdXs[idxI][0]
                    sensYMean += dIdXs[idxI][1]
                    sensZMean += dIdXs[idxI][2]
                    counterI+=1
                sensXMean /= faceSizes[i]
                sensYMean /= faceSizes[i]
                sensZMean /= faceSizes[i]
                fSens.write('(%f %f %f)\n'%(sensXMean,sensYMean,sensZMean))   
            fSens.write(')\n')
            fSens.write(';\n')
            fSens.write('    }\n')
            fSens.write('}\n')
            fSens.close()
            
            
        return

# general routines for the foam file handling

    def _countCells(self):
        '''
        loop over the owner and neighbour data and take the max and min 
        cell value on each proc.
        Actually just taking 1 more than the max owner index. Should we check the max
        neighbour index as well?
        '''

        # get the min and max owner index
        minIndOwner = self.owners.min()
        maxIndOwner = self.owners.max()

        # get the min and max neighbour index
        minIndNeighbour = self.neighbours.min()
        maxIndNeighbour = self.neighbours.max()

        # if we only do maxIndOwner-minIndOwner, we will miss some cells since 
        # one face can be owned by two different cells!
        # this is not a final solution.
        nCellsOwner = max( maxIndOwner, maxIndNeighbour) - min(minIndOwner,minIndNeighbour)
       
        # if not nCellsOwner == nCellsNeighbour:
        #     raise Error("Owner and neighbour cell counts don't match."
        #                 "Owner: %d, Neighbour: %d"%(nCellsOwner,nCellsNeighbour))

        nCells = nCellsOwner+1
        #nCells = self.owners.max()+1
        return nCells


    def _writeTurbulencePropertiesFile(self):
        '''
        Write out the turbulenceProperties.
        
        This will overwrite whateverfile is present so that the solve is completed
        with the currently specified options.
        '''
        if self.comm.rank==0:
            # Open the options file for writing
            workingDirectory = os.getcwd()        
            constDir = 'constant'
            varDir = os.path.join(workingDirectory,constDir)
            fileName = 'turbulenceProperties'
            fileLoc = os.path.join(varDir, fileName)
            f = open(fileLoc, 'w')

            # write the file header
            f.write('/*--------------------------------*- C++ -*---------------------------------*\ \n')
            f.write('| ========                 |                                                 | \n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n')
            f.write('|  \\    /   O peration     | Version:  v1812                                 | \n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.com                      | \n')
            f.write('|    \\/     M anipulation  |                                                 | \n')
            f.write('\*--------------------------------------------------------------------------*/ \n')
            f.write('FoamFile\n')
            f.write('{\n')
            f.write('    version     2.0;\n')
            f.write('    format      ascii;\n')
            f.write('    class       dictionary;\n')
            f.write('    location    "%s";\n'%constDir)
            f.write('    object      %s;\n'%fileName)
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')
            
            rasmodel = self.getOption('rasmodel')
            rasModelParameters=self.getOption('rasmodelparameters')
                
            f.write('simulationType RAS;\n')
            f.write('RAS \n')
            f.write('{ \n')
            f.write('    RASModel             %s;\n'%rasmodel)
            f.write('\n')
            f.write('    turbulence           on;\n')
            f.write('\n')
            f.write('    printCoeffs          on;\n')
            for key in rasModelParameters:
                f.write('    %-20s %s;\n'%(key,rasModelParameters[key]))
            if self.getOption('flowcondition') == 'Compressible':
                if not 'Prt' in  self.getOption('thermoproperties').keys():
                    raise Error('Prt not found in thermoproperties')
                else:
                    f.write('    Prt                  %f;\n'%self.getOption('thermoproperties')['Prt'])
            f.write('} \n')
            f.write('\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

            f.close()
        return
        
    def _writeFvSchemesFile(self):
        '''
        Write out the fvSchemes file.
        
        This will overwrite whateverfile is present so that the solve is completed
        with the currently specified options.
        '''
        if self.comm.rank==0:
            # Open the options file for writing
            workingDirectory = os.getcwd()  
            sysDir = 'system'
            varDir = os.path.join(workingDirectory,sysDir)
            fileName = 'fvSchemes'
            fileLoc = os.path.join(varDir, fileName)
            f = open(fileLoc, 'w')

            # write the file header
            f.write('/*--------------------------------*- C++ -*---------------------------------*\ \n')
            f.write('| ========                 |                                                 | \n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n')
            f.write('|  \\    /   O peration     | Version:  v1812                                 | \n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.com                      | \n')
            f.write('|    \\/     M anipulation  |                                                 | \n')
            f.write('\*--------------------------------------------------------------------------*/ \n')
            f.write('FoamFile\n')
            f.write('{\n')
            f.write('    version     2.0;\n')
            f.write('    format      ascii;\n')
            f.write('    class       dictionary;\n')
            f.write('    location    "%s";\n'%sysDir)
            f.write('    object      %s;\n'%fileName)
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')

            d2dt2Schemes = self.getOption('d2dt2schemes')
            f.write('d2dt2Schemes \n')
            f.write('{\n')
            f.write('    %-50s  %s;\n'%('default',d2dt2Schemes['default']))
            for key in d2dt2Schemes:
                if not key=='default':
                    f.write('    %-50s  %s;\n'%(key,d2dt2Schemes[key]))
            f.write('}\n')
            f.write('\n')

            ddtSchemes = self.getOption('ddtschemes')
            f.write('ddtSchemes \n')
            f.write('{\n')
            f.write('    %-50s  %s;\n'%('default',ddtSchemes['default']))
            for key in ddtSchemes:
                if not key=='default':
                    f.write('    %-50s  %s;\n'%(key,ddtSchemes[key]))
            f.write('}\n')
            f.write('\n')

            gradSchemes = self.getOption('gradschemes')
            f.write('gradSchemes\n')
            f.write('{\n')
            f.write('    %-50s  %s;\n'%('default',gradSchemes['default']))
            for key in gradSchemes:
                if not key=='default':
                    f.write('    %-50s  %s;\n'%(key,gradSchemes[key]))
            f.write('}\n')
            f.write('\n')

            divSchemes = self.getOption('divschemes')
            f.write('divSchemes\n')
            f.write('{\n')
            f.write('    %-50s  %s;\n'%('default',divSchemes['default']))
            for key in divSchemes:
                if not key=='default':
                    f.write('    %-50s  %s;\n'%(key,divSchemes[key]))
            f.write('}\n')
            f.write('\n')

            interpolationSchemes = self.getOption('interpolationschemes')
            f.write('interpolationSchemes\n')
            f.write('{\n')
            f.write('    %-50s  %s;\n'%('default',interpolationSchemes['default']))
            for key in interpolationSchemes:
                if not key=='default':
                    f.write('    %-50s  %s;\n'%(key,interpolationSchemes[key]))
            f.write('}\n')
            f.write('\n')

            laplacianSchemes = self.getOption('laplacianschemes')
            f.write('laplacianSchemes\n')
            f.write('{\n')
            f.write('    %-50s  %s;\n'%('default',laplacianSchemes['default']))
            for key in laplacianSchemes:
                if not key=='default':
                    f.write('    %-50s  %s;\n'%(key,laplacianSchemes[key]))
            f.write('}\n')
            f.write('\n')

            snGradSchemes = self.getOption('snGradSchemes')
            f.write('snGradSchemes\n')
            f.write('{\n')
            f.write('    %-50s  %s;\n'%('default',snGradSchemes['default']))
            for key in snGradSchemes:
                if not key=='default':
                    f.write('    %-50s  %s;\n'%(key,snGradSchemes[key]))
            f.write('}\n')
            f.write('\n')

            fluxRequired = self.getOption('fluxrequired')
            f.write('fluxRequired\n')
            f.write('{\n')
            f.write('    %-50s  no;\n'%'default')
            for key in fluxRequired:
                    f.write('    %s;\n'%key)
            f.write('}\n')
            f.write('\n')
            f.write('wallDist\n')
            f.write('{\n')
            f.write('    %-50s  %s;\n'%('method',self.getOption('walldistmethod')))
            f.write('}\n')
            f.write('\n')
            
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

            f.close()
        return
   
    def _writeFvSolutionFile(self):
        '''
        Write out the fvSolution file.
        
        This will overwrite whateverfile is present so that the solve is completed
        with the currently specified options.
        '''
        if self.comm.rank==0:
            # Open the options file for writing
            workingDirectory = os.getcwd()  
            sysDir = 'system'
            varDir = os.path.join(workingDirectory,sysDir)
            fileName = 'fvSolution'
            fileLoc = os.path.join(varDir, fileName)
            f = open(fileLoc, 'w')
            
            residualctrl =  self.getOption('residualcontrol')
            
            # write the file header
            f.write('/*--------------------------------*- C++ -*---------------------------------*\ \n')
            f.write('| ========                 |                                                 | \n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n')
            f.write('|  \\    /   O peration     | Version:  v1812                                 | \n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.com                      | \n')
            f.write('|    \\/     M anipulation  |                                                 | \n')
            f.write('\*--------------------------------------------------------------------------*/ \n')
            f.write('FoamFile\n')
            f.write('{\n')
            f.write('    version     2.0;\n')
            f.write('    format      ascii;\n')
            f.write('    class       dictionary;\n')
            f.write('    location    "%s";\n'%sysDir)
            f.write('    object      %s;\n'%fileName)
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')

            simpleControl = self.getOption('simplecontrol')
            f.write('SIMPLE\n')
            f.write('{\n')
            for key in simpleControl.keys():
                f.write('    %-30s     %s;\n'%(key,simpleControl[key]))
            f.write('    residualControl\n')
            f.write('    {\n')
            f.write('        U                              %e;\n'%residualctrl )
            f.write('        nuTilda                        %e;\n'%residualctrl )
            f.write('        p                              %e;\n'%residualctrl )
            f.write('        e                              %e;\n'%residualctrl )
            f.write('        h                              %e;\n'%residualctrl )
            f.write('        p_rgh                          %e;\n'%residualctrl )
            f.write('        k                              %e;\n'%residualctrl )
            f.write('        omega                          %e;\n'%residualctrl )
            f.write('        epsilon                        %e;\n'%residualctrl )
            f.write('        T                              %e;\n'%residualctrl )
            f.write('        rho                            %e;\n'%residualctrl )
            f.write('    }\n')
            f.write('}\n')
            f.write('\n')

            pisoControl = self.getOption('pisocontrol')
            f.write('PISO\n')
            f.write('{\n')
            for key in pisoControl.keys():
                f.write('    %-30s     %s;\n'%(key,pisoControl[key]))
            f.write('}\n')
            f.write('\n')

            fvSolvers = self.getOption('fvsolvers')
            f.write('solvers\n')
            f.write('{\n')
            for solver in fvSolvers:
                f.write('    %s\n'%solver)
                f.write('    {\n')
                for key in fvSolvers[solver]:
                    f.write('        %-30s %s;\n'%(key,fvSolvers[solver][key]))
                f.write('    }\n')
            f.write('}\n')
            
            f.write('\n')
            relaxFactors = self.getOption('fvrelaxfactors')
            f.write('relaxationFactors\n')
            f.write('{\n')
            f.write('    fields\n')
            f.write('    {\n')
            for key1 in relaxFactors['fields']:
                f.write('        %-20s           %.2f;\n'%(key1,relaxFactors['fields'][key1]))
            f.write('    }\n')
            f.write('    equations\n')
            f.write('    {\n')
            for key2 in relaxFactors['equations']:
                f.write('        %-20s           %.2f;\n'%(key2,relaxFactors['equations'][key2]))
            f.write('    }\n')
            f.write('\n')
            f.write('}')
            f.write('\n')
            f.write('potentialFlow\n')
            f.write('{\n')
            f.write('    nNonOrthogonalCorrectors           20;\n')
            f.write('    PhiRefCell                         0;\n')
            f.write('    PhiRefValue                        0;\n')
            f.write('}\n')
            f.write('\n')
            f.write('stressAnalysis\n')
            f.write('{\n')
            f.write('    compactNormalStress                yes;\n')
            f.write('    nCorrectors                        1;\n')
            f.write('    D                                  1e-20;\n')
            f.write('}\n')
            f.write('\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

            f.close()
        return   
        
    def _writeDecomposeParDictFile(self):
        '''
        Write out the decomposeParDict file.
        
        This will overwrite whateverfile is present so that the solve is completed
        with the currently specified options.
        '''
        if self.comm.rank==0:
            # Open the options file for writing
            workingDirectory = os.getcwd()  
            sysDir = 'system'
            varDir = os.path.join(workingDirectory,sysDir)
            fileName = 'decomposeParDict'
            fileLoc = os.path.join(varDir, fileName)
            f = open(fileLoc, 'w')

            # write the file header
            f.write('/*--------------------------------*- C++ -*---------------------------------*\ \n')
            f.write('| ========                 |                                                 | \n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n')
            f.write('|  \\    /   O peration     | Version:  v1812                                 | \n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.com                      | \n')
            f.write('|    \\/     M anipulation  |                                                 | \n')
            f.write('\*--------------------------------------------------------------------------*/ \n')
            f.write('FoamFile\n')
            f.write('{\n')
            f.write('    version     2.0;\n')
            f.write('    format      ascii;\n')
            f.write('    class       dictionary;\n')
            f.write('    location    "%s";\n'%sysDir)
            f.write('    object      %s;\n'%fileName)
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')

            decomDict=self.getOption('decomposepardict')

            f.write('numberOfSubdomains     %d;\n'%self.nProcs)
            f.write('\n')
            f.write('method             %s;\n'%decomDict['method'])
            f.write('\n')
            f.write('simpleCoeffs \n')
            f.write('{ \n')
            f.write('    n              %s; \n'%decomDict['simpleCoeffs']['n'])
            f.write('    delta          %s; \n'%decomDict['simpleCoeffs']['delta'])
            f.write('} \n')
            f.write('\n')
            f.write('distributed         false;\n')
            f.write('\n')
            f.write('roots();\n')
            f.write('\n')
            if not len(self.getOption('preservepatches'))==0:
                f.write('preservePatches (')
                for patch in self.getOption('preservepatches'):
                    f.write('%s '%patch)
                f.write(');\n')
            if not len(self.getOption('singleprocessorfacesets'))==0:
                f.write('constraints{ singleProcessorFaceSets{ type singleProcessorFaceSets; sets (')
                for patch in self.getOption('singleprocessorfacesets'):
                    f.write(' (%s -1) '%patch)
                f.write('); } }\n')
            f.write('\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

            f.close()
        return
        
        
    def _writeTransportPropertiesFile(self):
        '''
        Write out the transportProperties file.
        
        This will overwrite whateverfile is present so that the solve is completed
        with the currently specified options.
        '''
        if self.comm.rank==0:
            # Open the options file for writing
            workingDirectory = os.getcwd()  
            sysDir = 'constant'
            varDir = os.path.join(workingDirectory,sysDir)
            fileName = 'transportProperties'
            fileLoc = os.path.join(varDir, fileName)
            f = open(fileLoc, 'w')

            # write the file header
            f.write('/*--------------------------------*- C++ -*---------------------------------*\ \n')
            f.write('| ========                 |                                                 | \n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n')
            f.write('|  \\    /   O peration     | Version:  v1812                                 | \n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.com                      | \n')
            f.write('|    \\/     M anipulation  |                                                 | \n')
            f.write('\*--------------------------------------------------------------------------*/ \n')
            f.write('FoamFile\n')
            f.write('{\n')
            f.write('    version     2.0;\n')
            f.write('    format      ascii;\n')
            f.write('    class       dictionary;\n')
            f.write('    location    "%s";\n'%sysDir)
            f.write('    object      %s;\n'%fileName)
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')

            transP = self.getOption('transproperties')
            f.write('transportModel Newtonian;\n')
            f.write('\n')
            for key in transP.keys():
                if not isinstance(transP[key], (list,)):
                    f.write('%s %16.16f;\n'%(key,transP[key]))
            f.write('\n')
            
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

            f.close()
        return

    def _writeMRFPropertiesFile(self):
        '''
        Write out the MRFProperties file.
        
        This will overwrite whateverfile is present so that the solve is completed
        with the currently specified options.
        '''
        if self.comm.rank==0:
            # Open the options file for writing
            workingDirectory = os.getcwd()  
            sysDir = 'constant'
            varDir = os.path.join(workingDirectory,sysDir)
            fileName = 'MRFProperties'
            fileLoc = os.path.join(varDir, fileName)
            f = open(fileLoc, 'w')

            # write the file header
            f.write('/*--------------------------------*- C++ -*---------------------------------*\ \n')
            f.write('| ========                 |                                                 | \n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n')
            f.write('|  \\    /   O peration     | Version:  v1812                                 | \n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.com                      | \n')
            f.write('|    \\/     M anipulation  |                                                 | \n')
            f.write('\*--------------------------------------------------------------------------*/ \n')
            f.write('FoamFile\n')
            f.write('{\n')
            f.write('    version     2.0;\n')
            f.write('    format      ascii;\n')
            f.write('    class       dictionary;\n')
            f.write('    location    "%s";\n'%sysDir)
            f.write('    object      %s;\n'%fileName)
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')

            mrfP = self.getOption('mrfproperties')
            f.write('MRF\n')
            f.write('{\n')
            f.write('    active %s;\n'%mrfP['active'])
            f.write('    selectoinMode %s;\n'%mrfP['selectionmode'])
            f.write('    cellZone %s;\n'%mrfP['cellzone'])
            f.write('    nonRotatingPatches (')
            for patch in mrfP['nonrotatingpatches']:
                f.write('%s '%patch)
            f.write(');\n')
            f.write('    axis (%f %f %f);\n'%(mrfP['axis'][0],mrfP['axis'][1],mrfP['axis'][2]))
            f.write('    origin (%f %f %f);\n'%(mrfP['origin'][0],mrfP['origin'][1],mrfP['origin'][2]))
            f.write('    omega %f;\n'%mrfP['omega'])
            f.write('}\n')
            f.write('\n')
            
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

            f.close()
        return

    def _writeRadiationPropertiesFile(self):
        '''
        Write out the radiationProperties file.
        
        This will overwrite whateverfile is present so that the solve is completed
        with the currently specified options.
        '''
        if self.comm.rank==0:
            # Open the options file for writing
            workingDirectory = os.getcwd()  
            sysDir = 'constant'
            varDir = os.path.join(workingDirectory,sysDir)
            fileName = 'radiationProperties'
            fileLoc = os.path.join(varDir, fileName)
            f = open(fileLoc, 'w')

            # write the file header
            f.write('/*--------------------------------*- C++ -*---------------------------------*\ \n')
            f.write('| ========                 |                                                 | \n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n')
            f.write('|  \\    /   O peration     | Version:  v1812                                 | \n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.com                      | \n')
            f.write('|    \\/     M anipulation  |                                                 | \n')
            f.write('\*--------------------------------------------------------------------------*/ \n')
            f.write('FoamFile\n')
            f.write('{\n')
            f.write('    version     2.0;\n')
            f.write('    format      ascii;\n')
            f.write('    class       dictionary;\n')
            f.write('    location    "%s";\n'%sysDir)
            f.write('    object      %s;\n'%fileName)
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')

            radiationP = self.getOption('radiationproperties')
            f.write('radiation                                    %s;\n'%radiationP['radiation'])
            f.write('radiationModel                               %s;\n'%radiationP['radiationModel'])
            f.write('solverFreq                                   %d;\n'%radiationP['solverFreq'])
            f.write('absorptionEmissionModel                      constantAbsorptionEmission;\n')
            f.write('constantAbsorptionEmissionCoeffs\n')
            f.write('{\n')
            f.write('  absorptivity absorptivity [0 -1 0 0 0 0 0] %f;\n'%radiationP['absorptivity'])
            f.write('  emissivity   emissivity   [0 -1 0 0 0 0 0] %f;\n'%radiationP['emissivity'])
            f.write('  E            E           [1 -1 -3 0 0 0 0] %f;\n'%radiationP['E'])
            f.write('}\n')
            f.write('scatterModel                                 none;\n')
            f.write('sootModel                                    none;\n')
            f.write('transmissivityModel                          none;\n')
            f.write('\n')
            
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

            f.close()
        return
    
    def _writeGFile(self):
        '''
        Write out the g file.
        
        This will overwrite whateverfile is present so that the solve is completed
        with the currently specified options.
        '''
        if self.comm.rank==0:
            # Open the options file for writing
            workingDirectory = os.getcwd()  
            sysDir = 'constant'
            varDir = os.path.join(workingDirectory,sysDir)
            fileName = 'g'
            fileLoc = os.path.join(varDir, fileName)
            f = open(fileLoc, 'w')

            # write the file header
            f.write('/*--------------------------------*- C++ -*---------------------------------*\ \n')
            f.write('| ========                 |                                                 | \n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n')
            f.write('|  \\    /   O peration     | Version:  v1812                                 | \n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.com                      | \n')
            f.write('|    \\/     M anipulation  |                                                 | \n')
            f.write('\*--------------------------------------------------------------------------*/ \n')
            f.write('FoamFile\n')
            f.write('{\n')
            f.write('    version     2.0;\n')
            f.write('    format      ascii;\n')
            f.write('    class       uniformDimensionedVectorField;\n')
            f.write('    location    "%s";\n'%sysDir)
            f.write('    object      %s;\n'%fileName)
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')

            transP = self.getOption('transproperties')
            f.write('dimensions      [0 1 -2 0 0 0 0];\n')
            f.write('\n')
            if 'g' in transP.keys():
                f.write('value           (%f %f %f);\n'%(transP['g'][0],transP['g'][1],transP['g'][2]))
            else:
                f.write('value           (0 0 0);\n')
            f.write('\n')
            
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

            f.close()
        return
    
    def _writeThermophysicalPropertiesFile(self):
        '''
        Write out the thermophysicalProperties file.
        
        This will overwrite whateverfile is present so that the solve is completed
        with the currently specified options.
        '''
        if self.comm.rank==0:
            # Open the options file for writing
            workingDirectory = os.getcwd()  
            sysDir = 'constant'
            varDir = os.path.join(workingDirectory,sysDir)
            fileName = 'thermophysicalProperties'
            fileLoc = os.path.join(varDir, fileName)
            f = open(fileLoc, 'w')

            # write the file header
            f.write('/*--------------------------------*- C++ -*---------------------------------*\ \n')
            f.write('| ========                 |                                                 | \n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n')
            f.write('|  \\    /   O peration     | Version:  v1812                                 | \n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.com                      | \n')
            f.write('|    \\/     M anipulation  |                                                 | \n')
            f.write('\*--------------------------------------------------------------------------*/ \n')
            f.write('FoamFile\n')
            f.write('{\n')
            f.write('    version     2.0;\n')
            f.write('    format      ascii;\n')
            f.write('    class       dictionary;\n')
            f.write('    location    "%s";\n'%sysDir)
            f.write('    object      %s;\n'%fileName)
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')

            thermoP = self.getOption('thermoproperties')

            thermoType = self.getOption('thermotype')
            f.write('thermoType \n')
            f.write('{ \n')
            for key in thermoType:
                f.write('    %-20s  %s;\n'%(key,thermoType[key]))
            f.write('} \n')
            f.write('\n')

            f.write('mixture \n')
            f.write('{ \n')
            f.write('    specie \n')
            f.write('    { \n')
            f.write('        molWeight           %f; \n'%thermoP['molWeight'])
            f.write('    } \n')
            f.write('    thermodynamics \n')
            f.write('    { \n')
            f.write('        Cp                  %f; \n'%thermoP['Cp'])
            f.write('        Hf                  %f; \n'%thermoP['Hf'])
            f.write('    } \n')    
            f.write('    transport \n')    
            f.write('    { \n')    
            f.write('        mu                  %f; \n'%thermoP['mu'])
            f.write('        Pr                  %f; \n'%thermoP['Pr'])
            f.write('        TRef                %f; \n'%thermoP['TRef'])
            f.write('    } \n')
            f.write('} \n')

            f.write('\n')
            
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

            f.close()
        return
    
    def _writeThermalPropertiesFile(self):
        '''
        Write out the thermalProperties file.
        
        This will overwrite whateverfile is present so that the solve is completed
        with the currently specified options.
        '''
        if self.comm.rank==0:
            # Open the options file for writing
            workingDirectory = os.getcwd()  
            sysDir = 'constant'
            varDir = os.path.join(workingDirectory,sysDir)
            fileName = 'thermalProperties'
            fileLoc = os.path.join(varDir, fileName)
            f = open(fileLoc, 'w')

            # write the file header
            f.write('/*--------------------------------*- C++ -*---------------------------------*\ \n')
            f.write('| ========                 |                                                 | \n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n')
            f.write('|  \\    /   O peration     | Version:  v1812                                 | \n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.com                      | \n')
            f.write('|    \\/     M anipulation  |                                                 | \n')
            f.write('\*--------------------------------------------------------------------------*/ \n')
            f.write('FoamFile\n')
            f.write('{\n')
            f.write('    version     2.0;\n')
            f.write('    format      ascii;\n')
            f.write('    class       dictionary;\n')
            f.write('    location    "%s";\n'%sysDir)
            f.write('    object      %s;\n'%fileName)
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')

            CVal = self.getOption('thermalproperties')['C']
            kVal = self.getOption('thermalproperties')['k']
            alphaVal = self.getOption('thermalproperties')['alpha']
            thermalStress = self.getOption('thermalproperties')['thermalStress']


            f.write('C\n')
            f.write('{\n')
            f.write('    type        uniform;\n')
            f.write('    value       %f;\n'%CVal)
            f.write('}\n')
            f.write('\n')

            f.write('k\n')
            f.write('{\n')
            f.write('    type        uniform;\n')
            f.write('    value       %f;\n'%kVal)
            f.write('}\n')
            f.write('\n')

            f.write('alpha\n')
            f.write('{\n')
            f.write('    type        uniform;\n')
            f.write('    value       %e;\n'%alphaVal)
            f.write('}\n')
            f.write('\n')

            f.write('thermalStress   %s;\n'%thermalStress)

            f.write('\n')
            
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

            f.close()
        return
    
    def _writeMechanicalPropertiesFile(self):
        '''
        Write out the mechanicalProperties file.
        
        This will overwrite whateverfile is present so that the solve is completed
        with the currently specified options.
        '''
        if self.comm.rank==0:
            # Open the options file for writing
            workingDirectory = os.getcwd()  
            sysDir = 'constant'
            varDir = os.path.join(workingDirectory,sysDir)
            fileName = 'mechanicalProperties'
            fileLoc = os.path.join(varDir, fileName)
            f = open(fileLoc, 'w')

            # write the file header
            f.write('/*--------------------------------*- C++ -*---------------------------------*\ \n')
            f.write('| ========                 |                                                 | \n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n')
            f.write('|  \\    /   O peration     | Version:  v1812                                 | \n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.com                      | \n')
            f.write('|    \\/     M anipulation  |                                                 | \n')
            f.write('\*--------------------------------------------------------------------------*/ \n')
            f.write('FoamFile\n')
            f.write('{\n')
            f.write('    version     2.0;\n')
            f.write('    format      ascii;\n')
            f.write('    class       dictionary;\n')
            f.write('    location    "%s";\n'%sysDir)
            f.write('    object      %s;\n'%fileName)
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')

            rhoVal = self.getOption('mechanicalproperties')['rho']
            nuVal = self.getOption('mechanicalproperties')['nu']
            EVal = self.getOption('mechanicalproperties')['E']

            f.write('rho\n')
            f.write('{\n')
            f.write('    type        uniform;\n')
            f.write('    value       %f;\n'%rhoVal)
            f.write('}\n')
            f.write('\n')

            f.write('nu\n')
            f.write('{\n')
            f.write('    type        uniform;\n')
            f.write('    value       %f;\n'%nuVal)
            f.write('}\n')
            f.write('\n')

            f.write('E\n')
            f.write('{\n')
            f.write('    type        uniform;\n')
            f.write('    value       %e;\n'%EVal)
            f.write('}\n')
            f.write('\n')

            f.write('planeStress     false;\n')

            f.write('\n')
            
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

            f.close()
        return

    def _writeControlDictFile(self):
        '''
        Write out the controlDict file.
        
        This will overwrite whateverfile is present so that the solve is completed
        with the currently specified options.
        '''
        if self.comm.rank==0:
            
            dragdir = self.getOption('dragdir')
            liftdir = self.getOption('liftdir')
            cofr = self.getOption('cofr')
        
            # Open the options file for writing
            workingDirectory = os.getcwd()  
            sysDir = 'system'
            varDir = os.path.join(workingDirectory,sysDir)
            fileName = 'controlDict'
            fileLoc = os.path.join(varDir, fileName)
            f = open(fileLoc, 'w')
            
            # get force related objfuncgeoinfo
            objfuncs = self.getOption('objfuncs')
            forcepatches = []
            objfuncgeoinfo = self.getOption('objfuncgeoinfo')
            for i in range(len(objfuncs)):
                objfunc = objfuncs[i]
                if objfunc in ['CD','CL','CMX','CMY','CMZ']:
                    forcepatches = objfuncgeoinfo[i]
                    break
            
            # write the file header
            f.write('/*--------------------------------*- C++ -*---------------------------------*\ \n')
            f.write('| ========                 |                                                 | \n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n')
            f.write('|  \\    /   O peration     | Version:  v1812                                 | \n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.com                      | \n')
            f.write('|    \\/     M anipulation  |                                                 | \n')
            f.write('\*--------------------------------------------------------------------------*/ \n')
            f.write('FoamFile\n')
            f.write('{\n')
            f.write('    version     2.0;\n')
            f.write('    format      ascii;\n')
            f.write('    class       dictionary;\n')
            f.write('    location    "%s";\n'%sysDir)
            f.write('    object      %s;\n'%fileName)
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n') 
            f.write('\n')     
            f.write('libs\n') 
            f.write('(\n')
            f.write('    "libbuoyantPressureFvPatchScalarField.so" \n')
            if self.getOption('flowcondition')=="Incompressible":
                f.write('    "libDummyTurbulenceModelIncompressible.so" \n')
                f.write('    "libSpalartAllmarasFv3Incompressible.so" \n')
            elif self.getOption('flowcondition')=="Compressible":
                f.write('    "libDummyTurbulenceModelCompressible.so" \n')
                f.write('    "libSpalartAllmarasFv3Compressible.so" \n')
            f.write(');\n')
            f.write('\n')     
            f.write('application     %s;'%self.getOption('adjointsolver') )
            f.write('\n')
            f.write('startFrom       %s;'%self.solveFrom )
            f.write('\n')
            f.write('startTime       0;')
            f.write('\n')
            f.write('stopAt          endTime;')
            f.write('\n')
            f.write('endTime         %d;'%self.getOption('maxflowiters') )
            f.write('\n')
            f.write('deltaT          1;')
            f.write('\n')
            f.write('writeControl    timeStep;')
            f.write('\n')
            f.write('writeInterval   %d;'%self.getOption('writeinterval') )
            f.write('\n')
            f.write('purgeWrite      0;')
            f.write('\n')
            f.write('writeFormat     %s;'%self.getOption('writeformat') )
            f.write('\n')
            f.write('writePrecision  16;')
            f.write('\n')
            f.write('writeCompression %s;'%self.getOption('writecompress') )
            f.write('\n')
            f.write('timeFormat      general;')
            f.write('\n')
            f.write('timePrecision   16;')
            f.write('\n')
            f.write('runTimeModifiable true;')
            f.write('\n')
            f.write('\n')

            objfuncs = self.getOption('objfuncs')
            isForceObj=False
            for objFunc in objfuncs:
                if objfunc in ['CD','CL','CMX','CMY','CMZ']:
                    isForceObj=True
                    break
            if isForceObj:
                f.write('functions\n')
                f.write('{ \n')
                f.write('    forceCoeffs\n')
                f.write('    { \n')
                f.write('        type                forceCoeffs;\n')
                f.write('        libs                ("libforces.so");\n')
                f.write('        writeControl        timeStep;\n')
                f.write('        timeInterval        1;\n')
                f.write('        log                 yes;\n')
                forcepatches = ' '.join(forcepatches)
                f.write('        patches             (%s);\n'%forcepatches)
                if 'buoyant' in self.getOption('adjointsolver'):
                    f.write('        pName               p_rgh;\n')
                else:
                    f.write('        pName               p;\n')
                f.write('        UName               U;\n')
                f.write('        rho                 rhoInf;\n')
                f.write('        rhoInf              %f;\n'%self.getOption('referencevalues')['rhoRef'])
                
                f.write('        dragDir             (%12.10f %12.10f %12.10f);\n'%(dragdir[0],dragdir[1],dragdir[2]))
                f.write('        liftDir             (%12.10f %12.10f %12.10f);\n'%(liftdir[0],liftdir[1],liftdir[2]))
                f.write('        CofR                (%f %f %f);\n'%(cofr[0],cofr[1],cofr[2]))
                f.write('        pitchAxis           (0 1 0);\n')
                f.write('        magUInf             %f;\n'%self.getOption('referencevalues')['magURef'] )
                f.write('        lRef                %f;\n'%self.getOption('referencevalues')['LRef'] )
                f.write('        Aref                %f;\n'%self.getOption('referencevalues')['ARef'] )
                f.write('    } \n')
                f.write('} \n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

            f.close()
        return

    def _writeAdjointDictFile(self):
        '''
        Write out the adjointDict File.
        
        This will overwrite whateverfile is present so that the solve is completed
        with the currently specified options.
        '''
        if self.comm.rank==0:
            # Open the options file for writing
            workingDirectory = os.getcwd()        
            sysDir = 'system'
            varDir = os.path.join(workingDirectory,sysDir)
            fileName = 'adjointDict'
            fileLoc = os.path.join(varDir, fileName)
            f = open(fileLoc, 'w')

            # write the file header
            f.write('/*--------------------------------*- C++ -*---------------------------------*\ \n')
            f.write('| ========                 |                                                 | \n')
            f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n')
            f.write('|  \\    /   O peration     | Version:  v1812                                 | \n')
            f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.com                      | \n')
            f.write('|    \\/     M anipulation  |                                                 | \n')
            f.write('\*--------------------------------------------------------------------------*/ \n')
            f.write('FoamFile\n')
            f.write('{\n')
            f.write('    version     2.0;\n')
            f.write('    format      ascii;\n')
            f.write('    class       dictionary;\n')
            f.write('    location    "%s";\n'%sysDir)
            f.write('    object      %s;\n'%fileName)
            f.write('}\n')
            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.write('\n')
            
            solveAdjoint = self._checkBoolean(self.solveAdjoint)
            #options
            setFlowBCs = self._checkBoolean(self.getOption('setflowbcs'))
            useColoring = self._checkBoolean(self.getOption('usecoloring'))
            normalizeRes= self.getOption('normalizeresiduals')
            normalizeRes = ' '.join(normalizeRes)
            normalizeStates= self.getOption('normalizestates')
            normalizeStates = ' '.join(normalizeStates)
            writeMatrices =  self._checkBoolean(self.getOption('writematrices'))
            adjGMRESCalcEigen =  self._checkBoolean(self.getOption('adjgmrescalcEigen'))
            correctWallDist = self._checkBoolean(self.getOption('correctwalldist'))
            reduceResCon4JacMat = self._checkBoolean(self.getOption('reducerescon4jacmat'))
            delTurbProd4PCMat = self._checkBoolean(self.getOption('delturbprod4pcmat'))
            calcPCMat = self._checkBoolean(self.getOption('calcpcmat'))
            
            # Flow options
            f.write('flowOptions\n')
            f.write('{\n')
            f.write('    flowCondition          %s;\n'%self.getOption('flowcondition'))
            f.write('    setFlowBCs             %s;\n'%setFlowBCs)
            flowbcs=self.getOption('flowbcs')
            f.write('    flowBCs                {')
            for key in flowbcs.keys():
                f.write(' %s '%key)
                if key=='useWallFunction':
                    f.write('%s;'%flowbcs[key])
                else:
                    patch = flowbcs[key]['patch']
                    variable = flowbcs[key]['variable']
                    value = flowbcs[key]['value']
                    valStr = ' '.join(str(i) for i in value)
                    if 'pressure' in flowbcs[key].keys():
                        pressure = flowbcs[key]['pressure']
                        preStr = ' '.join(str(i) for i in pressure)
                        f.write('{patch %s; variable %s; pressure (%s); value (%s);}'%(patch, variable,preStr ,valStr))
                    else:
                        f.write('{patch %s; variable %s; value (%s);}'%(patch, variable, valStr))
            f.write(' }\n')
            inletPatches = self.getOption('inletpatches')
            outletPatches = self.getOption('outletpatches')
            inletPatches = ' '.join(inletPatches)
            outletPatches = ' '.join(outletPatches)
            f.write('    inletPatches           (%s);\n'%inletPatches)
            f.write('    outletPatches          (%s);\n'%outletPatches)

            derivUInInfo = self.getOption('derivuininfo')
            f.write('    derivUInInfo           {stateName %s; component %d; type %s;'%(derivUInInfo['stateName'],derivUInInfo['component'],derivUInInfo['type']))
            f.write('patchNames (')
            for p in derivUInInfo['patchNames']:
                f.write('%s '%p)
            f.write(');')
            f.write(' }\n')

            userDefinedPatchInfo = self.getOption('userdefinedpatchinfo')
            f.write('    userDefinedPatchInfo   {')
            for key in userDefinedPatchInfo.keys():
                f.write(' %s {'%key)
                for key1 in userDefinedPatchInfo[key].keys():
                    val = userDefinedPatchInfo[key][key1]
                    f.write(' %s %s;'%(key1,str(val)))
                f.write(' } ')
            f.write(' }\n')
            userDefinedVolumeInfo = self.getOption('userdefinedvolumeinfo')
            f.write('    userDefinedVolumeInfo  {')
            for key in userDefinedVolumeInfo.keys():
                f.write(' %s {'%key)
                for key1 in userDefinedVolumeInfo[key].keys():
                    val = userDefinedVolumeInfo[key][key1]
                    f.write(' %s %s;'%(key1,str(val)))
                f.write(' } ')
            f.write(' }\n')

            referenceValues = self.getOption('referencevalues')
            f.write('    referenceValues        (')
            for key in referenceValues.keys():
                f.write('%s %e '%(key,referenceValues[key]))
            f.write(');\n')
            f.write('    divDev2                %s;\n'%self._checkBoolean(self.getOption('divdev2')))
            f.write('    useNKSolver            %s;\n'%self._checkBoolean(self.getOption('usenksolver')))
            f.write('    nkSegregatedTurb       %s;\n'%self._checkBoolean(self.getOption('nksegregatedturb')))
            f.write('    nkSegregatedPhi        %s;\n'%self._checkBoolean(self.getOption('nksegregatedphi')))
            f.write('    nkRelTol               %e;\n'%self.getOption('nkreltol'))
            f.write('    nkAbsTol               %e;\n'%self.getOption('nkabstol'))
            f.write('    nkSTol                 %e;\n'%self.getOption('nkstol'))
            f.write('    nkEWRTol0              %f;\n'%self.getOption('nkewrtol0'))
            f.write('    nkEWRTolMax            %f;\n'%self.getOption('nkewrtolmax'))
            f.write('    nkPCLag                %d;\n'%self.getOption('nkpclag'))
            f.write('    nkMaxIters             %d;\n'%self.getOption('nkmaxiters'))
            f.write('    nkMaxFuncEvals         %d;\n'%self.getOption('nkmaxfuncevals'))
            f.write('    nkASMOverlap           %d;\n'%self.getOption('nkasmoverlap'))
            f.write('    nkGlobalPCIters        %d;\n'%self.getOption('nkglobalpciters'))     
            f.write('    nkLocalPCIters         %d;\n'%self.getOption('nklocalpciters'))
            f.write('    nkPCFillLevel          %d;\n'%self.getOption('nkpcfilllevel'))
            f.write('    nkJacMatReOrdering     %s;\n'%self.getOption('nkjacmatreordering'))
            f.write('    nkGMRESMaxIters        %d;\n'%self.getOption('nkgmresmaxiters'))
            f.write('    nkGMRESRestart         %d;\n'%self.getOption('nkgmresrestart'))
            f.write('}\n')
            f.write('\n')
            
            #Adjoint options 
            f.write('adjointOptions\n')
            f.write('{\n')
            f.write('    solveAdjoint           %s;\n'%solveAdjoint)
            f.write('    useColoring            %s;\n'%useColoring)
            f.write('    normalizeResiduals     (%s);\n'%normalizeRes)
            f.write('    normalizeStates        (%s);\n'%normalizeStates)
            f.write('    nFFDPoints             %d;\n'%self.getOption('nffdpoints'))
            f.write('    correctWallDist        %s;\n'%correctWallDist)
            f.write('    reduceResCon4JacMat    %s;\n'%reduceResCon4JacMat)
            f.write('    calcPCMat              %s;\n'%calcPCMat)
            f.write('    fastPCMat              %s;\n'%self._checkBoolean(self.getOption('fastpcmat')))
            f.write('    delTurbProd4PCMat      %s;\n'%delTurbProd4PCMat)
            f.write('    writeMatrices          %s;\n'%writeMatrices)   
            f.write('    adjGMRESCalcEigen      %s;\n'%adjGMRESCalcEigen)                 
            f.write('    adjGMRESMaxIters       %d;\n'%self.getOption('adjgmresmaxiters'))
            f.write('    adjGMRESRestart        %d;\n'%self.getOption('adjgmresrestart'))
            f.write('    adjASMOverlap          %d;\n'%self.getOption('adjasmoverlap'))
            f.write('    adjJacMatOrdering      %s;\n'%self.getOption('adjjacmatordering'))
            f.write('    adjJacMatReOrdering    %s;\n'%self.getOption('adjjacmatreordering'))
            f.write('    adjGlobalPCIters       %d;\n'%self.getOption('adjglobalpciters'))     
            f.write('    adjLocalPCIters        %d;\n'%self.getOption('adjlocalpciters'))
            f.write('    adjPCFillLevel         %d;\n'%self.getOption('adjpcfilllevel'))
            f.write('    adjGMRESRelTol         %e;\n'%self.getOption('adjgmresreltol'))
            f.write('    adjGMRESAbsTol         %e;\n'%self.getOption('adjgmresabstol'))
            f.write('    minTolJac              %e;\n'%self.getOption('mintoljac'))
            f.write('    maxTolJac              %e;\n'%self.getOption('maxtoljac'))
            f.write('    minTolPC               %e;\n'%self.getOption('mintolpc')) 
            f.write('    maxTolPC               %e;\n'%self.getOption('maxtolpc')) 
            f.write('    stateResetTol          %e;\n'%self.getOption('stateresettol')) 
            f.write('    tractionBCMaxIter      %d;\n'%self.getOption('tractionbcmaxiter'))              
            f.write('    epsDeriv               %e;\n'%self.getOption('epsderiv'))  
            f.write('    epsDerivFFD            %e;\n'%self.getOption('epsderivffd'))
            f.write('    epsDerivXv             %e;\n'%self.getOption('epsderivxv'))
            f.write('    epsDerivUIn            %e;\n'%self.getOption('epsderivuin'))
            f.write('    epsDerivVis            %e;\n'%self.getOption('epsderivvis'))
            
            scaling = ''
            statescaling = self.getOption('statescaling')
            for key in statescaling.keys():
                scaling = scaling+' '+key+' '+str(statescaling[key])
            f.write('    stateScaling           (%s);\n'%scaling)

            scaling = ''
            residualscaling = self.getOption('residualscaling')
            for key in residualscaling.keys():
                scaling = scaling+' '+key+' '+str(residualscaling[key])
            f.write('    residualScaling        (%s);\n'%scaling)
            
            maxResConLv4JacPCMat = self.getOption('maxresconlv4jacpcmat')
            pcConLv = ''
            for key in maxResConLv4JacPCMat.keys():
                pcConLv = pcConLv+' '+key+' '+str(maxResConLv4JacPCMat[key])
            f.write('    maxResConLv4JacPCMat   (%s);\n'%pcConLv)
            
            adjDVTypes = self.getOption('adjdvtypes')
            adjDVTypes = ' '.join(adjDVTypes)
            f.write('    adjDVTypes             (%s);\n'%adjDVTypes)
            f.write('}\n')
            f.write('\n')

            #Actuator disk options 
            f.write('actuatorDiskOptions\n')
            f.write('{\n')
            f.write('    actuatorActive         %d;\n'%self.getOption('actuatoractive'))
            f.write('    actuatorAdjustThrust   %d;\n'%self.getOption('actuatoradjustthrust'))
            f.write('    actuatorVolumeNames    (')
            for c1 in self.getOption('actuatorvolumenames'):
                f.write('%s '%c1)
            f.write(');\n')
            f.write('    actuatorThrustCoeff    (')
            for d1 in self.getOption('actuatorthrustcoeff'):
                f.write('%f '%d1)
            f.write(');\n')
            f.write('    actuatorPOverD         (')
            for f1 in self.getOption('actuatorpoverd'):
                f.write('%f '%f1)
            f.write(');\n')
            f.write('    actuatorRotationDir    (')
            for g1 in self.getOption('actuatorrotationdir'):
                f.write('%s '%g1)
            f.write(');\n')
            f.write('}\n')
            f.write('\n')

            #Objective function options 
            objfuncs = self.getOption('objfuncs')
            objfuncs = ' '.join(objfuncs)
            objfuncgeoinfo = self.getOption('objfuncgeoinfo')
            patches = ' '
            for i in range(len(objfuncgeoinfo)):
                patches = patches + ' ('
                tmp = ' '.join(objfuncgeoinfo[i])
                patches = patches + tmp
                patches = patches + ') '
            f.write('objectiveFunctionOptions\n')
            f.write('{\n')
            f.write('    objFuncs               (%s);\n'%objfuncs)
            f.write('    objFuncGeoInfo         (%s);\n'%patches)
            
            f.write('    dragDir                (')
            for dir1 in self.getOption('dragdir'):
                f.write('%12.10f '%dir1)
            f.write(');\n')
            f.write('    liftDir                (')
            for dir1 in self.getOption('liftdir'):
                f.write('%12.10f '%dir1)
            f.write(');\n')
            f.write('    CofR                   (')
            for dir1 in self.getOption('cofr'):
                f.write('%12.10f '%dir1)
            f.write(');\n')
            f.write('    rotRad                 (')
            for rot in self.getOption('rotrad'):
                f.write('%12.10f '%rot)
            f.write(');\n')
            f.write('}\n')
            f.write('\n')

            f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
            f.close()
        return
        

    def _checkBoolean(self,var):
        if var:
            return 'true'
        else:
            return 'false'               

if __name__ == "__main__" :
  options = {}
  pof = PYDAFOAM(options=options)

  pof()
