#!/usr/bin/env python

"""

DAFoam  : Discrete Adjoint with OpenFOAM
Version : v4

Description:
The Python interface to DAFoam. It controls the adjoint
solvers and external modules for design optimization

"""

__version__ = "4.0.2"

import subprocess
import os
import sys
import copy
import shutil
import numpy as np
from mpi4py import MPI
from collections import OrderedDict
import petsc4py
from petsc4py import PETSc

petsc4py.init(sys.argv)
try:
    import tensorflow as tf
except ImportError:
    pass


class DAOPTION(object):
    """
    Define a set of options to use in PYDAFOAM and set their initial values.
    This class will be used by PYDAFOAM._getDefOptions()

    NOTE: Give an initial value for a new option, this help PYDAFOAM determine the type of this
    option. If it is a list, give at least one default value. If it is a dict, you can leave it
    blank, e.g., {}. Also, use ## to add comments before the new option such that these comments
    will be  picked up by Doxygen. If possible, give examples.

    NOTE: We group these options into three categories.
    - The basic options are those options that will be used for EVERY solvers and EVERY cases.
    - The intermediate options are options that will be used in some of solvers for special
      situation (e.g., primalVarBounds to prevent solution from divergence).
    - The advanced options will be used in special situation to improve performance, e.g.,
      maxResConLv4JacPCMat to reduce memory of preconditioner. Its usage is highly case dependent.
      It may also include options that have a default value that is rarely changed, except for
      very special situation.
    """

    def __init__(self):

        # *********************************************************************************************
        # *************************************** Basic Options ***************************************
        # *********************************************************************************************

        ## The name of the DASolver to use for primal and adjoint computation.
        ## See dafoam/src/adjoint/DASolver for more details
        ## Currently support:
        ## - DASimpleFoam:            Incompressible steady-state flow solver for Navier-Stokes equations
        ## - DASimpleTFoam:           Incompressible steady-state flow solver for Navier-Stokes equations with temperature
        ## - DAPisoFoam:              Incompressible transient flow solver for Navier-Stokes equations
        ## - DAPimpleFoam:            Incompressible transient flow solver for Navier-Stokes equations
        ## - DARhoSimpleFoam:         Compressible steady-state flow solver for Navier-Stokes equations (subsonic)
        ## - DARhoSimpleCFoam:        Compressible steady-state flow solver for Navier-Stokes equations (transonic)
        ## - DATurboFoam:             Compressible steady-state flow solver for Navier-Stokes equations (turbomachinery)
        ## - DASolidDisplacementFoam: Steady-state structural solver for linear elastic equations
        self.solverName = "DASimpleFoam"

        ## The convergence tolerance for the primal solver. If the primal can not converge to 2 orders
        ## of magnitude (default) higher than this tolerance, the primal solution will return fail=True
        self.primalMinResTol = 1.0e-8

        ## The boundary condition for primal solution. The keys should include "variable", "patch",
        ## and "value". For turbulence variable, one can also set "useWallFunction" [bool].
        ## Note that setting "primalBC" will overwrite any values defined in the "0" folder.
        ## The primalBC setting will be printed to screen for each primal solution during the optimization
        ## Example
        ##    "primalBC": {
        ##        "U0": {"variable": "U", "patches": ["inlet"], "value": [10.0, 0.0, 0.0]},
        ##        "p0": {"variable": "p", "patches": ["outlet"], "value": [101325.0]},
        ##        "nuTilda0": {"variable": "nuTilda", "patches": ["inlet"], "value": [1.5e-4]},
        ##        "useWallFunction": True,
        ##    },
        self.primalBC = {}

        ## State normalization for dRdWT computation. Typically, we set far field value for each state
        ## variable. NOTE: If you forget to set normalization value for a state variable, the adjoint
        ## may not converge or it may be inaccurate! For "phi", use 1.0 to normalization
        ## Example
        ##     normalizeStates = {"U": 10.0, "p": 101325.0, "phi": 1.0, "nuTilda": 1.e-4}
        self.normalizeStates = {}

        ## Information on objective and constraint functions. Each function requires a different input form
        ## But for all functions, we need to give a name to the function, e.g., CD or any
        ## other preferred name. Then, define the type of function (e.g., force, moment; we need to use
        ## the reserved type names), how to select the discrete mesh faces to compute the function
        ## (e.g., we select them from the name of a patch patchToFace), the name of the patch (wing)
        ## for patchToFace, the scaling factor "scale". For forces, we need to project the force vector to a
        ## specific direction. The following example defines that CD is the force that is parallel to flow
        ## (parallelToFlow). Alternative, we can also use fixedDirection and provide a direction key for
        ## force, i.e., "directionMode": "fixedDirection", "direction": [1.0, 0.0, 0.0]. Since we select
        ## parallelToFlow, we need to prescribe the name of angle of patchVelocityInputName to determine
        ## the flow direction.
        ## NOTE: if patchVelocity not added in inputInfo, we can NOT use parallelToFlow.
        ## For this case, we have to use "directionMode": "fixedDirection".
        ## Example
        ##     "function": {
        ##         "CD": {
        ##             "type": "force",
        ##             "source": "patchToFace",
        ##             "patches": ["wing"],
        ##             "directionMode": "parallelToFlow",
        ##             "patchVelocityInputName": "patchV",
        ##             "scale": 1.0 / (0.5 * UmagIn * UmagIn * ARef),
        ##         },
        ##         "CL": {
        ##             "type": "force",
        ##             "source": "patchToFace",
        ##             "patches": ["wing"],
        ##             "directionMode": "normalToFlow",
        ##             "patchVelocityInputName": "patchV",
        ##             "scale": 1.0 / (0.5 * UmagIn * UmagIn * ARef),
        ##         },
        ##         "CMZ": {
        ##             "type": "moment",
        ##             "source": "patchToFace",
        ##             "patches": ["wing"],
        ##             "axis": [0.0, 0.0, 1.0],
        ##             "center": [0.25, 0.0, 0.05],
        ##             "scale": 1.0 / (0.5 * UmagIn * UmagIn * ARef * LRef),
        ##         },
        ##         "TPR": {
        ##             "type": "totalPressureRatio",
        ##             "source": "patchToFace",
        ##             "patches": ["inlet", "outlet"],
        ##             "inletPatches": ["inlet"],
        ##             "outletPatches": ["outlet"],
        ##             "scale": 1.0,
        ##         },
        ##         "TTR": {
        ##             "type": "totalTemperatureRatio",
        ##             "source": "patchToFace",
        ##             "patches": ["inlet", "outlet"],
        ##             "inletPatches": ["inlet"],
        ##             "outletPatches": ["outlet"],
        ##             "scale": 1.0,
        ##         },
        ##         "MFR": {
        ##             "part1": {
        ##                 "type": "massFlowRate",
        ##                 "source": "patchToFace",
        ##                 "patches": ["inlet"],
        ##                 "scale": -1.0,
        ##                 "addToAdjoint": True,
        ##             }
        ##         },
        ##        "TP": {
        ##            "type": "totalPressure",
        ##            "source": "patchToFace",
        ##            "patches": ["inlet"],
        ##            "scale": 1.0 / (0.5 * U0 * U0),
        ##        },
        ##        "NU": {
        ##            "type": "wallHeatFlux",
        ##            "source": "patchToFace",
        ##            "patches": ["ubend"],
        ##            "scale": 1.0,
        ##        },
        ##        "VMS": {
        ##            "type": "vonMisesStressKS",
        ##            "source": "boxToCell",
        ##            "min": [-10.0, -10.0, -10.0],
        ##            "max": [10.0, 10.0, 10.0],
        ##            "scale": 1.0,
        ##            "coeffKS": 2.0e-3,
        ##        },
        ##        "M": {
        ##            "type": "mass",
        ##            "source": "boxToCell",
        ##            "min": [-10.0, -10.0, -10.0],
        ##            "max": [10.0, 10.0, 10.0],
        ##            "scale": 1.0,
        ##        },
        ##        "THRUST": {
        ##            "type": "variableVolSum",
        ##            "source": "boxToCell",
        ##            "min": [-50.0, -50.0, -50.0],
        ##            "max": [50.0, 50.0, 50.0],
        ##            "varName": "fvSource",
        ##            "varType": "vector",
        ##            "component": 0,
        ##            "isSquare": 0,
        ##            "scale": 1.0,
        ##        },
        ##        "COP": {
        ##            "type": "centerOfPressure",
        ##            "source": "patchToFace",
        ##            "patches": ["wing"],
        ##            "axis": [1.0, 0.0, 0.0],
        ##            "forceAxis": [0.0, 1.0, 0.0],
        ##            "center": [0, 0, 0],
        ##            "scale": 1.0,
        ##            "addToAdjoint": True,
        ##        },
        ##    },
        self.function = {}

        ## General input information. Different type of inputs require different keys
        ## For patchVelocity, we need to set a list of far field patch names from which the angle of
        ## attack is computed, this is usually a far field patch. Also, we need to prescribe
        ## flow and normal axies, and alpha = atan( U_normal / U_flow ) at patches
        ## Example
        ##     inputInfo = {
        ##         "aero_vol_coords" : {"type": "volCoord", "addToSolver": True},
        ##         "patchV" = {
        ##             "type": "patchVelocity",
        ##             "patches": ["farField"],
        ##             "flowAxis": "x",
        ##             "normalAxis": "y",
        ##             "addToSolver": True,
        ##         },
        ##         "ux0" = {
        ##             "type": "patchVariable",
        ##             "patches": ["inlet"],
        ##             "variable": "U",
        ##             "comp": 0,
        ##             "addToSolver": True,
        ##         },
        ##     }
        self.inputInfo = {}

        ## General input information. Different type of outputs require different keys
        ## Example
        ##     outputInfo = {
        ##         "T_conduct" : {"type": "thermalVar", "patches": ["wing"]},
        ##         "f_aero": {"type": "surfForce", "patches": ["wing"]}
        ##     }
        self.outputInfo = {}

        ## List of patch names for the design surface. These patch names need to be of wall type
        ## and shows up in the constant/polyMesh/boundary file
        self.designSurfaces = ["ALL_OPENFOAM_WALL_PATCHES"]

        # *********************************************************************************************
        # ****************************** Intermediate Options *****************************************
        # *********************************************************************************************

        ## Information for the finite volume source term, which will be added in the momentum equation
        ## We support multiple source terms
        ## Example
        ## "fvSource": {
        ##     "disk1": {
        ##         "type": "actuatorDisk", # Actuator disk source. This is a child class in DAFvSource
        ##         "source": "cylinderAnnulusToCell", # Define a volume to add the fvSource term
        ##         "p1": [-0.4, -0.1, 0.05],  # p1 and p2 define the axis and width
        ##         "p2": [-0.1, -0.1, 0.05],  # p2-p1 should be streamwise
        ##         "innerRadius": 0.01,
        ##         "outerRadius": 0.5,
        ##         "rotDir": "left",
        ##         "scale": 50.0,
        ##         "POD": 0.7, # pitch/diameter
        ##     },
        ##     "disk2": {
        ##         "type": "actuatorDisk",
        ##         "source": "cylinderAnnulusToCell",
        ##         "p1": [-0.4, 0.1, 0.05],
        ##         "p2": [-0.1, 0.1, 0.05],
        ##         "innerRadius": 0.01,
        ##         "outerRadius": 0.5,
        ##         "rotDir": "right",
        ##         "scale": 25.0,  # scale the source such the integral equals desired thrust
        ##         "POD": 1.0,
        ##     },
        ##    "line1":
        ##    {
        ##        "type": "actuatorPoint",
        ##        "smoothFunction": "hyperbolic", # or gaussian
        ##        "center": [-0.55, 0.0, 0.05],  # center and size define a rectangular
        ##        "size": [0.2, 0.2, 0.1],
        ##        "amplitude": [0.0, 0.2, 0.0],
        ##        "thrustDirIdx": 0,
        ##        "periodicity": 0.1,
        ##        "eps": 10.0,
        ##        "scale": 10.0  # scale the source such the integral equals desired thrust
        ##    },
        ##    "gradP"
        ##    {
        ##        "type": "uniformPressureGradient",
        ##        "value": 1e-3,
        ##        "direction": [1.0, 0.0, 0.0],
        ##    },
        ## },
        self.fvSource = {}

        ## The adjoint equation solution method. Options are: Krylov or fixedPoint
        self.adjEqnSolMethod = "Krylov"

        ## whether the dynamic mesh is activated. The default is False, but if we need to use
        ## DAPimpleDyMFoam, we need to set this flaf to True
        self.dynamicMesh = {
            "active": False,
            "mode": "rotation",
            "center": [0.25, 0.0, 0.0],
            "axis": "z",
            "omega": 0.1,
        }

        ## The variable upper and lower bounds for primal solution. The key is variable+"Max/Min".
        ## Setting the bounds increases the robustness of primal solution for compressible solvers.
        ## Also, we set lower bounds for turbulence variables to ensure they are physical
        ## Example
        ##     primalValBounds = {"UMax": 1000, "UMin": -1000, "pMax": 1000000}
        self.primalVarBounds = {
            "UMax": 1000.0,
            "UMin": -1000.0,
            "pMax": 500000.0,
            "pMin": 20000.0,
            "p_rghMax": 500000.0,
            "p_rghMin": 20000.0,
            "eMax": 500000.0,
            "eMin": 100000.0,
            "TMax": 1000.0,
            "TMin": 100.0,
            "hMax": 500000.0,
            "hMin": 100000.0,
            "DMax": 1e16,
            "DMin": -1e16,
            "rhoMax": 5.0,
            "rhoMin": 0.2,
            "nuTildaMax": 1e16,
            "nuTildaMin": 1e-16,
            "kMax": 1e16,
            "kMin": 1e-16,
            "omegaMax": 1e16,
            "omegaMin": 1e-16,
            "epsilonMax": 1e16,
            "epsilonMin": 1e-16,
            "ReThetatMax": 1e16,
            "ReThetatMin": 1e-16,
            "gammaIntMax": 1e16,
            "gammaIntMin": 1e-16,
        }

        ## The discipline name. The default is "aero". If we need to couple two solvers using
        ## DAFoam, e.g., aero+thermal, we need to set it to something like "thermal"
        self.discipline = "aero"

        ## The step size for finite-difference computation of partial derivatives. The default values
        ## will work for most of the case.
        self.adjPartDerivFDStep = {
            "State": 1.0e-6,
        }

        ## Which options to use to improve the adjoint equation convergence of transonic conditions
        ## This is used only for transonic solvers such as DARhoSimpleCFoam
        self.transonicPCOption = -1

        ## Options for unsteady adjoint. mode can be hybrid or timeAccurate
        ## Here nTimeInstances is the number of time instances and periodicity is the
        ## periodicity of flow oscillation (hybrid adjoint only)
        self.unsteadyAdjoint = {
            "mode": "None",
            "PCMatPrecomputeInterval": 100,
            "PCMatUpdateInterval": 1,
            "reduceIO": True,
            "additionalOutput": ["None"],
            "readZeroFields": True,
        }

        ## The interval of recomputing the pre-conditioner matrix dRdWTPC for solveAdjoint
        ## By default, dRdWTPC will be re-computed each time the solveAdjoint function is called
        ## However, one can increase the lag to skip it and reuse the dRdWTPC computed previously.
        ## This obviously increses the speed because the dRdWTPC computation takes about 30% of
        ## the adjoint total runtime. However, setting a too large lag value will decreases the speed
        ## of solving the adjoint equations. One needs to balance these factors
        self.adjPCLag = 10000

        ## Whether to use AD: Mode options: forward, reverse, or fd. If forward mode AD is used
        ## the seedIndex will be set to compute derivative by running the whole primal solver.
        ## dvName is the name of design variable to set the seed for the forward AD
        ## setting seedIndex to -1 for dFdField will assign seeds for all design variables.
        ## If reverse mode is used, the adjoint will be computed by a Jacobian free approach
        ## refer to: Kenway et al. Effective adjoint approach for computational fluid dynamics,
        ## Progress in Aerospace Science, 2019.
        self.useAD = {"mode": "reverse", "dvName": "None", "seedIndex": -9999}

        ## whether to use the constrainHbyA in the pEqn. The DASolvers are similar to OpenFOAM's native
        ## solvers except that we directly compute the HbyA term without any constraints. In other words,
        ## we comment out the constrainHbyA line in the pEqn. However, some cases may diverge without
        ## the constrainHbyA, e.g., the MRF cases with the SST model. Here we have an option to add the
        ## constrainHbyA back to the primal and adjoint solvers.
        self.useConstrainHbyA = True

        ## parameters for regression models
        ## we support defining multiple regression models. Each regression model can have only one output
        ## but it can have multiple input features. Refer to src/adjoint/DARegression/DARegression.C for
        ## a full list of supported input features. There are two supported regression model types:
        ## neural network and radial basis function. We can shift and scale the inputs and outputs
        ## we can also prescribe a default output value. The default output will be used in resetting
        ## when they are nan, inf, or out of the prescribe upper and lower bounds
        self.regressionModel = {
            "active": False,
            # "model1": {
            #    "modelType": "neuralNetwork",
            #    "inputNames": ["PoD", "VoS", "PSoSS", "chiSA"],
            #    "outputName": "betaFINuTilda",
            #    "hiddenLayerNeurons": [20, 20],
            #    "inputShift": [0.0],
            #    "inputScale": [1.0],
            #    "outputShift": 1.0,
            #    "outputScale": 1.0,
            #    "outputUpperBound": 1e1,
            #    "outputLowerBound": -1e1,
            #    "activationFunction": "sigmoid",  # other options are relu and tanh
            #    "printInputInfo": True,
            #    "defaultOutputValue": 1.0,
            # },
            # "model2": {
            #    "modelType": "radialBasisFunction",
            #    "inputNames": ["KoU2", "ReWall", "CoP", "TauoK"],
            #    "outputName": "betaFIOmega",
            #    "nRBFs": 50,
            #    "inputShift": [0.0],
            #    "inputScale": [1.0],
            #    "outputShift": 1.0,
            #    "outputScale": 1.0,
            #    "outputUpperBound": 1e1,
            #    "outputLowerBound": -1e1,
            #    "printInputInfo": True,
            #    "defaultOutputValue": 1.0,
            # },
        }

        ## whether to use step-averaged state variables. This can be useful when the primal solution
        ## exhibits LCO. To make this work, users need to setup function-fieldAverage in system/controlDict
        ## for all state variables
        self.useMeanStates = False

        # *********************************************************************************************
        # ************************************ Advance Options ****************************************
        # *********************************************************************************************

        ## The current objective function name. This variable will be reset in DAFoamFunctions
        ## Then, we know which objective function is used in solve_linear in DAFoamSolver
        self.solveLinearFunctionName = "None"

        ## Whether to print all DAOption defined in the C++ layer to screen before optimization.
        self.printDAOptions = True

        ## Whether running the optimization in the debug mode, which prints extra information.
        self.debug = False

        ## Whether to write Jacobian matrices to file for debugging
        ## Example:
        ##    writeJacobians = ["dRdWT", "dFdW"]
        ## This will write the dRdWT and dFdW matrices to the disk
        self.writeJacobians = ["None"]

        ## The print interval of primal and adjoint solution, e.g., how frequent to print the primal
        ## solution steps, how frequent to print the dRdWT partial derivative computation.
        self.printInterval = 100

        ## The print interval of unsteady primal solvers, e.g., for DAPisoFoam
        self.printIntervalUnsteady = 1

        ## Users can adjust primalMinResTolDiff to tweak how much difference between primalMinResTol
        ## and the actual primal convergence is consider to be fail=True for the primal solution.
        self.primalMinResTolDiff = 1.0e2

        ## Whether to use graph coloring to accelerate partial derivative computation. Unless you are
        ## debugging the accuracy of partial computation, always set it to True
        self.adjUseColoring = True

        ## The Petsc options for solving the adjoint linear equation. These options should work for
        ## most of the case. If the adjoint does not converge, try to increase pcFillLevel to 2, or
        ## try "jacMatReOrdering": "nd"
        self.adjEqnOption = {
            "globalPCIters": 0,
            "asmOverlap": 1,
            "localPCIters": 1,
            "jacMatReOrdering": "rcm",
            "pcFillLevel": 1,
            "gmresMaxIters": 1000,
            "gmresRestart": 1000,
            "gmresRelTol": 1.0e-6,
            "gmresAbsTol": 1.0e-14,
            "gmresTolDiff": 1.0e2,
            "useNonZeroInitGuess": False,
            "useMGSO": False,
            "printInfo": 1,
            "fpMaxIters": 1000,
            "fpRelTol": 1e-6,
            "fpMinResTolDiff": 1.0e2,
            "fpPCUpwind": False,
            "dynAdjustTol": False,
        }

        ## Normalization for residuals. We should normalize all residuals!
        self.normalizeResiduals = [
            "URes",
            "pRes",
            "p_rghRes",
            "nuTildaRes",
            "phiRes",
            "TRes",
            "DRes",
            "kRes",
            "omegaRes",
            "epsilonRes",
        ]

        ## The maximal connectivity level for the dRdWTPC matrix. Reducing the connectivity level
        ## reduce the memory usage, however, it may slow down the adjoint equation convergence.
        ## The default value should have the best convergence speed but not optimal memory usage.
        self.maxResConLv4JacPCMat = {
            "pRes": 2,
            "phiRes": 1,
            "URes": 2,
            "TRes": 2,
            "nuTildaRes": 2,
            "kRes": 2,
            "epsilonRes": 2,
            "omegaRes": 2,
            "p_rghRes": 2,
            "DRes": 2,
            "gammaIntRes": 2,
            "ReThetatRes": 2,
        }

        ## The min bound for Jacobians, any value that is smaller than the bound will be set to 0
        ## Setting a large lower bound for preconditioner (PC) can help to reduce memory.
        self.jacLowerBounds = {
            "dRdW": 1.0e-30,
            "dRdWPC": 1.0e-30,
        }

        ## The maximal iterations of tractionDisplacement boundary conditions
        self.maxTractionBCIters = 100

        ## decomposeParDict option. This file will be automatically written such that users
        ## can run optimization with any number of CPU cores without the need to manually
        ## change decomposeParDict
        self.decomposeParDict = {
            "method": "scotch",
            "simpleCoeffs": {"n": [2, 2, 1], "delta": 0.001},
            "preservePatches": ["None"],
            "singleProcessorFaceSets": ["None"],
            "args": ["None"],
        }

        ## The ordering of state variable. Options are: state or cell. Most of the case, the state
        ## odering is the best choice.
        self.adjStateOrdering = "state"

        ## The threshold for check mesh call
        self.checkMeshThreshold = {
            "maxAspectRatio": 1000.0,
            "maxNonOrth": 70.0,
            "maxSkewness": 4.0,
            "maxIncorrectlyOrientedFaces": 0,
        }

        ## The sensitivity map will be saved to disk during optimization for the given design variable
        ## names in the list. Currently only support design variable type FFD and Field
        ## NOTE: this function only supports useAD->mode:reverse
        ## Example:
        ##     "writeSensMap" : ["shapex", "shapey"]
        self.writeSensMap = ["NONE"]

        ## Whether to write deformed FFDs to the disk during optimization, i.e., DVGeo.writeTecplot
        self.writeDeformedFFDs = False

        ## Whether to write deformed constraints to disk during optimization, i.e., DVCon.writeTecplot
        self.writeDeformedConstraints = False

        ## Whether to write adjoint variables in OpenFOAM field format for post-processing
        self.writeAdjointFields = False

        ## The max number of correctBoundaryConditions calls in the updateOFField function.
        self.maxCorrectBCCalls = 2

        ## Whether to write the primal solutions for minor iterations (i.e., line search).
        ## The default is False. If set it to True, it will write flow fields (and the deformed geometry)
        ## for each primal solution. This will significantly increases the IO runtime, so it should never
        ## be True for production runs. However, it is useful for debugging purpose (e.g., to find out
        ## the poor quality mesh during line search)
        self.writeMinorIterations = False

        ## number of minimal primal iterations. The primal has to run this many iterations, even the primal residual
        ## has reduced below the tolerance. The default is 1: the primal has to run for at least one iteration
        self.primalMinIters = 1

        ## tensorflow related functions
        self.tensorflow = {
            "active": False,
            # "model1": {
            #    "predictBatchSize": 1000
            # }
        }

        ## Whether to use OpenFOAMs snGrad() function or to manually compute distance for wall interfaces
        self.wallDistanceMethod = "default"

        ## the component output for the unsteady solvers. This will be used in mphys_dafoam's
        ## DAFoamBuilderUnsteady to determine the component's output
        ##
        ## Example
        ##     "unsteadyCompOutput": {"output1": ["function1", "function2"], "output2": ["function3"]}
        ##
        ## here we have two outputs and they can be used as objective or constraints.
        ## the first output is the summation of function1 and function2, here function1 and function2
        ## should be the function name defined in the function dict in DAOption
        self.unsteadyCompOutput = {}


class PYDAFOAM(object):
    """
    Main class for pyDAFoam

    Parameters
    ----------

    comm : mpi4py communicator
        An optional argument to pass in an external communicator.

    options : dictionary
        The list of options to use with pyDAFoam.

    """

    def __init__(self, comm=None, options=None):
        """
        Initialize class members
        """

        assert not os.getenv("WM_PROJECT") is None, "$WM_PROJECT not found. Please source OpenFOAM-v1812/etc/bashrc"

        self.version = __version__

        Info(" ")
        Info("-------------------------------------------------------------------------------")
        Info("|                               DAFoam v%s                                 |" % self.version)
        Info("-------------------------------------------------------------------------------")
        Info(" ")

        # name
        self.name = "PYDAFOAM"

        # register solver names and set their types
        self._solverRegistry()

        # initialize options for adjoints
        self._initializeOptions(options)

        # initialize comm for parallel communication
        self._initializeComm(comm)

        # Initialize families
        self.families = OrderedDict()

        # Default it to fault, after calling setSurfaceCoordinates, set it to true
        self._updateGeomInfo = False

        # Use double data type: 'd'
        self.dtype = "d"

        # run decomposePar for parallel runs
        self.runDecomposePar()

        # initialize the pySolvers
        self.solverInitialized = 0
        self._initSolver()

        # set the primal boundary condition after initializing the solver
        self.setPrimalBoundaryConditions()

        # initialize the number of primal and adjoint calls
        self.nSolvePrimals = 1
        self.nSolveAdjoints = 1

        # flags for primal and adjoint failure
        self.primalFail = 0
        self.adjointFail = 0

        # initialize mesh information and read grids
        self._readMeshInfo()

        # check if the combination of options is valid.
        self._checkOptions()

        # get the reduced point connectivities for the base patches in the mesh
        self._computeBasicFamilyInfo()

        # Add a couple of special families.
        self.allSurfacesGroup = "allSurfaces"
        self.addFamilyGroup(self.allSurfacesGroup, self.basicFamilies)

        self.allWallsGroup = "allWalls"
        self.addFamilyGroup(self.allWallsGroup, self.wallList)

        # Set the design surfaces group
        self.designSurfacesGroup = "designSurfaces"
        if "ALL_OPENFOAM_WALL_PATCHES" in self.getOption("designSurfaces"):
            self.addFamilyGroup(self.designSurfacesGroup, self.wallList)
        else:
            self.addFamilyGroup(self.designSurfacesGroup, self.getOption("designSurfaces"))

        # By Default we don't have an external mesh object or a
        # geometric manipulation object
        self.mesh = None

        # preconditioner matrix
        self.dRdWTPC = None

        # a KSP object which may be used outside of the pyDAFoam class
        self.ksp = None

        # a flag used in deformDynamicMesh for runMode=runOnce
        self.dynamicMeshDeformed = 0

        if self.getOption("tensorflow")["active"]:
            TensorFlowHelper.options = self.getOption("tensorflow")
            TensorFlowHelper.initialize()
            # pass this helper function to the C++ layer
            self.solver.initTensorFlowFuncs(
                TensorFlowHelper.predict, TensorFlowHelper.calcJacVecProd, TensorFlowHelper.setModelName
            )
            self.solverAD.initTensorFlowFuncs(
                TensorFlowHelper.predict, TensorFlowHelper.calcJacVecProd, TensorFlowHelper.setModelName
            )

        if self.getOption("printDAOptions"):
            self.solver.printAllOptions()

        Info("pyDAFoam initialization done!")

        return

    def _solverRegistry(self):
        """
        Register solver names and set their types. For a new solver, first identify its
        type and add their names to the following dict
        """

        self.solverRegistry = {
            "Incompressible": ["DASimpleFoam", "DAPimpleFoam", "DAPimpleDyMFoam"],
            "Compressible": ["DARhoSimpleFoam", "DARhoSimpleCFoam", "DATurboFoam", "DARhoPimpleFoam"],
            "Solid": ["DASolidDisplacementFoam", "DAHeatTransferFoam"],
        }

    def __call__(self):
        """
        Solve the primal
        """

        Info("Running Primal Solver %03d" % self.nSolvePrimals)

        self.deletePrevPrimalSolTime()

        # self.primalFail: if the primal solution fails, assigns 1, otherwise 0
        self.primalFail = 0
        if self.getOption("useAD")["mode"] == "forward":
            self.primalFail = self.solverAD.solvePrimal()
        else:
            self.primalFail = self.solver.solvePrimal()

        if self.getOption("writeMinorIterations"):
            self.renameSolution(self.nSolvePrimals)

        self.nSolvePrimals += 1

        return

    def _getDefOptions(self):
        """
        Setup default options

        Returns
        -------

        defOpts : dict
            All the DAFoam options.
        """

        # initialize the DAOPTION object
        daOption = DAOPTION()

        defOpts = {}

        # assign all the attribute of daOptoin to defOpts
        for key in vars(daOption):
            value = getattr(daOption, key)
            defOpts[key] = [type(value), value]

        return defOpts

    def _checkOptions(self):
        """
        Check if the combination of options are valid.
        NOTE: we should add all possible checks here!
        """

        if not self.getOption("useAD")["mode"] in ["reverse", "forward"]:
            raise Error("useAD->mode only supports reverse, or forward!")

        if "NONE" not in self.getOption("writeSensMap"):
            if not self.getOption("useAD")["mode"] in ["reverse"]:
                raise Error("writeSensMap is only compatible with useAD->mode=reverse")

        if self.getOption("adjEqnSolMethod") == "fixedPoint":
            # for the fixed-point adjoint, we should not normalize the states and residuals
            if self.comm.rank == 0:
                print("Fixed-point adjoint mode detected. Unset normalizeStates and normalizeResiduals...")

            # force the normalize states to be an empty dict
            if len(self.getOption("normalizeStates")) > 0:
                raise Error("Please do not set any normalizeStates for the fixed-point adjoint!")
            # force the normalize residuals to be None; don't normalize any residuals
            self.setOption("normalizeResiduals", ["None"])
            # update this option to the C++ layer
            self.updateDAOption()

        if self.getOption("discipline") not in ["aero", "thermal"]:
            raise Error("discipline: %s not supported. Options are: aero or thermal" % self.getOption("discipline"))

        # check the patchNames from primalBC dict
        primalBCDict = self.getOption("primalBC")
        for bcKey in primalBCDict:
            try:
                patches = primalBCDict[bcKey]["patches"]
            except Exception:
                continue
            for patchName in patches:
                if patchName not in self.boundaries.keys():
                    raise Error(
                        "primalBC-%s-patches-%s is not valid. Please use a patchName from the boundaries list: %s"
                        % (bcKey, patchName, self.boundaries.keys())
                    )

        # check the patch names from function dict
        functionDict = self.getOption("function")
        for objKey in functionDict:
            try:
                patches = functionDict[objKey]["patches"]
            except Exception:
                continue
            for patchName in patches:
                if patchName not in self.boundaries.keys():
                    raise Error(
                        "function-%s-patches-%s is not valid. Please use a patchName from the boundaries list: %s"
                        % (objKey, patchName, self.boundaries.keys())
                    )

        # check other combinations...

    def calcPrimalResidualStatistics(self, mode):
        if self.getOption("useAD")["mode"] in ["forward", "reverse"]:
            self.solverAD.calcPrimalResidualStatistics(mode)
        else:
            self.solver.calcPrimalResidualStatistics(mode)

    def writeAdjointFields(self, function, writeTime, psi):
        """
        Write the adjoint variables in OpenFOAM field format for post-processing
        """

        if self.getOption("writeAdjointFields"):
            if len(self.getOption("function").keys()) > 1:
                raise Error("writeAdjointFields supports only one function, while multiple are defined!")
            self.solver.writeAdjointFields(function, writeTime, psi)

    def evalFunctions(self, funcs):
        """
        Evaluate the desired functions given in iterable object,

        Examples
        --------
        >>> funcs = {}
        >>> CFDsolver()
        >>> CFDsolver.evalFunctions(funcs)
        >>> # Result will look like:
        >>> # {'CL':0.501, 'CD':0.02750}
        """

        for funcName in list(self.getOption("function").keys()):
            # call self.solver.getFunctionValue to get the functionValue from
            # the DASolver
            if self.getOption("useAD")["mode"] == "forward":
                functionValue = self.solverAD.getTimeOpFuncVal(funcName)
            else:
                functionValue = self.solver.getTimeOpFuncVal(funcName)
            funcs[funcName] = functionValue

        return

    def addFamilyGroup(self, groupName, families):
        """
        Add a custom grouping of families called groupName. The groupName
        must be distinct from the existing families. All families must
        in the 'families' list must be present in the mesh file.
        Parameters
        ----------
        groupName : str
            User-supplied custom name for the family groupings
        families : list
            List of string. Family names to combine into the family group
        """

        # Do some error checking
        if groupName in self.families:
            raise Error(
                "The specified groupName '%s' already exists in the mesh file or has already been added." % groupName
            )

        # We can actually allow for nested groups. That is, an entry
        # in families may already be a group added in a previous call.
        indices = []
        for fam in families:
            if fam not in self.families:
                raise Error(
                    "The specified family '%s' for group '%s', does "
                    "not exist in the mesh file or has "
                    "not already been added. The current list of "
                    "families (original and grouped) is: %s" % (fam, groupName, repr(self.families.keys()))
                )

            indices.extend(self.families[fam])

        # It is very important that the list of families is sorted
        # because in fortran we always use a binary search to check if
        # a famID is in the list.
        self.families[groupName] = sorted(np.unique(indices))

    def setMesh(self, mesh):
        """
        Set the mesh object to the aero_solver to do geometric deformations
        Parameters
        ----------
        mesh : MBMesh or USMesh object
            The mesh object for doing the warping
        """

        # Store a reference to the mesh
        self.mesh = mesh

        # Setup External Warping with volume indices
        meshInd = self.getSolverMeshIndices()
        self.mesh.setExternalMeshIndices(meshInd)

        # Set the surface the user has supplied:
        conn, faceSizes = self.getSurfaceConnectivity(self.allWallsGroup)
        pts = self.getSurfaceCoordinates(self.allWallsGroup)
        self.mesh.setSurfaceDefinition(pts, conn, faceSizes)

    def getSurfaceConnectivity(self, groupName=None):
        """
        Return the connectivity of the coordinates at which the forces (or tractions) are
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
            nPts = len(bc["indicesRed"])

            # get the number of reduced faces associated with this boundary
            nFace = len(bc["facesRed"])

            # check that this isn't an empty boundary
            if nFace > 0:
                # loop over the faces and add them to the connectivity and faceSizes array
                for iFace in range(nFace):
                    face = copy.copy(bc["facesRed"][iFace])
                    for i in range(len(face)):
                        face[i] += pointOffset
                    conn.extend(face)
                    faceSizes.append(len(face))

                pointOffset += nPts

        return conn, faceSizes

    def getTriangulatedMeshSurface(self, groupName=None, **kwargs):
        """
        This function returns a trianguled verision of the surface
        mesh on all processors. The intent is to use this for doing
        constraints in DVConstraints.
        Returns
        -------
        surf : list
           List of points and vectors describing the surface. This may
           be passed directly to DVConstraint setSurface() function.
        """

        if groupName is None:
            groupName = self.allWallsGroup

        # Obtain the points and connectivity for the specified
        # groupName
        pts = self.comm.allgather(self.getSurfaceCoordinates(groupName, **kwargs))
        conn, faceSizes = self.getSurfaceConnectivity(groupName)
        conn = np.array(conn).flatten()
        conn = self.comm.allgather(conn)
        faceSizes = self.comm.allgather(faceSizes)

        # Triangle info...point and two vectors
        p0 = []
        v1 = []
        v2 = []

        # loop over the faces
        for iProc in range(len(faceSizes)):

            connCounter = 0
            for iFace in range(len(faceSizes[iProc])):
                # Get the number of nodes on this face
                faceSize = faceSizes[iProc][iFace]
                faceNodes = conn[iProc][connCounter : connCounter + faceSize]

                # Start by getting the centerpoint
                ptSum = [0, 0, 0]
                for i in range(faceSize):
                    # idx = ptCounter+i
                    idx = faceNodes[i]
                    ptSum += pts[iProc][idx]

                avgPt = ptSum / faceSize

                # Now go around the face and add a triangle for each adjacent pair
                # of points. This assumes an ordered connectivity from the
                # meshwarping
                for i in range(faceSize):
                    idx = faceNodes[i]
                    p0.append(avgPt)
                    v1.append(pts[iProc][idx] - avgPt)
                    if i < (faceSize - 1):
                        idxp1 = faceNodes[i + 1]
                        v2.append(pts[iProc][idxp1] - avgPt)
                    else:
                        # wrap back to the first point for the last element
                        idx0 = faceNodes[0]
                        v2.append(pts[iProc][idx0] - avgPt)

                # Now increment the connectivity
                connCounter += faceSize

        return [p0, v1, v2]

    def printFamilyList(self):
        """
        Print a nicely formatted dictionary of the family names
        """
        Info(self.families)

    def _initializeOptions(self, options):
        """
        Initialize the options passed into pyDAFoam

        Parameters
        ----------

        options : dictionary
            The list of options to use with pyDAFoam.
        """

        # If 'options' is None raise an error
        if options is None:
            raise Error("The 'options' keyword argument must be passed pyDAFoam.")

        # set immutable options that users should not change during the optimization
        self.imOptions = self._getImmutableOptions()

        # Load all the option information:
        self.defaultOptions = self._getDefOptions()

        # we need to adjust the default p primalValueBounds for incompressible solvers
        if options["solverName"] in self.solverRegistry["Incompressible"]:
            self.defaultOptions["primalVarBounds"][1]["pMin"] = -50000.0
            self.defaultOptions["primalVarBounds"][1]["pMax"] = 50000.0
            self.defaultOptions["primalVarBounds"][1]["p_rghMin"] = -50000.0
            self.defaultOptions["primalVarBounds"][1]["p_rghMax"] = 50000.0

        # Set options based on defaultOptions
        # we basically overwrite defaultOptions with the given options
        # first assign self.defaultOptions to self.options
        self.options = OrderedDict()
        for key in self.defaultOptions:
            if len(self.defaultOptions[key]) != 2:
                raise Error(
                    "key %s has wrong format! \
                    Example: {'iters' : [int, 1]}"
                    % key
                )
            self.options[key] = self.defaultOptions[key]
        # now set options to self.options
        for key in options:
            self._initOption(key, options[key])

        return

    def _initializeComm(self, comm):
        """
        Initialize MPI COMM and setup parallel flags
        """

        # Set the MPI Communicators and associated info
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm

        # Check whether we are running in parallel
        nProc = self.comm.size
        self.parallel = False
        if nProc > 1:
            self.parallel = True

        # Save the rank and number of processors
        self.rank = self.comm.rank
        self.nProcs = self.comm.size

        # Setup the parallel flag for OpenFOAM executives
        self.parallelFlag = ""
        if self.parallel:
            self.parallelFlag = "-parallel"

        return

    def deformDynamicMesh(self):
        """
        Deform the dynamic mesh and save to them to the disk
        """

        if not self.getOption("dynamicMesh")["active"]:
            return

        Info("Deforming dynamic mesh")

        # if we do not have the volCoord as the input, we need to run this only once
        # otherwise, we need to deform the dynamic mesh for each primal solve
        if self.solver.hasVolCoordInput() == 0:
            # if the mesh has been deformed, return
            if self.dynamicMeshDeformed == 1:
                return

        mode = self.getOption("dynamicMesh")["mode"]

        deltaT = self.solver.getDeltaT()

        endTime = self.solver.getEndTime()
        endTimeIndex = round(endTime / deltaT)
        nLocalPoints = self.solver.getNLocalPoints()

        if mode == "rotation":
            center = self.getOption("dynamicMesh")["center"]
            axis = self.getOption("dynamicMesh")["axis"]
            omega = self.getOption("dynamicMesh")["omega"]

            # always get the initial mesh from OF layer
            points0 = np.zeros(nLocalPoints * 3)
            self.solver.getOFMeshPoints(points0)
            # NOTE: we also write the mesh point for t = 0
            self.solver.writeMeshPoints(points0, 0.0)

            # do a for loop to incrementally deform the mesh by a deltaT
            points = np.reshape(points0, (-1, 3))
            for i in range(1, endTimeIndex + 1):
                t = i * deltaT
                dTheta = omega * deltaT
                dCosTheta = np.cos(dTheta)
                dSinTheta = np.sin(dTheta)

                for pointI in range(nLocalPoints):

                    if axis == "z":
                        xTemp = points[pointI][0] - center[0]
                        yTemp = points[pointI][1] - center[1]

                        points[pointI][0] = dCosTheta * xTemp - dSinTheta * yTemp + center[0]
                        points[pointI][1] = dSinTheta * xTemp + dCosTheta * yTemp + center[1]
                    else:
                        raise Error("axis not valid! Options are: z")

                pointsWrite = points.flatten()
                self.solver.writeMeshPoints(pointsWrite, t)
        else:
            raise Error("mode not valid! Options are: rotation")

        # reset the time
        self.solver.setTime(0.0, 0)
        self.dynamicMeshDeformed = 1

    def readDynamicMeshPoints(self, timeVal, deltaT, timeIndex, ddtSchemeOrder):
        """
        Read the dynamic mesh points saved in the folders 0.001, 0.002
        NOTE: if the backward scheme is used we need to read the mesh
        for 3 time levels to get the correct V0, V00 etc
        NOTE: setting the proper time index is critical because the fvMesh
        will use timeIndex to calculate meshPhi, V0 etc
        """
        if timeVal < deltaT:
            raise Error("timeVal not valid")

        if ddtSchemeOrder == 1:
            # no special treatment
            pass
        elif ddtSchemeOrder == 2:
            # need to read timeVal - 2*deltaT
            time_2 = max(timeVal - 2 * deltaT, 0.0)
            # NOTE: the index can go to negative, just to force the fvMesh to update V0, V00 etc
            index_2 = timeIndex - 2
            self.solver.setTime(time_2, index_2)
            self.solver.readMeshPoints(time_2)
            self.solverAD.setTime(time_2, index_2)
            self.solverAD.readMeshPoints(time_2)
        else:
            raise Error("ddtSchemeOrder not supported")

        # read timeVal - deltaT points
        time_1 = max(timeVal - deltaT, 0.0)
        index_1 = timeIndex - 1
        self.solver.setTime(time_1, index_1)
        self.solver.readMeshPoints(time_1)
        self.solverAD.setTime(time_1, index_1)
        self.solverAD.readMeshPoints(time_1)
        # read timeVal points
        self.solver.setTime(timeVal, timeIndex)
        self.solver.readMeshPoints(timeVal)
        self.solverAD.setTime(timeVal, timeIndex)
        self.solverAD.readMeshPoints(timeVal)

    def readStateVars(self, timeVal, deltaT):
        """
        Read the state variables in to OpenFOAM's state fields
        """

        # read current time
        self.solver.readStateVars(timeVal, 0)
        self.solverAD.readStateVars(timeVal, 0)

        # read old time
        t0 = timeVal - deltaT
        self.solver.readStateVars(t0, 1)
        self.solverAD.readStateVars(t0, 1)

        # read old old time
        t00 = timeVal - 2 * deltaT
        self.solver.readStateVars(t00, 2)
        self.solverAD.readStateVars(t00, 2)

        # assign the state from OF field to wVec so that the wVec
        # is update to date for unsteady adjoint
        # self.solver.ofField2StateVec(self.wVec)

    def set_solver_input(self, inputs, DVGeo=None):
        """
        Set solver input. If it is forward mode, we also set the seeds
        """
        inputDict = self.getOption("inputInfo")

        for inputName in list(inputDict.keys()):
            # this input is attached to solver comp
            if "solver" in inputDict[inputName]["components"]:
                inputType = inputDict[inputName]["type"]
                input = inputs[inputName]
                inputSize = len(input)
                seeds = np.zeros(inputSize)
                if self.getOption("useAD")["mode"] == "forward":
                    if inputType == "volCoord":
                        if self.getOption("useAD")["dvName"] not in list(inputDict.keys()):
                            seeds = self.calcFFD2XvSeeds(DVGeo)
                    else:
                        if inputName == self.getOption("useAD")["dvName"]:
                            seedIndex = self.getOption("useAD")["seedIndex"]
                            seeds[seedIndex] = 1.0

            # here we need to update the solver input for both solver and solverAD
            self.solver.setSolverInput(inputName, inputType, inputSize, input, seeds)
            self.solverAD.setSolverInput(inputName, inputType, inputSize, input, seeds)

    def calcFFD2XvSeeds(self, DVGeo=None):
        # Calculate the FFD2XvSeed array:
        # Given a FFD seed xDvDot, run pyGeo and IDWarp and propagate the seed to Xv seed xVDot:
        #     xSDot = \\frac{dX_{S}}{dX_{DV}}\\xDvDot
        #     xVDot = \\frac{dX_{V}}{dX_{S}}\\xSDot

        # Then, we assign this vector to FFD2XvSeed in the mphys_dafoam
        # This will be used in forward mode AD runs

        discipline = self.getOption("discipline")

        if DVGeo is None:
            raise Error("calcFFD2XvSeeds is call but no DVGeo object found! Call add_dvgeo in the run script!")

        if self.mesh is None:
            raise Error("calcFFD2XvSeeds is call but no mesh object found!")

        dvName = self.getOption("useAD")["dvName"]
        seedIndex = self.getOption("useAD")["seedIndex"]
        # create xDVDot vec and initialize it with zeros
        xDV = DVGeo.getValues()

        # create a copy of xDV and set the seed to 1.0
        # the dv and index depends on dvName and seedIndex
        xDvDot = {}
        for key in list(xDV.keys()):
            xDvDot[key] = np.zeros_like(xDV[key])
        xDvDot[dvName][seedIndex] = 1.0

        # get the original surf coords
        xs0 = self.getSurfaceCoordinates(self.allSurfacesGroup)
        xSDot0 = np.zeros_like(xs0)
        xSDot0 = self.mapVector(xSDot0, self.allSurfacesGroup, self.designSurfacesGroup)

        # get xSDot
        xSDot = DVGeo.totalSensitivityProd(xDvDot, ptSetName="x_%s0" % discipline).reshape(xSDot0.shape)
        # get xVDot
        xVDot = self.mesh.warpDerivFwd(xSDot)

        return xVDot

    def _initSolver(self):
        """
        Initialize the solvers. This needs to be called before calling any runs
        """

        if self.solverInitialized == 1:
            raise Error("pyDAFoam: self._initSolver has been called! One shouldn't initialize solvers twice!")

        solverName = self.getOption("solverName")
        solverArg = solverName + " -python " + self.parallelFlag

        from .libs.pyDASolvers import pyDASolvers

        self.solver = pyDASolvers(solverArg.encode(), self.options)

        if self.getOption("useAD")["mode"] == "forward":

            from .libs.ADF.pyDASolvers import pyDASolvers as pyDASolversAD

            self.solverAD = pyDASolversAD(solverArg.encode(), self.options)

        elif self.getOption("useAD")["mode"] == "reverse":

            from .libs.ADR.pyDASolvers import pyDASolvers as pyDASolversAD

            self.solverAD = pyDASolversAD(solverArg.encode(), self.options)

        self.solver.initSolver()
        self.solverAD.initSolver()

        Info("Init solver done! ElapsedClockTime %f s" % self.solver.getElapsedClockTime())
        Info("Init solver done! ElapsedCpuTime %f s" % self.solver.getElapsedCpuTime())

        self.solverInitialized = 1

        return

    def runDecomposePar(self):
        """
        Run decomposePar to parallel run
        """

        # don't run it if it is a serial case
        if self.comm.size == 1:
            return

        # write the decomposeParDict file with the correct numberOfSubdomains number
        self._writeDecomposeParDict()

        command = ["decomposePar"]
        args = self.getOption("decomposeParDict")["args"]

        for arg in args:
            if arg != "None":
                command.append(arg)

        if self.comm.rank == 0:
            status = subprocess.call(command, stdout=sys.stdout, stderr=subprocess.STDOUT, shell=False)
            if status != 0:
                # raise Error('pyDAFoam: status %d: Unable to run decomposePar'%status)
                print("\nUnable to run decomposePar, the domain has been already decomposed?\n", flush=True)
        self.comm.Barrier()

        return

    def deletePrevPrimalSolTime(self):
        """
        Delete the previous primal solution time folder
        """

        solTime = self.solver.getPrevPrimalSolTime()

        rootDir = os.getcwd()
        if self.parallel:
            checkPath = os.path.join(rootDir, "processor%d/%g" % (self.comm.rank, solTime))
        else:
            checkPath = os.path.join(rootDir, "%g" % solTime)

        if os.path.isdir(checkPath):
            try:
                shutil.rmtree(checkPath)
            except Exception:
                raise Error("Can not delete %s" % checkPath)

            Info("Previous solution time %g found and deleted." % solTime)
        else:
            Info("Previous solution time %g not found and nothing deleted." % solTime)

        return

    def renameSolution(self, solIndex):
        """
        Rename the primal solution folder to specific format for post-processing. The renamed time has the
        format like 0.0001, 0.0002, etc. One can load these intermediate shapes and fields and
        plot them in paraview.
        The way it is implemented is that we sort the solution folder and consider the largest time folder
        as the solution folder and rename it

        Parameters
        ----------
        solIndex: int
            The major interation index
        """

        rootDir = os.getcwd()
        if self.parallel:
            checkPath = os.path.join(rootDir, "processor%d" % self.comm.rank)
        else:
            checkPath = rootDir

        latestTime = self.solver.getLatestTime()

        if latestTime < 1.0:
            Info("Latest solution time %g less than 1, not renamed." % latestTime)
            renamed = False
            return latestTime, renamed

        distTime = "%g" % (solIndex / 1e4)
        targetTime = "%g" % latestTime

        src = os.path.join(checkPath, targetTime)
        dst = os.path.join(checkPath, distTime)

        Info("Moving time %s to %s" % (targetTime, distTime))

        if os.path.isdir(dst):
            raise Error("%s already exists, moving failed!" % dst)
        else:
            try:
                shutil.move(src, dst)
            except Exception:
                raise Error("Can not move %s to %s" % (src, dst))

        renamed = True
        return distTime, renamed

    def _readMeshInfo(self):
        """
        Initialize mesh information and read mesh information
        """

        dirName = os.getcwd()

        self.fileNames, self.xv0, self.faces, self.boundaries, self.owners, self.neighbours = self._readOFGrid(dirName)
        self.xv = copy.copy(self.xv0)

        return

    def setSurfaceCoordinates(self, coordinates, groupName=None):
        """
        Set the updated surface coordinates for a particular group.
        Parameters
        ----------
        coordinates : numpy array
            Numpy array of size Nx3, where N is the number of coordinates on this processor.
            This array must have the same shape as the array obtained with getSurfaceCoordinates()
        groupName : str
            Name of family or group of families for which to return coordinates for.
        """
        if self.mesh is None:
            return

        if groupName is None:
            groupName = self.allWallsGroup

        self._updateGeomInfo = True
        if self.mesh is None:
            raise Error("Cannot set new surface coordinate locations without a mesh" "warping object present.")

        # First get the surface coordinates of the meshFamily in case
        # the groupName is a subset, those values will remain unchanged.

        meshSurfCoords = self.getSurfaceCoordinates(self.allWallsGroup)
        meshSurfCoords = self.mapVector(coordinates, groupName, self.allWallsGroup, meshSurfCoords)

        self.mesh.setSurfaceCoordinates(meshSurfCoords)

    def getSurfaceCoordinates(self, groupName=None):
        """
        Return the coordinates for the surfaces defined by groupName.

        Parameters
        ----------
        groupName : str
            Group identifier to get only coordinates cooresponding to
            the desired group. The group must be a family or a
            user-supplied group of families. The default is None which
            corresponds to all wall-type surfaces.

        Output
        ------
        xs: numpy array of size nPoints * 3 for surface points
        """

        if groupName is None:
            groupName = self.allWallsGroup

        # Get the required size
        npts, ncell = self._getSurfaceSize(groupName)
        xs = np.zeros((npts, 3), self.dtype)

        # loop over the families in this group and populate the surface
        famInd = self.families[groupName]
        counter = 0
        for Ind in famInd:
            name = self.basicFamilies[Ind]
            bc = self.boundaries[name]
            for ptInd in bc["indicesRed"]:
                xs[counter, :] = self.xv[ptInd]
                counter += 1

        return xs

    def _getSurfaceSize(self, groupName):
        """
        Internal routine to return the size of a particular surface. This
        does *NOT* set the actual family group
        """
        if groupName is None:
            groupName = self.allSurfacesGroup

        if groupName not in self.families:
            raise Error(
                "'%s' is not a family in the OpenFoam Case or has not been added"
                " as a combination of families" % groupName
            )

        # loop over the basic surfaces in the family group and sum up the number of
        # faces and nodes

        famInd = self.families[groupName]
        nPts = 0
        nCells = 0
        for Ind in famInd:
            name = self.basicFamilies[Ind]
            bc = self.boundaries[name]
            nCells += len(bc["facesRed"])
            nPts += len(bc["indicesRed"])

        return nPts, nCells

    def setPrimalBoundaryConditions(self, printInfo=1, printInfoAD=0):
        """
        Assign the boundary condition defined in primalBC to the OF fields
        """
        self.solver.setPrimalBoundaryConditions(printInfo)
        self.solverAD.setPrimalBoundaryConditions(printInfoAD)

    def _computeBasicFamilyInfo(self):
        """
        Loop over the boundary data and compute necessary family
        information for the basic patches

        """
        # get the list of basic families
        self.basicFamilies = sorted(self.boundaries.keys())

        # save and return a list of the wall boundaries
        self.wallList = []
        counter = 0
        # for each boundary, figure out the unique list of volume node indices it uses
        for name in self.basicFamilies:
            # setup the basic families dictionary
            self.families[name] = [counter]
            counter += 1

            # Create a handle for this boundary
            bc = self.boundaries[name]

            # get the number of faces associated with this boundary
            nFace = len(bc["faces"])

            # create the point index list
            indices = []

            # check that this isn't an empty boundary
            if nFace > 0:
                for iFace in bc["faces"]:
                    # get the node information for the current face
                    face = self.faces[iFace]
                    indices.extend(face)

            # Get the unique point entries for this boundary
            indices = np.unique(indices)

            # now create the reverse dictionary to connect the reduced set with the original
            inverseInd = {}
            for i in range(len(indices)):
                inverseInd[indices[i]] = i

            # Now loop back over the faces and store the connectivity in terms of the reduces index set
            # Here facesRed store the boundary face reduced-point-index
            # For example,
            # 'indicesRed': [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80] <- unique point index for this boundary
            # 'facesRed': [0, 8, 9, 1], [1, 9, 10, 2] <- Here 0 means the 0th point index (0) in indicesRed, and 8 means the 8th
            # point index (64) in incidexRed. So [0 8 9 1] corresponds to the face [0 64 72 8] in the original point index system.
            # NOTE: using the reduce face indexing will faciliate the connectivity calls
            facesRed = []
            for iFace in bc["faces"]:
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

            # Check that the length of faces and facesRed are equal
            if not (len(bc["faces"]) == len(facesRed)):
                raise Error("Connectivity for faces on reduced index set is not the same length as original.")

            # put the reduced faces and index list in the boundary dict
            bc["facesRed"] = facesRed
            bc["indicesRed"] = list(indices)

            # now check for walls
            if bc["type"] == "wall" or bc["type"] == "slip" or bc["type"] == "cyclic":
                self.wallList.append(name)

        return

    def getSolverMeshIndices(self):
        """
        Get the list of indices to pass to the mesh object for the
        volume mesh mapping
        """

        # Setup External Warping
        nCoords = len(self.xv0.flatten())

        nCoords = self.comm.allgather(nCoords)
        offset = 0
        for i in range(self.comm.rank):
            offset += nCoords[i]

        meshInd = np.arange(nCoords[self.comm.rank]) + offset

        return meshInd

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
            The input vector mapped to the families defined in groupName2.
        """
        if groupName1 not in self.families or groupName2 not in self.families:
            raise Error(
                "'%s' or '%s' is not a family in the mesh file or has not been added"
                " as a combination of families" % (groupName1, groupName2)
            )

        # Shortcut:
        if groupName1 == groupName2:
            return vec1

        if vec2 is None:
            npts, ncell = self._getSurfaceSize(groupName2)
            vec2 = np.zeros((npts, 3), self.dtype)

        famList1 = self.families[groupName1]
        famList2 = self.families[groupName2]

        """
        This functionality is predicated on the surfaces being traversed in the
        same order every time. Loop over the allfamilies list, keeping track of sizes
        as we go and if the family is in both famLists, copy the values from vec1 to vec2.

        """

        vec1counter = 0
        vec2counter = 0

        for ind in self.families[self.allSurfacesGroup]:
            npts, ncell = self._getSurfaceSize(self.basicFamilies[ind])

            if ind in famList1 and ind in famList2:
                vec2[vec2counter : npts + vec2counter] = vec1[vec1counter : npts + vec1counter]

            if ind in famList1:
                vec1counter += npts

            if ind in famList2:
                vec2counter += npts

        return vec2

    # base case files
    def _readOFGrid(self, caseDir):
        """
        Read in the mesh information we need to run the case using pyofm

        Parameters
        ----------
        caseDir : str
            The directory containing the openFOAM Mesh files
        """

        Info("Reading OpenFOAM mesh information...")

        from pyofm import PYOFM

        # Initialize pyOFM
        ofm = PYOFM(comm=self.comm)

        # generate the file names
        fileNames = ofm.getFileNames(caseDir, comm=self.comm)

        # Read in the volume points
        x0 = ofm.readVolumeMeshPoints()

        # Read the face info for the mesh
        faces = ofm.readFaceInfo()

        # Read the boundary info
        boundaries = ofm.readBoundaryInfo(faces)

        # Read the cell info for the mesh
        owners, neighbours = ofm.readCellInfo()

        return fileNames, x0, faces, boundaries, owners, neighbours

    def setOption(self, name, value):
        """
        Set a value to options.
        NOTE: calling this function will only change the values in self.options
        It will NOT change values for allOptions_ in DAOption. To make the options changes
        from pyDAFoam to DASolvers, call self.updateDAOption()

        Parameters
        ----------
        name : str
           Name of option to set. Not case sensitive
        value : varies
           Value to set. Type is checked for consistency.

        Examples
        --------
        If self.options reads:
        self.options =
        {
            'solverName': [str, 'DASimpleFoam'],
            'flowEndTime': [float, 1.0]
        }
        Then, calling self.options('solverName', 'DARhoSimpleFoam') will give:
        self.options =
        {
            'solverName': [str, 'DARhoSimpleFoam'],
            'flowEndTime': [float, 1.0]
        }


        NOTE: if 'value' is of dict type, we will set all the subKey values in
        'value' dict to self.options, instead of overiding it. This works for
        only THREE levels of subDicts

        For example, if self.options reads
        self.options =
        {
            'function': [dict, {
                'name': 'CD',
                'direction': [1.0, 0.0, 0.0],
                'scale': 1.0}]
        }

        Then, calling self.setOption('function', {'name': 'CL'}) will give:

        self.options =
        {
            'function': [dict, {
                'name': 'CL',
                'direction': [1.0, 0.0, 0.0],
                'scale': 1.0}]
        }

        INSTEAD OF

        self.options =
        {
            'function': [dict, {'name': 'CL'}]
        }
        """

        try:
            self.defaultOptions[name]
        except KeyError:
            Error("Option '%-30s' is not a valid %s option." % (name, self.name))

        # Make sure we are not trying to change an immutable option if
        # we are not allowed to.
        if name in self.imOptions:
            raise Error("Option '%-35s' cannot be modified after the solver " "is created." % name)

        # Now we know the option exists, lets check if the type is ok:
        if isinstance(value, self.defaultOptions[name][0]):
            # the type matches, now we need to check if the 'value' is of dict type, if yes, we only
            # replace the subKey values of 'value', instead of overiding all the subKey values
            # NOTE. we only check 3 levels of subKeys
            if isinstance(value, dict):
                for subKey1 in value:
                    # check if this subKey is still a dict.
                    if isinstance(value[subKey1], dict):
                        for subKey2 in value[subKey1]:
                            # check if this subKey is still a dict.
                            if isinstance(value[subKey1][subKey2], dict):
                                for subKey3 in value[subKey1][subKey2]:
                                    self.options[name][1][subKey1][subKey2][subKey3] = value[subKey1][subKey2][subKey3]
                            else:
                                self.options[name][1][subKey1][subKey2] = value[subKey1][subKey2]
                    else:
                        # no need to set self.options[name][0] since it has the right type
                        self.options[name][1][subKey1] = value[subKey1]
            else:
                # It is not dict, just set
                # no need to set self.options[name][0] since it has the right type
                self.options[name][1] = value
        else:
            raise Error(
                "Datatype for Option %-35s was not valid \n "
                "Expected data type is %-47s \n "
                "Received data type is %-47s" % (name, self.defaultOptions[name][0], type(value))
            )

    def _initOption(self, name, value):
        """
        Set a value to options. This function will be used only for initializing the options internally.
        Do NOT call this function from the run script!

        Parameters
        ----------
        name : str
           Name of option to set. Not case sensitive
        value : varies
           Value to set. Type is checked for consistency.
        """

        try:
            self.defaultOptions[name]
        except KeyError:
            Error("Option '%-30s' is not a valid %s option." % (name, self.name))

        # Make sure we are not trying to change an immutable option if
        # we are not allowed to.
        if name in self.imOptions:
            raise Error("Option '%-35s' cannot be modified after the solver " "is created." % name)

        # Now we know the option exists, lets check if the type is ok:
        if isinstance(value, self.defaultOptions[name][0]):
            # the type matches, now we need to check if the 'value' is of dict type, if yes, we only
            # replace the subKey values of 'value', instead of overiding all the subKey values
            if isinstance(value, dict):
                for subKey in value:
                    # no need to set self.options[name][0] since it has the right type
                    self.options[name][1][subKey] = value[subKey]
            else:
                # It is not dict, just set
                # no need to set self.options[name][0] since it has the right type
                self.options[name][1] = value
        else:
            raise Error(
                "Datatype for Option %-35s was not valid \n "
                "Expected data type is %-47s \n "
                "Received data type is %-47s" % (name, self.defaultOptions[name][0], type(value))
            )

    def getNRegressionParameters(self, modelName):
        """
        Get the number of regression model parameters
        """
        nParameters = self.solver.getNRegressionParameters(modelName.encode())
        return nParameters

    def getOption(self, name):
        """
        Get a value from options

        Parameters
        ----------
        name : str
           Name of option to get. Not case sensitive

        Returns
        -------
        value : varies
           Return the value of the option.
        """

        if name in self.defaultOptions:
            return self.options[name][1]
        else:
            raise Error("%s is not a valid option name." % name)

    def updateDAOption(self):
        """
        Update the allOptions_ in DAOption based on the latest self.options in
        pyDAFoam. This will pass the changes of self.options from pyDAFoam
        to DASolvers. NOTE: need to call this function after calling
        self.initSolver
        """

        if self.solverInitialized == 0:
            raise Error("self._initSolver not called!")

        self.solver.updateDAOption(self.options)

        if self.getOption("useAD")["mode"] in ["forward", "reverse"]:
            self.solverAD.updateDAOption(self.options)

    def getNLocalAdjointStates(self):
        """
        Get number of local adjoint states
        """
        return self.solver.getNLocalAdjointStates()

    def getNLocalPoints(self):
        """
        Get number of local points
        """
        return self.solver.getNLocalPoints()

    def getStates(self):
        """
        Return the adjoint state array owns by this processor
        """

        nLocalStateSize = self.solver.getNLocalAdjointStates()
        states = np.zeros(nLocalStateSize, self.dtype)
        self.solver.getOFFields(states)

        return states

    def setStates(self, states):
        """
        Set the state to the OpenFOAM field
        """

        self.solver.updateOFFields(states)
        self.solverAD.updateOFFields(states)

        return

    def setVolCoords(self, vol_coords):
        """
        Set the vol_coords to the OpenFOAM field
        """

        self.solver.updateOFMesh(vol_coords)
        self.solverAD.updateOFMesh(vol_coords)

        return

    def arrayVal2Vec(self, array1, vec):
        """
        Assign the values from array1 to vec
        """

        size = len(array1)

        Istart, Iend = vec.getOwnershipRange()

        if (Iend - Istart) != size:
            raise Error("array and vec's sizes are not consistent")

        for i in range(Istart, Iend):
            iRel = i - Istart
            vec[i] = array1[iRel]

        vec.assemblyBegin()
        vec.assemblyEnd()

    def vecVal2Array(self, vec, array1):
        """
        Assign the values from vec to array1
        """

        size = len(array1)

        Istart, Iend = vec.getOwnershipRange()

        if (Iend - Istart) != size:
            raise Error("array and vec's sizes are not consistent")

        for i in range(Istart, Iend):
            iRel = i - Istart
            array1[iRel] = vec[i]

    def vec2Array(self, vec):
        """
        Convert a Petsc vector to numpy array
        """

        Istart, Iend = vec.getOwnershipRange()
        size = Iend - Istart
        array1 = np.zeros(size, self.dtype)
        for i in range(Istart, Iend):
            iRel = i - Istart
            array1[iRel] = vec[i]
        return array1

    def array2Vec(self, array1):
        """
        Convert a numpy array to Petsc vector
        """
        size = len(array1)

        vec = PETSc.Vec().create(PETSc.COMM_WORLD)
        vec.setSizes((size, PETSc.DECIDE), bsize=1)
        vec.setFromOptions()
        vec.zeroEntries()

        Istart, Iend = vec.getOwnershipRange()
        for i in range(Istart, Iend):
            iRel = i - Istart
            vec[i] = array1[iRel]

        vec.assemblyBegin()
        vec.assemblyEnd()

        return vec

    def _getImmutableOptions(self):
        """
        We define the list of options that *cannot* be changed after the
        object is created. pyDAFoam will raise an error if a user tries to
        change these. The strings for these options are placed in a set
        """

        return ()

    def _writeDecomposeParDict(self):
        """
        Write system/decomposeParDict
        """
        if self.comm.rank == 0:
            # Open the options file for writing

            workingDirectory = os.getcwd()
            sysDir = "system"
            varDir = os.path.join(workingDirectory, sysDir)
            fileName = "decomposeParDict"
            fileLoc = os.path.join(varDir, fileName)
            f = open(fileLoc, "w")
            # write header
            self._writeOpenFoamHeader(f, "dictionary", sysDir, fileName)
            # write content
            decomDict = self.getOption("decomposeParDict")
            n = decomDict["simpleCoeffs"]["n"]
            f.write("numberOfSubdomains     %d;\n" % self.nProcs)
            f.write("\n")
            f.write("method                 %s;\n" % decomDict["method"])
            f.write("\n")
            f.write("simpleCoeffs \n")
            f.write("{ \n")
            f.write("    n                  (%d %d %d);\n" % (n[0], n[1], n[2]))
            f.write("    delta              %g;\n" % decomDict["simpleCoeffs"]["delta"])
            f.write("} \n")
            f.write("\n")
            f.write("distributed            false;\n")
            f.write("\n")
            f.write("roots();\n")
            if len(decomDict["preservePatches"]) == 1 and decomDict["preservePatches"][0] == "None":
                pass
            else:
                f.write("\n")
                f.write("preservePatches        (")
                for pPatch in decomDict["preservePatches"]:
                    f.write("%s " % pPatch)
                f.write(");\n")
            if decomDict["singleProcessorFaceSets"][0] != "None":
                f.write("singleProcessorFaceSets  (")
                for pPatch in decomDict["singleProcessorFaceSets"]:
                    f.write(" (%s -1) " % pPatch)
                f.write(");\n")
            f.write("\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n")

            f.close()
        self.comm.Barrier()

    def _writeOpenFoamHeader(self, f, className, location, objectName):
        """
        Write OpenFOAM header file
        """

        f.write("/*--------------------------------*- C++ -*---------------------------------*\ \n")
        f.write("| ========                 |                                                 | \n")
        f.write("| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n")
        f.write("|  \\    /   O peration     | Version:  v1812                                 | \n")
        f.write("|   \\  /    A nd           | Web:      www.OpenFOAM.com                      | \n")
        f.write("|    \\/     M anipulation  |                                                 | \n")
        f.write("\*--------------------------------------------------------------------------*/ \n")
        f.write("FoamFile\n")
        f.write("{\n")
        f.write("    version     2.0;\n")
        f.write("    format      ascii;\n")
        f.write("    class       %s;\n" % className)
        f.write('    location    "%s";\n' % location)
        f.write("    object      %s;\n" % objectName)
        f.write("}\n")
        f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n")
        f.write("\n")


class Error(Exception):
    """
    Format the error message in a box to make it clear this
    was a expliclty raised exception.
    """

    def __init__(self, message):
        msg = "\n+" + "-" * 78 + "+" + "\n" + "| pyDAFoam Error: "
        i = 19
        for word in message.split():
            if len(word) + i + 1 > 78:  # Finish line and start new one
                msg += " " * (78 - i) + "|\n| " + word + " "
                i = 1 + len(word) + 1
            else:
                msg += word + " "
                i += len(word) + 1
        msg += " " * (78 - i) + "|\n" + "+" + "-" * 78 + "+" + "\n"
        print(msg, flush=True)
        Exception.__init__(self)

        return


class Info(object):
    """
    Print information and flush to screen for parallel cases
    """

    def __init__(self, message):
        if MPI.COMM_WORLD.rank == 0:
            print(message, flush=True)
        MPI.COMM_WORLD.Barrier()


class TensorFlowHelper:
    """
    TensorFlow helper class.
    NOTE: this is a static class because the callback function
    does not accept non-static class members (seg fault)
    """

    options = {}

    model = {}

    modelName = None

    predictBatchSize = {}

    nInputs = {}

    @staticmethod
    def initialize():
        """
        Initialize parameters and load models
        """
        Info("Initializing the TensorFlowHelper")
        for key in list(TensorFlowHelper.options.keys()):
            if key != "active":
                modelName = key
                TensorFlowHelper.predictBatchSize[modelName] = TensorFlowHelper.options[modelName]["predictBatchSize"]
                TensorFlowHelper.nInputs[modelName] = TensorFlowHelper.options[modelName]["nInputs"]
                TensorFlowHelper.model[modelName] = tf.keras.models.load_model(modelName)

    @staticmethod
    def setModelName(modelName):
        """
        Set the model name from the C++ to Python layer
        """
        TensorFlowHelper.modelName = modelName.decode()

    @staticmethod
    def predict(inputs, n, outputs, m):
        """
        Calculate the outputs based on the inputs using the saved model
        """

        modelName = TensorFlowHelper.modelName
        nInputs = TensorFlowHelper.nInputs[modelName]

        inputs_tf = np.reshape(inputs, (-1, nInputs))
        batchSize = TensorFlowHelper.predictBatchSize[modelName]
        outputs_tf = TensorFlowHelper.model[modelName].predict(inputs_tf, verbose=False, batch_size=batchSize)

        for i in range(m):
            outputs[i] = outputs_tf[i, 0]

    @staticmethod
    def calcJacVecProd(inputs, inputs_b, n, outputs, outputs_b, m):
        """
        Calculate the gradients of the outputs wrt the inputs
        """

        modelName = TensorFlowHelper.modelName
        nInputs = TensorFlowHelper.nInputs[modelName]

        inputs_tf = np.reshape(inputs, (-1, nInputs))
        inputs_tf_var = tf.Variable(inputs_tf, dtype=tf.float32)

        with tf.GradientTape() as tape:
            outputs_tf = TensorFlowHelper.model[modelName](inputs_tf_var)

        gradients_tf = tape.gradient(outputs_tf, inputs_tf_var)

        for i in range(gradients_tf.shape[0]):
            for j in range(gradients_tf.shape[1]):
                idx = i * gradients_tf.shape[1] + j
                inputs_b[idx] = gradients_tf.numpy()[i, j] * outputs_b[i]
