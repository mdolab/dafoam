#!/usr/bin/env python

"""

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
    The Python interface to DAFoam. It controls the adjoint
    solvers and external modules for design optimization

"""

__version__ = "3.0.5"

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

        ## Information on objective function. Each objective function requires a different input forma
        ## But for all objectives, we need to give a name to the objective function, e.g., CD or any
        ## other preferred name, and the information for each part of the objective function. Most of
        ## the time, the objective has only one part (in this case part1), but one can also combine two
        ## parts of objectives, e.g., we can define a new objective that is the sum of force and moment.
        ## For each part, we need to define the type of objective (e.g., force, moment; we need to use
        ## the reserved type names), how to select the discrete mesh faces to compute the objective
        ## (e.g., we select them from the name of a patch patchToFace), the name of the patch (wing)
        ## for patchToFace, the scaling factor "scale", and whether to compute adjoint for this
        ## objective "addToAdjoint". For force objectives, we need to project the force vector to a
        ## specific direction. The following example defines that CD is the force that is parallel to flow
        ## (parallelToFlow). Alternative, we can also use fixedDirection and provide a direction key for
        ## force, i.e., "directionMode": "fixedDirection", "direction": [1.0, 0.0, 0.0]. Since we select
        ## parallelToFlow, we need to prescribe the name of angle of attack design variable to determine
        ## the flow direction. Here alpha will be defined in runScript.py:
        ## DVGeo.addGeoDVGlobal("alpha", [alpha0], alpha, lower=-10.0, upper=10.0, scale=1.0).
        ## NOTE: if no alpha is added in DVGeo.addGeoDVGlobal, we can NOT use parallelToFlow.
        ## For this case, we have to use "directionMode": "fixedDirection".
        ## Example
        ##     "objFunc": {
        ##         "CD": {
        ##             "part1": {
        ##                 "type": "force",
        ##                 "source": "patchToFace",
        ##                 "patches": ["wing"],
        ##                 "directionMode": "parallelToFlow",
        ##                 "alphaName": "alpha",
        ##                 "scale": 1.0 / (0.5 * UmagIn * UmagIn * ARef),
        ##                 "addToAdjoint": True,
        ##             }
        ##         },
        ##         "CL": {
        ##             "part1": {
        ##                 "type": "force",
        ##                 "source": "patchToFace",
        ##                 "patches": ["wing"],
        ##                 "directionMode": "normalToFlow",
        ##                 "alphaName": "alpha",
        ##                 "scale": 1.0 / (0.5 * UmagIn * UmagIn * ARef),
        ##                 "addToAdjoint": True,
        ##             }
        ##         },
        ##         "CMZ": {
        ##             "part1": {
        ##                 "type": "moment",
        ##                 "source": "patchToFace",
        ##                 "patches": ["wing"],
        ##                 "axis": [0.0, 0.0, 1.0],
        ##                 "center": [0.25, 0.0, 0.05],
        ##                 "scale": 1.0 / (0.5 * UmagIn * UmagIn * ARef * LRef),
        ##                 "addToAdjoint": True,
        ##             }
        ##         },
        ##         "TPR": {
        ##             "part1": {
        ##                 "type": "totalPressureRatio",
        ##                 "source": "patchToFace",
        ##                 "patches": ["inlet", "outlet"],
        ##                 "inletPatches": ["inlet"],
        ##                 "outletPatches": ["outlet"],
        ##                 "scale": 1.0,
        ##                 "addToAdjoint": True,
        ##             }
        ##         },
        ##         "TTR": {
        ##             "part1": {
        ##                 "type": "totalTemperatureRatio",
        ##                 "source": "patchToFace",
        ##                 "patches": ["inlet", "outlet"],
        ##                 "inletPatches": ["inlet"],
        ##                 "outletPatches": ["outlet"],
        ##                 "scale": 1.0,
        ##                 "addToAdjoint": False,
        ##             }
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
        ##        "PL": {
        ##            "part1": {
        ##                "type": "totalPressure",
        ##                "source": "patchToFace",
        ##                "patches": ["inlet"],
        ##                "scale": 1.0 / (0.5 * U0 * U0),
        ##                "addToAdjoint": True,
        ##            },
        ##            "part2": {
        ##                "type": "totalPressure",
        ##                "source": "patchToFace",
        ##                "patches": ["outlet"],
        ##                "scale": -1.0 / (0.5 * U0 * U0),
        ##                "addToAdjoint": True,
        ##            }
        ##        },
        ##        "NU": {
        ##            "part1": {
        ##                "type": "wallHeatFlux",
        ##                "source": "patchToFace",
        ##                "patches": ["ubend"],
        ##                "scale": 1.0,
        ##                "addToAdjoint": True,
        ##            }
        ##        },
        ##        "VMS": {
        ##            "part1": {
        ##                "type": "vonMisesStressKS",
        ##                "source": "boxToCell",
        ##                "min": [-10.0, -10.0, -10.0],
        ##                "max": [10.0, 10.0, 10.0],
        ##                "scale": 1.0,
        ##                "coeffKS": 2.0e-3,
        ##                "addToAdjoint": True,
        ##            }
        ##        },
        ##        "M": {
        ##            "part1": {
        ##                "type": "mass",
        ##                "source": "boxToCell",
        ##                "min": [-10.0, -10.0, -10.0],
        ##                "max": [10.0, 10.0, 10.0],
        ##                "scale": 1.0,
        ##                "addToAdjoint": True,
        ##            }
        ##        },
        ##        "THRUST": {
        ##            "part1": {
        ##                "type": "variableVolSum",
        ##                "source": "boxToCell",
        ##                "min": [-50.0, -50.0, -50.0],
        ##                "max": [50.0, 50.0, 50.0],
        ##                "varName": "fvSource",
        ##                "varType": "vector",
        ##                "component": 0,
        ##                "isSquare": 0,
        ##                "scale": 1.0,
        ##                "addToAdjoint": True,
        ##            },
        ##        },
        ##        "FI": {
        ##            "part1": {
        ##                "type": "stateErrorNorm",
        ##                "source": "boxToCell",
        ##                "min": [-100.0, -100.0, -100.0],
        ##                "max": [100.0, 100.0, 100.0],
        ##                "stateName": "U",
        ##                "stateRefName": "UTrue",
        ##                "stateType": "vector",
        ##                "scale": 1.0,
        ##                "addToAdjoint": True,
        ##            },
        ##            "part2": {
        ##                "type": "stateErrorNorm",
        ##                "source": "boxToCell",
        ##                "min": [-100.0, -100.0, -100.0],
        ##                "max": [100.0, 100.0, 100.0],
        ##                "stateName": "betaSA",
        ##                "stateRefName": "betaSATrue",
        ##                "stateType": "scalar",
        ##                "scale": 0.01,
        ##                "addToAdjoint": True,
        ##            },
        ##        },
        ##        "COP": {
        ##            "part1": {
        ##                "type": "centerOfPressure",
        ##                "source": "patchToFace",
        ##                "patches": ["wing"],
        ##                "axis": [1.0, 0.0, 0.0],
        ##                "forceAxis": [0.0, 1.0, 0.0],
        ##                "center": [0, 0, 0],
        ##                "scale": 1.0,
        ##                "addToAdjoint": True,
        ##            },
        ##        },
        ##    },
        self.objFunc = {}

        ## Design variable information. Different type of design variables require different keys
        ## For alpha, we need to prescribe a list of far field patch names from which the angle of
        ## attack is computed, this is usually a far field patch. Also, we need to prescribe
        ## flow and normal axies, and alpha = atan( U_normal / U_flow ) at patches
        ## Example
        ##     designVar = {
        ##         "shapey" : {"designVarType": "FFD"},
        ##         "twist": {"designVarType": "FFD"},
        ##         "alpha" = {
        ##             "designVarType": "AOA",
        ##             "patches": ["farField"],
        ##             "flowAxis": "x",
        ##             "normalAxis": "y"
        ##         },
        ##         "ux0" = {
        ##             "designVarType": "BC",
        ##             "patches": ["inlet"],
        ##             "variable": "U",
        ##             "comp": 0
        ##         },
        ##     }
        self.designVar = {}

        ## List of patch names for the design surface. These patch names need to be of wall type
        ## and shows up in the constant/polyMesh/boundary file
        self.designSurfaces = ["ALL_OPENFOAM_WALL_PATCHES"]

        ## MDO coupling information for aerostructural, aerothermal, or aeroacoustic optimization.
        ## We can have ONLY one coupling scenario active, e.g., aerostructural and aerothermal can't be
        ## both active. We can have more than one couplingSurfaceGroups, e.g., wingGroup and tailGroup
        ## or blade1Group, blade2Group, and blade3Group. Each group subdict can have multiple patches.
        ## These patches should be consistent with the patch names defined in constant/polyMesh/boundary
        self.couplingInfo = {
            "aerostructural": {
                "active": False,
                "pRef": 0,
                "propMovement": False,
                "couplingSurfaceGroups": {
                    "wingGroup": ["wing", "wing_te"],
                },
            },
            "aerothermal": {
                "active": False,
                "couplingSurfaceGroups": {
                    "wallGroup": ["fin_wall"],
                },
            },
            "aeroacoustic": {
                "active": False,
                "pRef": 0,
                "couplingSurfaceGroups": {
                    "blade1Group": ["blade1_ps", "blade1_ss"],
                    "blade2Group": ["blade2"],
                },
            },
        }

        ## Aero-propulsive options
        self.aeroPropulsive = {}

        ## An option to run the primal only; no adjoint or optimization will be run
        self.primalOnly = False

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

        ## Whether to perform multipoint optimization.
        self.multiPoint = False

        ## If multiPoint = True, how many primal configurations for the multipoint optimization.
        self.nMultiPoints = 1

        ## The step size for finite-difference computation of partial derivatives. The default values
        ## will work for most of the case.
        self.adjPartDerivFDStep = {
            "State": 1.0e-6,
            "FFD": 1.0e-3,
            "BC": 1.0e-2,
            "AOA": 1.0e-3,
            "ACTP": 1.0e-2,
            "ACTD": 1.0e-2,
            "ACTL": 1.0e-2,
        }

        ## Which options to use to improve the adjoint equation convergence of transonic conditions
        ## This is used only for transonic solvers such as DARhoSimpleCFoam
        self.transonicPCOption = -1

        ## Options for unsteady adjoint. mode can be hybridAdjoint or timeAccurateAdjoint
        ## Here nTimeInstances is the number of time instances and periodicity is the
        ## periodicity of flow oscillation (hybrid adjoint only)
        self.unsteadyAdjoint = {"mode": "None", "nTimeInstances": -1, "periodicity": -1.0}

        ## At which iteration should we start the averaging of objective functions.
        ## This is only used for unsteady solvers
        self.objFuncAvgStart = 1

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

        ## Rigid body motion for dynamic mesh
        ## This option will be used in DAPimpleDyMFoam to simulate dynamicMesh motion
        self.rigidBodyMotion = {"mode": "dummy"}

        # *********************************************************************************************
        # ************************************ Advance Options ****************************************
        # *********************************************************************************************

        ## The run status which can be solvePrimal, solveAdjoint, or calcTotalDeriv. This parameter is
        ## used internally, so users should never change this option in the Python layer.
        self.runStatus = "None"

        ## Whether to print all options defined in pyDAFoam to screen before optimization.
        self.printPYDAFOAMOptions = False

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
        self.printIntervalUnsteady = 500

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
            "dynAdjustTol": True,
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
        }

        ## The ordering of state variable. Options are: state or cell. Most of the case, the state
        ## odering is the best choice.
        self.adjStateOrdering = "state"

        ## Default name for the mesh surface family. Users typically don't need to change
        self.meshSurfaceFamily = "None"

        ## The threshold for check mesh call
        self.checkMeshThreshold = {
            "maxAspectRatio": 1000.0,
            "maxNonOrth": 70.0,
            "maxSkewness": 4.0,
            "maxIncorrectlyOrientedFaces": 0,
        }

        ## The sensitivity map will be saved to disk during optimization for the given design variable
        ## names in the list. Currently only support design variable type FFD and Field
        ## The surface sensitivity map is separated from the primal solution because they only have surface mesh.
        ## They will be saved to folders such as 1e-11, 2e-11, 3e-11, etc,
        ## When loading in paraview, you need to uncheck the "internalMesh", and check "allWalls" on the left panel
        ## If your design variable is of field type, the sensitivity map will be saved along with the primal
        ## solution because they share the same mesh. The sensitivity files read sens_objFuncName_designVarName
        ## NOTE: this function only supports useAD->mode:reverse
        ## Example:
        ##     "writeSensMap" : ["shapex", "shapey"]
        self.writeSensMap = ["NONE"]

        ## Whether to write deformed FFDs to the disk during optimization
        self.writeDeformedFFDs = False

        ## The max number of correctBoundaryConditions calls in the updateOFField function.
        self.maxCorrectBCCalls = 10

        ## Whether to write the primal solutions for minor iterations (i.e., line search).
        ## The default is False. If set it to True, it will write flow fields (and the deformed geometry)
        ## for each primal solution. This will significantly increases the IO runtime, so it should never
        ## be True for production runs. However, it is useful for debugging purpose (e.g., to find out
        ## the poor quality mesh during line search)
        self.writeMinorIterations = False

        ## whether to run the primal using the first order div scheme. This can be used to generate smoother
        ## flow field for computing the preconditioner matrix to avoid singularity. It can help the adjoint
        ## convergence for y+ = 1 meshes. If True, we will run the primal using low order scheme when computing
        ## or updating the PC mat. To enable this option, set "active" to True.
        self.runLowOrderPrimal4PC = {"active": False}

        ## Parameters for wing-propeller coupling optimizations
        self.wingProp = {"nForceSections": 10, "axis": [1.0, 0.0, 0.0], "actEps": 0.02, "rotDir": "right"}


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

        # initialize options for adjoints
        self._initializeOptions(options)

        # initialize comm for parallel communication
        self._initializeComm(comm)

        # check if the combination of options is valid.
        self._checkOptions()

        # Initialize families
        self.families = OrderedDict()

        # Default it to fault, after calling setSurfaceCoordinates, set it to true
        self._updateGeomInfo = False

        # Use double data type: 'd'
        self.dtype = "d"

        # write all the setup files
        self._writeOFCaseFiles()

        # initialize point set name
        self.ptSetName = self.getPointSetName()

        # Remind the user of all the DAFoam options:
        if self.getOption("printPYDAFOAMOptions"):
            self._printCurrentOptions()

        # run decomposePar for parallel runs
        self.runDecomposePar()

        # register solver names and set their types
        self._solverRegistry()

        # initialize the pySolvers
        self.solverInitialized = 0
        self._initSolver()

        # initialize the number of primal and adjoint calls
        self.nSolvePrimals = 1
        self.nSolveAdjoints = 1

        # flags for primal and adjoint failure
        self.primalFail = 0
        self.adjointFail = 0

        # if the primalOnly flag is on, init xvVec and wVec and return
        if self.getOption("primalOnly"):
            self.xvVec = None
            self.wVec = None
            return

        # initialize mesh information and read grids
        self._readMeshInfo()

        # initialize the mesh point vector xvVec
        self._initializeMeshPointVec()

        # initialize state variable vector self.wVec
        self._initializeStateVec()

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

        # Set the couplingSurfacesGroup if any of the MDO scenario is active
        # otherwise, set the couplingSurfacesGroup to designSurfacesGroup
        # NOTE: the treatment of aeroacoustic is different because it supports more than
        # one couplingSurfaceGroups. For other scenarios, only one couplingSurfaceGroup
        # can be defined. TODO. we need to make them consistent in the future..
        couplingInfo = self.getOption("couplingInfo")
        self.couplingSurfacesGroup = self.designSurfacesGroup
        if couplingInfo["aerostructural"]["active"]:
            # we support only one aerostructural surfaceGroup for now
            self.couplingSurfacesGroup = list(couplingInfo["aerostructural"]["couplingSurfaceGroups"].keys())[0]
            patchNames = couplingInfo["aerostructural"]["couplingSurfaceGroups"][self.couplingSurfacesGroup]
            self.addFamilyGroup(self.couplingSurfacesGroup, patchNames)
        elif couplingInfo["aeroacoustic"]["active"]:
            for groupName in couplingInfo["aeroacoustic"]["couplingSurfaceGroups"]:
                self.addFamilyGroup(groupName, couplingInfo["aeroacoustic"]["couplingSurfaceGroups"][groupName])
        elif couplingInfo["aerothermal"]["active"]:
            # we support only one aerothermal coupling surfaceGroup for now
            self.couplingSurfacesGroup = list(couplingInfo["aerothermal"]["couplingSurfaceGroups"].keys())[0]
            patchNames = couplingInfo["aerothermal"]["couplingSurfaceGroups"][self.couplingSurfacesGroup]
            self.addFamilyGroup(self.couplingSurfacesGroup, patchNames)

        # get the surface coordinate of allSurfacesGroup
        self.xs0 = self.getSurfaceCoordinates(self.allSurfacesGroup)

        # By Default we don't have an external mesh object or a
        # geometric manipulation object
        self.mesh = None
        self.DVGeo = None

        # objFuncValuePreIter stores the objective function value from the previous
        # iteration. When the primal solution fails, the evalFunctions function will read
        # value from self.objFuncValuePreIter
        self.objFuncValuePrevIter = {}

        # compute the objective function names for which we solve the adjoint equation
        self.objFuncNames4Adj = self._calcObjFuncNames4Adj()

        # dictionary to save the total derivative vectors
        # NOTE: this function need to be called after initializing self.objFuncNames4Adj
        self.adjTotalDeriv = self._initializeAdjTotalDeriv()

        # preconditioner matrix
        self.dRdWTPC = None

        # a KSP object which may be used outside of the pyDAFoam class
        self.ksp = None

        # the surface geometry/mesh displacement computed by the structural solver
        # this is used in FSI. Here self.surfGeoDisp is a N by 3 numpy array
        # that stores the displacement vector for each surface mesh point. The order of
        # is same as the surface point return by self.getSurfaceCoordinates
        self.surfGeoDisp = None

        # initialize the adjoint vector dict
        self.adjVectors = self._initializeAdjVectors()

        # initialize the dRdWOldTPsi vectors
        self._initializeTimeAccurateAdjointVectors()

        Info("pyDAFoam initialization done!")

        return

    def _solverRegistry(self):
        """
        Register solver names and set their types. For a new solver, first identify its
        type and add their names to the following dict
        """

        self.solverRegistry = {
            "Incompressible": ["DASimpleFoam", "DASimpleTFoam", "DAPisoFoam", "DAPimpleFoam", "DAPimpleDyMFoam"],
            "Compressible": ["DARhoSimpleFoam", "DARhoSimpleCFoam", "DATurboFoam"],
            "Solid": ["DASolidDisplacementFoam", "DALaplacianFoam", "DAScalarTransportFoam"],
        }

    def __call__(self):
        """
        Solve the primal
        """

        # update the mesh coordinates if DVGeo is set
        # add point set and update the mesh based on the DV values

        if self.DVGeo is not None:

            # if the point set is not in DVGeo add it first
            if self.ptSetName not in self.DVGeo.points:

                xs0 = self.mapVector(self.xs0, self.allSurfacesGroup, self.designSurfacesGroup)

                self.DVGeo.addPointSet(xs0, self.ptSetName)
                self.pointsSet = True

            # set the surface coords xs
            Info("DVGeo PointSet UpToDate: " + str(self.DVGeo.pointSetUpToDate(self.ptSetName)))
            if not self.DVGeo.pointSetUpToDate(self.ptSetName):
                Info("Updating DVGeo PointSet....")
                xs = self.DVGeo.update(self.ptSetName, config=None)

                # if we have surface geometry/mesh displacement computed by the structural solver,
                # add the displace mesh here.
                if self.surfGeoDisp is not None:
                    xs += self.surfGeoDisp

                self.setSurfaceCoordinates(xs, self.designSurfacesGroup)
                Info("DVGeo PointSet UpToDate: " + str(self.DVGeo.pointSetUpToDate(self.ptSetName)))

                # warp the mesh to get the new volume coordinates
                Info("Warping the volume mesh....")
                self.mesh.warpMesh()

                xvNew = self.mesh.getSolverGrid()
                self.xvFlatten2XvVec(xvNew, self.xvVec)

            # if it is forward AD mode and we are computing the Xv derivatives
            # call calcFFD2XvSeedVec
            if self.getOption("useAD")["mode"] == "forward":
                dvName = self.getOption("useAD")["dvName"]
                dvType = self.getOption("designVar")[dvName]["designVarType"]
                if dvType == "FFD":
                    self.calcFFD2XvSeedVec()

        # update the primal boundary condition right before calling solvePrimal
        self.setPrimalBoundaryConditions()

        # solve the primal to get new state variables
        self.solvePrimal()

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

    def _initializeAdjVectors(self):
        """
        Initialize the adjoint vector dict

        Returns
        -------

        adjAdjVectors : dict
            A dict that contains adjoint vectors, stored in Petsc format
        """

        wSize = self.solver.getNLocalAdjointStates()

        objFuncDict = self.getOption("objFunc")

        adjVectors = {}
        for objFuncName in objFuncDict:
            if objFuncName in self.objFuncNames4Adj:
                psi = PETSc.Vec().create(PETSc.COMM_WORLD)
                psi.setSizes((wSize, PETSc.DECIDE), bsize=1)
                psi.setFromOptions()
                psi.zeroEntries()
                adjVectors[objFuncName] = psi

        return adjVectors

    def _initializeAdjTotalDeriv(self):
        """
        Initialize the adjoint total derivative dict
        NOTE: this function need to be called after initializing self.objFuncNames4Adj

        Returns
        -------

        adjTotalDeriv : dict
            An empty dict that contains total derivative of objective function with respect design variables
        """

        designVarDict = self.getOption("designVar")
        objFuncDict = self.getOption("objFunc")

        adjTotalDeriv = {}
        for objFuncName in objFuncDict:
            if objFuncName in self.objFuncNames4Adj:
                adjTotalDeriv[objFuncName] = {}
                for designVarName in designVarDict:
                    adjTotalDeriv[objFuncName][designVarName] = None

        return adjTotalDeriv

    def _initializeTimeAccurateAdjointVectors(self):
        """
        Initialize the dRdWTPsi vectors for time accurate adjoint.
        Here we need to initialize current time step and two previous
        time steps 0 and 00 for both state and residuals. This is
        because the backward ddt scheme depends on U, U0, and U00
        """
        if self.getOption("unsteadyAdjoint")["mode"] == "timeAccurateAdjoint":
            objFuncDict = self.getOption("objFunc")
            wSize = self.solver.getNLocalAdjointStates()
            self.dRdW0TPsi = {}
            self.dRdW00TPsi = {}
            self.dR0dW0TPsi = {}
            self.dR0dW00TPsi = {}
            self.dR00dW0TPsi = {}
            self.dR00dW00TPsi = {}
            for objFuncName in objFuncDict:
                if objFuncName in self.objFuncNames4Adj:
                    vecA = PETSc.Vec().create(PETSc.COMM_WORLD)
                    vecA.setSizes((wSize, PETSc.DECIDE), bsize=1)
                    vecA.setFromOptions()
                    vecA.zeroEntries()
                    self.dRdW0TPsi[objFuncName] = vecA

                    vecB = vecA.duplicate()
                    vecB.zeroEntries()
                    self.dRdW00TPsi[objFuncName] = vecB

                    vecC = vecA.duplicate()
                    vecC.zeroEntries()
                    self.dR0dW0TPsi[objFuncName] = vecC

                    vecD = vecA.duplicate()
                    vecD.zeroEntries()
                    self.dR0dW00TPsi[objFuncName] = vecD

                    vecE = vecA.duplicate()
                    vecE.zeroEntries()
                    self.dR00dW0TPsi[objFuncName] = vecE

                    vecF = vecA.duplicate()
                    vecF.zeroEntries()
                    self.dR00dW00TPsi[objFuncName] = vecF

    def zeroTimeAccurateAdjointVectors(self):
        if self.getOption("unsteadyAdjoint")["mode"] == "timeAccurateAdjoint":
            objFuncDict = self.getOption("objFunc")
            for objFuncName in objFuncDict:
                if objFuncName in self.objFuncNames4Adj:
                    self.dRdW0TPsi[objFuncName].zeroEntries()
                    self.dRdW00TPsi[objFuncName].zeroEntries()
                    self.dR0dW0TPsi[objFuncName].zeroEntries()
                    self.dR0dW00TPsi[objFuncName].zeroEntries()
                    self.dR00dW0TPsi[objFuncName].zeroEntries()
                    self.dR00dW00TPsi[objFuncName].zeroEntries()

    def _calcObjFuncNames4Adj(self):
        """
        Compute the objective function names for which we solve the adjoint equation

        Returns
        -------

        objFuncNames4Adj : list
            A list of objective function names we will solve the adjoint for
        """

        objFuncList = []
        objFuncDict = self.getOption("objFunc")
        for objFuncName in objFuncDict:
            objFuncSubDict = objFuncDict[objFuncName]
            for objFuncPart in objFuncSubDict:
                objFuncSubDictPart = objFuncSubDict[objFuncPart]
                if objFuncSubDictPart["addToAdjoint"] is True:
                    if objFuncName not in objFuncList:
                        objFuncList.append(objFuncName)
                elif objFuncSubDictPart["addToAdjoint"] is False:
                    pass
                else:
                    raise Error("addToAdjoint can be either True or False")
        return objFuncList

    def _checkOptions(self):
        """
        Check if the combination of options are valid.
        NOTE: we should add all possible checks here!
        """

        if not self.getOption("useAD")["mode"] in ["fd", "reverse", "forward"]:
            raise Error("useAD->mode only supports fd, reverse, or forward!")

        # check time accurate adjoint
        if self.getOption("unsteadyAdjoint")["mode"] == "timeAccurateAdjoint":
            if not self.getOption("useAD")["mode"] in ["forward", "reverse"]:
                raise Error("timeAccurateAdjoint only supports useAD->mode=forward|reverse")

        if "NONE" not in self.getOption("writeSensMap"):
            if not self.getOption("useAD")["mode"] in ["reverse"]:
                raise Error("writeSensMap is only compatible with useAD->mode=reverse")

        if self.getOption("runLowOrderPrimal4PC")["active"]:
            self.setOption("runLowOrderPrimal4PC", {"active": True, "isPC": False})

        if self.getOption("adjEqnSolMethod") == "fixedPoint":
            # for the fixed-point adjoint, we should not normalize the states and residuals
            if self.comm.rank == 0:
                print("Fixed-point adjoint mode detected. Unset normalizeStates and normalizeResiduals...")

            # force the normalize states to be an empty dict
            if len(self.getOption("normalizeStates")) > 0:
                raise Error("Please do not set any normalizeStates for the fixed-point adjoint!")
            # force the normalize residuals to be None; don't normalize any residuals
            self.setOption("normalizeResiduals", ["None"])

        if self.getOption("discipline") not in ["aero", "thermal"]:
            raise Error("discipline: %s not supported. Options are: aero or thermal" % self.getOption("discipline"))

        nActivated = 0
        for coupling in self.getOption("couplingInfo"):
            if self.getOption("couplingInfo")[coupling]["active"]:
                nActivated += 1
        if nActivated > 1:
            raise Error("Only one coupling scenario can be active, while %i found" % nActivated)

        nAerothermalSurfaces = len(self.getOption("couplingInfo")["aerothermal"]["couplingSurfaceGroups"].keys())
        if nAerothermalSurfaces > 1:
            raise Error(
                "Only one couplingSurfaceGroups is supported for aerothermal, while %i found" % nAerothermalSurfaces
            )

        nAeroStructSurfaces = len(self.getOption("couplingInfo")["aerostructural"]["couplingSurfaceGroups"].keys())
        if nAeroStructSurfaces > 1:
            raise Error(
                "Only one couplingSurfaceGroups is supported for aerostructural, while %i found" % nAeroStructSurfaces
            )

        # check other combinations...

    def saveMultiPointField(self, indexMP):
        """
        Save the state variable vector to self.wVecMPList
        """

        Istart, Iend = self.wVec.getOwnershipRange()
        for i in range(Istart, Iend):
            self.wVecMPList[indexMP][i] = self.wVec[i]

        self.wVecMPList[indexMP].assemblyBegin()
        self.wVecMPList[indexMP].assemblyEnd()

        return

    def setMultiPointField(self, indexMP):
        """
        Set the state variable vector based on self.wVecMPList
        """

        Istart, Iend = self.wVec.getOwnershipRange()
        for i in range(Istart, Iend):
            self.wVec[i] = self.wVecMPList[indexMP][i]

        self.wVec.assemblyBegin()
        self.wVec.assemblyEnd()

        return

    def calcPrimalResidualStatistics(self, mode):
        if self.getOption("useAD")["mode"] in ["forward", "reverse"]:
            self.solverAD.calcPrimalResidualStatistics(mode.encode())
        else:
            self.solver.calcPrimalResidualStatistics(mode.encode())

    def setTimeInstanceField(self, instanceI):
        """
        Set the OpenFOAM state variables based on instance index
        """

        if self.getOption("useAD")["mode"] in ["forward", "reverse"]:
            solver = self.solverAD
        else:
            solver = self.solver

        solver.setTimeInstanceField(instanceI)
        # NOTE: we need to set the OF field to wVec vector here!
        # this is because we will assign self.wVec to the solveAdjoint function later
        solver.ofField2StateVec(self.wVec)

        return

    def initTimeInstanceMats(self):

        nLocalAdjointStates = self.solver.getNLocalAdjointStates()
        nLocalAdjointBoundaryStates = self.solver.getNLocalAdjointBoundaryStates()
        nTimeInstances = -99999
        adjMode = self.getOption("unsteadyAdjoint")["mode"]
        if adjMode == "hybridAdjoint" or adjMode == "timeAccurateAdjoint":
            nTimeInstances = self.getOption("unsteadyAdjoint")["nTimeInstances"]

        self.stateMat = PETSc.Mat().create(PETSc.COMM_WORLD)
        self.stateMat.setSizes(((nLocalAdjointStates, None), (None, nTimeInstances)))
        self.stateMat.setFromOptions()
        self.stateMat.setPreallocationNNZ((nTimeInstances, nTimeInstances))
        self.stateMat.setUp()

        self.stateBCMat = PETSc.Mat().create(PETSc.COMM_WORLD)
        self.stateBCMat.setSizes(((nLocalAdjointBoundaryStates, None), (None, nTimeInstances)))
        self.stateBCMat.setFromOptions()
        self.stateBCMat.setPreallocationNNZ((nTimeInstances, nTimeInstances))
        self.stateBCMat.setUp()

        self.timeVec = PETSc.Vec().createSeq(nTimeInstances, bsize=1, comm=PETSc.COMM_SELF)
        self.timeIdxVec = PETSc.Vec().createSeq(nTimeInstances, bsize=1, comm=PETSc.COMM_SELF)

    def setTimeInstanceVar(self, mode):

        if mode == "list2Mat":
            self.solver.setTimeInstanceVar(mode.encode(), self.stateMat, self.stateBCMat, self.timeVec, self.timeIdxVec)
        elif mode == "mat2List":
            if self.getOption("useAD")["mode"] in ["forward", "reverse"]:
                self.solverAD.setTimeInstanceVar(
                    mode.encode(), self.stateMat, self.stateBCMat, self.timeVec, self.timeIdxVec
                )
            else:
                self.solver.setTimeInstanceVar(
                    mode.encode(), self.stateMat, self.stateBCMat, self.timeVec, self.timeIdxVec
                )
        else:
            raise Error("mode can only be either mat2List or list2Mat!")

    def writeDesignVariable(self, fileName, xDV):
        """
        Write the design variable history to files in the json format
        """
        # Write the design variable history to files
        if self.comm.rank == 0:
            if self.nSolveAdjoints == 1:
                f = open(fileName, "w")
            else:
                f = open(fileName, "a")
            # write design variables
            f.write('\n"Optimization_Iteration_%03d":\n' % self.nSolveAdjoints)
            f.write("{\n")
            nDVNames = len(xDV)
            dvNameCounter = 0
            for dvName in sorted(xDV):
                f.write('    "%s": ' % dvName)
                try:
                    nDVs = len(xDV[dvName])
                    f.write("[ ")
                    for i in range(nDVs):
                        if i < nDVs - 1:
                            f.write("%20.15e, " % xDV[dvName][i])
                        else:
                            f.write("%20.15e " % xDV[dvName][i])
                    f.write("]")
                except Exception:
                    f.write(" %20.15e" % xDV[dvName])
                # check whether to add a comma
                dvNameCounter = dvNameCounter + 1
                if dvNameCounter < nDVNames:
                    f.write(",\n")
                else:
                    f.write("\n")
            f.write("},\n")
            f.close()

    def writeDeformedFFDs(self, counter=None):
        """
        Write the deformed FFDs to the disk during optimization
        """

        if self.comm.rank == 0:
            print("writeDeformedFFDs is deprecated since v3.0.1!")

        """
        if self.getOption("writeDeformedFFDs"):
            if counter is None:
                self.DVGeo.writeTecplot("deformedFFD.dat", self.nSolveAdjoints)
            else:
                self.DVGeo.writeTecplot("deformedFFD.dat", counter)
        """

    def writeTotalDeriv(self, fileName, sens, evalFuncs):
        """
        Write the total derivatives history to files in the json format
        This will only write total derivative for evalFuncs
        """
        # Write the sens history to files
        if self.comm.rank == 0:
            if self.nSolveAdjoints == 2:
                f = open(fileName, "w")
            else:
                f = open(fileName, "a")
            # write design variables
            f.write('\n"Optimization_Iteration_%03d":\n' % (self.nSolveAdjoints - 1))
            f.write("{\n")
            nFuncNames = len(evalFuncs)
            funcNameCounter = 0
            for funcName in sorted(evalFuncs):
                f.write('    "%s": \n    {\n' % funcName)
                nDVNames = len(sens[funcName])
                dvNameCounter = 0
                for dvName in sorted(sens[funcName]):
                    f.write('        "%s": ' % dvName)
                    try:
                        nDVs = len(sens[funcName][dvName])
                        f.write("[ ")
                        for i in range(nDVs):
                            if i < nDVs - 1:
                                f.write("%20.15e, " % sens[funcName][dvName][i])
                            else:
                                f.write("%20.15e " % sens[funcName][dvName][i])
                        f.write("]")
                    except Exception:
                        f.write(" %20.15e" % sens[funcName][dvName])
                    # check whether to add a comma
                    dvNameCounter = dvNameCounter + 1
                    if dvNameCounter < nDVNames:
                        f.write(",\n")
                    else:
                        f.write("\n")
                f.write("    }")
                # check whether to add a comma
                funcNameCounter = funcNameCounter + 1
                if funcNameCounter < nFuncNames:
                    f.write(",\n")
                else:
                    f.write("\n")
            f.write("},\n")
            f.close()

    def getTimeInstanceObjFunc(self, instanceI, objFuncName):
        """
        Return the value of objective function at the given time instance and name
        """

        return self.solver.getTimeInstanceObjFunc(instanceI, objFuncName.encode())

    def getForwardADDerivVal(self, objFuncName):
        """
        Return the derivative value computed by forward mode AD primal solution
        """
        return self.solverAD.getForwardADDerivVal(objFuncName.encode())

    def evalFunctions(self, funcs, evalFuncs=None, ignoreMissing=False):
        """
        Evaluate the desired functions given in iterable object,
        'evalFuncs' and add them to the dictionary 'funcs'. The keys
        in the funcs dictionary will be have an _<ap.name> appended to
        them. Additionally, information regarding whether or not the
        last analysis with the solvePrimal was successful is
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
            Flag to suppress checking for a valid function. Please use
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

        for funcName in evalFuncs:
            if self.primalFail:
                if len(self.objFuncValuePrevIter) == 0:
                    raise Error("Primal solution failed for the baseline design!")
                else:
                    # do not call self.solver.getObjFuncValue because they can be nonphysical,
                    # assign funcs based on self.objFuncValuePrevIter instead
                    funcs[funcName] = self.objFuncValuePrevIter[funcName]
            else:
                # call self.solver.getObjFuncValue to get the objFuncValue from
                # the DASolver
                objFuncValue = self.solver.getObjFuncValue(funcName.encode())
                funcs[funcName] = objFuncValue
                # assign the objFuncValuePrevIter
                self.objFuncValuePrevIter[funcName] = funcs[funcName]

        if self.primalFail:
            funcs["fail"] = True
        else:
            funcs["fail"] = False

        return

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

        if self.DVGeo is None:
            raise Error("DVGeo not set!")

        dvs = self.DVGeo.getValues()

        for funcName in evalFuncs:
            funcsSens[funcName] = {}
            for dvName in dvs:
                nDVs = len(dvs[dvName])
                funcsSens[funcName][dvName] = np.zeros(nDVs, self.dtype)
                for i in range(nDVs):
                    funcsSens[funcName][dvName][i] = self.adjTotalDeriv[funcName][dvName][i]

        if self.adjointFail:
            funcsSens["fail"] = True
        else:
            funcsSens["fail"] = False

        return

    def setDVGeo(self, DVGeo):
        """
        Set the DVGeometry object that will manipulate 'geometry' in
        this object. Note that <SOLVER> does not **strictly** need a
        DVGeometry object, but if optimization with geometric
        changes is desired, then it is required.
        Parameters
        ----------
        dvGeo : A DVGeometry object.
            Object responsible for manipulating the constraints that
            this object is responsible for.
        Examples
        --------
        >>> CFDsolver = <SOLVER>(comm=comm, options=CFDoptions)
        >>> CFDsolver.setDVGeo(DVGeo)
        """

        self.DVGeo = DVGeo

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

    def setEvalFuncs(self, evalFuncs):
        objFuncs = self.getOption("objFunc")
        for funcName in objFuncs:
            for funcPart in objFuncs[funcName]:
                if objFuncs[funcName][funcPart]["addToAdjoint"] is True:
                    if funcName not in evalFuncs:
                        evalFuncs.append(funcName)
        return

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

    def setDesignVars(self, x):
        """
        Set the internal design variables.
        At the moment we don't have any internal DVs to set.
        """
        pass

        return

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

    def _writeOFCaseFiles(self):

        return

    def writeFieldSensitivityMap(self, objFuncName, designVarName, solutionTime, fieldType, sensVec):
        """
        Save the field sensitivity dObjFunc/dDesignVar map to disk.

        Parameters
        ----------

        objFuncName : str
            Name of the objective function
        designVarName : str
            Name of the design variable
        solutionTime : float
            The solution time where the sensitivity will be save
        fieldType : str
            The type of the field, either scalar or vector
        sensVec : petsc vec
            The Petsc vector that contains the sensitivity
        """

        workingDir = os.getcwd()
        if self.parallel:
            sensDir = "processor%d/%.8f/" % (self.rank, solutionTime)
        else:
            sensDir = "%.8f/" % solutionTime

        sensDir = os.path.join(workingDir, sensDir)

        sensList = []
        Istart, Iend = sensVec.getOwnershipRange()
        for idxI in range(Istart, Iend):
            sensList.append(sensVec[idxI])

        # write sens
        if not os.path.isfile(os.path.join(sensDir, "sens_%s_%s" % (objFuncName, designVarName))):
            fSens = open(os.path.join(sensDir, "sens_%s_%s" % (objFuncName, designVarName)), "w")
            if fieldType == "scalar":
                self._writeOpenFoamHeader(fSens, "volScalarField", sensDir, "sens_%s_%s" % (objFuncName, designVarName))
                fSens.write("dimensions      [0 0 0 0 0 0 0];\n")
                fSens.write("internalField   nonuniform List<scalar>\n")
                fSens.write("%d\n" % len(sensList))
                fSens.write("(\n")
                for i in range(len(sensList)):
                    fSens.write("%g\n" % sensList[i])
                fSens.write(")\n")
                fSens.write(";\n")
            elif fieldType == "vector":
                self._writeOpenFoamHeader(fSens, "volVectorField", sensDir, "sens_%s_%s" % (objFuncName, designVarName))
                fSens.write("dimensions      [0 0 0 0 0 0 0];\n")
                fSens.write("internalField   nonuniform List<vector>\n")
                fSens.write("%d\n" % len(sensList) / 3)
                fSens.write("(\n")
                counterI = 0
                for i in range(len(sensList) / 3):
                    fSens.write("(")
                    for j in range(3):
                        fSens.write("%g " % sensList[counterI])
                        counterI = counterI + 1
                    fSens.write(")\n")
                fSens.write(")\n")
                fSens.write(";\n")
            else:
                raise Error("fieldType %s not valid! Options are: scalar or vector" % fieldType)

            fSens.write("boundaryField\n")
            fSens.write("{\n")
            fSens.write('    "(.*)"\n')
            fSens.write("    {\n")
            fSens.write("        type  zeroGradient;\n")
            fSens.write("    }\n")
            fSens.write("}\n")
            fSens.close()

    def writeSurfaceSensitivityMap(self, objFuncName, designVarName, solutionTime):
        """
        Save the sensitivity dObjFunc/dXs map to disk. where Xs is the wall surface mesh coordinate

        Parameters
        ----------

        objFuncName : str
            Name of the objective function
        designVarName : str
            Name of the design variable
        solutionTime : float
            The solution time where the sensitivity will be save
        """

        dFdXs = self.mesh.getdXs()
        dFdXs = self.mapVector(dFdXs, self.allWallsGroup, self.allWallsGroup)

        pts = self.getSurfaceCoordinates(self.allWallsGroup)
        conn, faceSizes = self.getSurfaceConnectivity(self.allWallsGroup)
        conn = np.array(conn).flatten()

        workingDir = os.getcwd()
        if self.parallel:
            meshDir = "processor%d/%.11f/polyMesh/" % (self.rank, solutionTime)
            sensDir = "processor%d/%.11f/" % (self.rank, solutionTime)
        else:
            meshDir = "%.11f/polyMesh/" % solutionTime
            sensDir = "%.11f/" % solutionTime

        meshDir = os.path.join(workingDir, meshDir)
        sensDir = os.path.join(workingDir, sensDir)

        if not os.path.isdir(sensDir):
            try:
                os.mkdir(sensDir)
            except Exception:
                raise Error("Can not make a directory at %s" % sensDir)
        if not os.path.isdir(meshDir):
            try:
                os.mkdir(meshDir)
            except Exception:
                raise Error("Can not make a directory at %s" % meshDir)

        # write points
        if not os.path.isfile(os.path.join(meshDir, "points")):
            fPoints = open(os.path.join(meshDir, "points"), "w")
            self._writeOpenFoamHeader(fPoints, "dictionary", meshDir, "points")
            fPoints.write("%d\n" % len(pts))
            fPoints.write("(\n")
            for i in range(len(pts)):
                fPoints.write("(%g %g %g)\n" % (float(pts[i][0]), float(pts[i][1]), float(pts[i][2])))
            fPoints.write(")\n")
            fPoints.close()

        # write faces
        if not os.path.isfile(os.path.join(meshDir, "faces")):
            fFaces = open(os.path.join(meshDir, "faces"), "w")
            self._writeOpenFoamHeader(fFaces, "dictionary", meshDir, "faces")
            counterI = 0
            fFaces.write("%d\n" % len(faceSizes))
            fFaces.write("(\n")
            for i in range(len(faceSizes)):
                fFaces.write("%d(" % faceSizes[i])
                for j in range(faceSizes[i]):
                    fFaces.write(" %d " % conn[counterI])
                    counterI += 1
                fFaces.write(")\n")
            fFaces.write(")\n")
            fFaces.close()

        # write owner
        if not os.path.isfile(os.path.join(meshDir, "owner")):
            fOwner = open(os.path.join(meshDir, "owner"), "w")
            self._writeOpenFoamHeader(fOwner, "dictionary", meshDir, "owner")
            fOwner.write("%d\n" % len(faceSizes))
            fOwner.write("(\n")
            for i in range(len(faceSizes)):
                fOwner.write("0\n")
            fOwner.write(")\n")
            fOwner.close()

        # write neighbour
        if not os.path.isfile(os.path.join(meshDir, "neighbour")):
            fNeighbour = open(os.path.join(meshDir, "neighbour"), "w")
            self._writeOpenFoamHeader(fNeighbour, "dictionary", meshDir, "neighbour")
            fNeighbour.write("%d\n" % len(faceSizes))
            fNeighbour.write("(\n")
            for i in range(len(faceSizes)):
                fNeighbour.write("0\n")
            fNeighbour.write(")\n")
            fNeighbour.close()

        # write boundary
        if not os.path.isfile(os.path.join(meshDir, "boundary")):
            fBoundary = open(os.path.join(meshDir, "boundary"), "w")
            self._writeOpenFoamHeader(fBoundary, "dictionary", meshDir, "boundary")
            fBoundary.write("1\n")
            fBoundary.write("(\n")
            fBoundary.write("    allWalls\n")
            fBoundary.write("    {\n")
            fBoundary.write("        type       wall;\n")
            fBoundary.write("        nFaces     %d;\n" % len(faceSizes))
            fBoundary.write("        startFace  0;\n")
            fBoundary.write("    }\n")
            fBoundary.write(")\n")
            fBoundary.close()

        # write sens
        if not os.path.isfile(os.path.join(sensDir, "sens_%s_%s" % (objFuncName, designVarName))):
            fSens = open(os.path.join(sensDir, "sens_%s_%s" % (objFuncName, designVarName)), "w")
            self._writeOpenFoamHeader(fSens, "volVectorField", sensDir, "sens_%s_%s" % (objFuncName, designVarName))
            fSens.write("dimensions      [0 0 0 0 0 0 0];\n")
            fSens.write("internalField   uniform (0 0 0);\n")

            counterI = 0
            fSens.write("boundaryField\n")
            fSens.write("{\n")
            fSens.write("    allWalls\n")
            fSens.write("    {\n")
            fSens.write("        type  wall;\n")
            fSens.write("        value nonuniform List<vector>\n")
            fSens.write("%d\n" % len(faceSizes))
            fSens.write("(\n")
            counterI = 0
            for i in range(len(faceSizes)):
                sensXMean = 0.0
                sensYMean = 0.0
                sensZMean = 0.0
                for j in range(faceSizes[i]):
                    idxI = conn[counterI]
                    sensXMean += dFdXs[idxI][0]
                    sensYMean += dFdXs[idxI][1]
                    sensZMean += dFdXs[idxI][2]
                    counterI += 1
                sensXMean /= faceSizes[i]
                sensYMean /= faceSizes[i]
                sensZMean /= faceSizes[i]
                fSens.write("(%f %f %f)\n" % (sensXMean, sensYMean, sensZMean))
            fSens.write(")\n")
            fSens.write(";\n")
            fSens.write("    }\n")
            fSens.write("}\n")
            fSens.close()

    def writePetscVecMat(self, name, vecMat, mode="Binary"):
        """
        Write Petsc vectors or matrices
        """

        Info("Saving %s to disk...." % name)
        if mode == "ASCII":
            viewer = PETSc.Viewer().createASCII(name + ".dat", mode="w", comm=PETSc.COMM_WORLD)
            viewer.pushFormat(1)
            viewer(vecMat)
        elif mode == "Binary":
            viewer = PETSc.Viewer().createBinary(name + ".bin", mode="w", comm=PETSc.COMM_WORLD)
            viewer(vecMat)
        else:
            raise Error("mode not valid! Options are: ASCII or Binary")

    def readPetscVecMat(self, name, vecMat):
        """
        Read Petsc vectors or matrices
        """

        Info("Reading %s from disk...." % name)
        viewer = PETSc.Viewer().createBinary(name + ".bin", comm=PETSc.COMM_WORLD)
        vecMat.load(viewer)

    def solvePrimal(self):
        """
        Run primal solver to compute state variables and objectives

        Input:
        ------
        xvVec: vector that contains all the mesh point coordinates

        Output:
        -------
        wVec: vector that contains all the state variables

        self.primalFail: if the primal solution fails, assigns 1, otherwise 0
        """

        Info("Running Primal Solver %03d" % self.nSolvePrimals)

        self.deletePrevPrimalSolTime()

        self.primalFail = 0
        if self.getOption("useAD")["mode"] == "forward":
            self.primalFail = self.solverAD.solvePrimal(self.xvVec, self.wVec)
        else:
            self.primalFail = self.solver.solvePrimal(self.xvVec, self.wVec)

        if self.getOption("writeMinorIterations"):
            self.renameSolution(self.nSolvePrimals)
            self.writeDeformedFFDs(self.nSolvePrimals)

        self.nSolvePrimals += 1

        return

    def solveAdjoint(self):
        """
        Run adjoint solver to compute the adjoint vector psiVec

        Input:
        ------
        xvVec: vector that contains all the mesh point coordinates

        wVec: vector that contains all the state variables

        Output:
        -------
        self.adjTotalDeriv: the dict contains all the total derivative vectors

        self.adjointFail: if the adjoint solution fails, assigns 1, otherwise 0
        """

        # save the point vector and state vector to disk
        """
        Info("Saving the xvVec and wVec vectors to disk....")
        self.comm.Barrier()
        viewerXv = PETSc.Viewer().createBinary("xvVec_%03d.bin" % self.nSolveAdjoints, mode="w", comm=PETSc.COMM_WORLD)
        viewerXv(self.xvVec)
        viewerW = PETSc.Viewer().createBinary("wVec_%03d.bin" % self.nSolveAdjoints, mode="w", comm=PETSc.COMM_WORLD)
        viewerW(self.wVec)
        """

        if self.getOption("useAD")["mode"] == "forward":
            raise Error("solveAdjoint only supports useAD->mode=reverse|fd")

        if not self.getOption("writeMinorIterations"):
            solutionTime, renamed = self.renameSolution(self.nSolveAdjoints)

        Info("Running adjoint Solver %03d" % self.nSolveAdjoints)

        self.setOption("runStatus", "solveAdjoint")
        self.updateDAOption()

        if self.getOption("multiPoint"):
            self.solver.updateOFField(self.wVec)
            if self.getOption("useAD")["mode"] == "reverse":
                self.solverAD.updateOFField(self.wVec)

        self.adjointFail = 0

        # calculate dRdWT
        if self.getOption("useAD")["mode"] == "fd":
            dRdWT = PETSc.Mat().create(PETSc.COMM_WORLD)
            self.solver.calcdRdWT(self.xvVec, self.wVec, 0, dRdWT)
        elif self.getOption("useAD")["mode"] == "reverse":
            self.solverAD.initializedRdWTMatrixFree(self.xvVec, self.wVec)

        # calculate dRdWTPC. If runLowOrderPrimal4PC is true, we compute the PC mat
        # before solving the primal, so we will skip it here
        if not self.getOption("runLowOrderPrimal4PC")["active"]:
            adjPCLag = self.getOption("adjPCLag")
            if self.nSolveAdjoints == 1 or (self.nSolveAdjoints - 1) % adjPCLag == 0:
                self.dRdWTPC = PETSc.Mat().create(PETSc.COMM_WORLD)
                self.solver.calcdRdWT(self.xvVec, self.wVec, 1, self.dRdWTPC)

        # Initialize the KSP object
        ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
        if self.getOption("useAD")["mode"] == "fd":
            self.solver.createMLRKSP(dRdWT, self.dRdWTPC, ksp)
        elif self.getOption("useAD")["mode"] == "reverse":
            self.solverAD.createMLRKSPMatrixFree(self.dRdWTPC, ksp)

        # loop over all objFunc, calculate dFdW, and solve the adjoint
        objFuncDict = self.getOption("objFunc")
        wSize = self.solver.getNLocalAdjointStates()
        for objFuncName in objFuncDict:
            if objFuncName in self.objFuncNames4Adj:
                dFdW = PETSc.Vec().create(PETSc.COMM_WORLD)
                dFdW.setSizes((wSize, PETSc.DECIDE), bsize=1)
                dFdW.setFromOptions()
                if self.getOption("useAD")["mode"] == "fd":
                    self.solver.calcdFdW(self.xvVec, self.wVec, objFuncName.encode(), dFdW)
                elif self.getOption("useAD")["mode"] == "reverse":
                    self.solverAD.calcdFdWAD(self.xvVec, self.wVec, objFuncName.encode(), dFdW)

                # if it is time accurate adjoint, add extra terms for dFdW
                if self.getOption("unsteadyAdjoint")["mode"] == "timeAccurateAdjoint":
                    # first copy the vectors from previous residual time step level
                    self.dR0dW0TPsi[objFuncName].copy(self.dR00dW0TPsi[objFuncName])
                    self.dR0dW00TPsi[objFuncName].copy(self.dR00dW00TPsi[objFuncName])
                    self.dRdW0TPsi[objFuncName].copy(self.dR0dW0TPsi[objFuncName])
                    self.dRdW00TPsi[objFuncName].copy(self.dR0dW00TPsi[objFuncName])
                    dFdW.axpy(-1.0, self.dR0dW0TPsi[objFuncName])
                    dFdW.axpy(-1.0, self.dR00dW00TPsi[objFuncName])

                # Initialize the adjoint vector psi and solve for it
                if self.getOption("useAD")["mode"] == "fd":
                    self.adjointFail = self.solver.solveLinearEqn(ksp, dFdW, self.adjVectors[objFuncName])
                elif self.getOption("useAD")["mode"] == "reverse":
                    self.adjointFail = self.solverAD.solveLinearEqn(ksp, dFdW, self.adjVectors[objFuncName])

                if self.getOption("unsteadyAdjoint")["mode"] == "timeAccurateAdjoint":
                    self.solverAD.calcdRdWOldTPsiAD(1, self.adjVectors[objFuncName], self.dRdW0TPsi[objFuncName])
                    self.solverAD.calcdRdWOldTPsiAD(2, self.adjVectors[objFuncName], self.dRdW00TPsi[objFuncName])

                dFdW.destroy()

        ksp.destroy()
        if self.getOption("useAD")["mode"] == "fd":
            dRdWT.destroy()
        elif self.getOption("useAD")["mode"] == "reverse":
            self.solverAD.destroydRdWTMatrixFree()
        # we destroy dRdWTPC only when we need to recompute it next time
        # see the bottom of this function

        # ************ Now compute the total derivatives **********************
        Info("Computing total derivatives....")

        designVarDict = self.getOption("designVar")
        for designVarName in designVarDict:
            Info("Computing total derivatives for %s" % designVarName)
            ###################### BC: boundary condition as design variable ###################
            if designVarDict[designVarName]["designVarType"] == "BC":
                if self.getOption("useAD")["mode"] == "fd":
                    nDVs = 1
                    # calculate dRdBC
                    dRdBC = PETSc.Mat().create(PETSc.COMM_WORLD)
                    self.solver.calcdRdBC(self.xvVec, self.wVec, designVarName.encode(), dRdBC)
                    # loop over all objectives
                    for objFuncName in objFuncDict:
                        if objFuncName in self.objFuncNames4Adj:
                            # calculate dFdBC
                            dFdBC = PETSc.Vec().create(PETSc.COMM_WORLD)
                            dFdBC.setSizes((PETSc.DECIDE, nDVs), bsize=1)
                            dFdBC.setFromOptions()
                            self.solver.calcdFdBC(
                                self.xvVec, self.wVec, objFuncName.encode(), designVarName.encode(), dFdBC
                            )
                            # call the total deriv
                            totalDeriv = PETSc.Vec().create(PETSc.COMM_WORLD)
                            totalDeriv.setSizes((PETSc.DECIDE, nDVs), bsize=1)
                            totalDeriv.setFromOptions()
                            self.calcTotalDeriv(dRdBC, dFdBC, self.adjVectors[objFuncName], totalDeriv)
                            # assign the total derivative to self.adjTotalDeriv
                            self.adjTotalDeriv[objFuncName][designVarName] = np.zeros(nDVs, self.dtype)
                            # we need to convert the parallel vec to seq vec
                            totalDerivSeq = PETSc.Vec().createSeq(nDVs, bsize=1, comm=PETSc.COMM_SELF)
                            self.solver.convertMPIVec2SeqVec(totalDeriv, totalDerivSeq)
                            for i in range(nDVs):
                                self.adjTotalDeriv[objFuncName][designVarName][i] = totalDerivSeq[i]

                            totalDeriv.destroy()
                            totalDerivSeq.destroy()
                            dFdBC.destroy()
                    dRdBC.destroy()
                elif self.getOption("useAD")["mode"] == "reverse":
                    nDVs = 1
                    # loop over all objectives
                    for objFuncName in objFuncDict:
                        if objFuncName in self.objFuncNames4Adj:
                            # calculate dFdBC
                            dFdBC = PETSc.Vec().create(PETSc.COMM_WORLD)
                            dFdBC.setSizes((PETSc.DECIDE, nDVs), bsize=1)
                            dFdBC.setFromOptions()
                            self.solverAD.calcdFdBCAD(
                                self.xvVec, self.wVec, objFuncName.encode(), designVarName.encode(), dFdBC
                            )
                            # Calculate dRBCT^Psi
                            totalDeriv = PETSc.Vec().create(PETSc.COMM_WORLD)
                            totalDeriv.setSizes((PETSc.DECIDE, nDVs), bsize=1)
                            totalDeriv.setFromOptions()
                            self.solverAD.calcdRdBCTPsiAD(
                                self.xvVec, self.wVec, self.adjVectors[objFuncName], designVarName.encode(), totalDeriv
                            )
                            # totalDeriv = dFdBC - dRdBCT*psi
                            totalDeriv.scale(-1.0)
                            totalDeriv.axpy(1.0, dFdBC)
                            # assign the total derivative to self.adjTotalDeriv
                            self.adjTotalDeriv[objFuncName][designVarName] = np.zeros(nDVs, self.dtype)
                            # we need to convert the parallel vec to seq vec
                            totalDerivSeq = PETSc.Vec().createSeq(nDVs, bsize=1, comm=PETSc.COMM_SELF)
                            self.solver.convertMPIVec2SeqVec(totalDeriv, totalDerivSeq)
                            for i in range(nDVs):
                                self.adjTotalDeriv[objFuncName][designVarName][i] = totalDerivSeq[i]

                            totalDeriv.destroy()
                            totalDerivSeq.destroy()
                            dFdBC.destroy()
            ###################### AOA: angle of attack as design variable ###################
            elif designVarDict[designVarName]["designVarType"] == "AOA":
                if self.getOption("useAD")["mode"] == "fd":
                    nDVs = 1
                    # calculate dRdAOA
                    dRdAOA = PETSc.Mat().create(PETSc.COMM_WORLD)
                    self.solver.calcdRdAOA(self.xvVec, self.wVec, designVarName.encode(), dRdAOA)
                    # loop over all objectives
                    for objFuncName in objFuncDict:
                        if objFuncName in self.objFuncNames4Adj:
                            # calculate dFdAOA
                            dFdAOA = PETSc.Vec().create(PETSc.COMM_WORLD)
                            dFdAOA.setSizes((PETSc.DECIDE, nDVs), bsize=1)
                            dFdAOA.setFromOptions()
                            self.solver.calcdFdAOA(
                                self.xvVec, self.wVec, objFuncName.encode(), designVarName.encode(), dFdAOA
                            )
                            # call the total deriv
                            totalDeriv = PETSc.Vec().create(PETSc.COMM_WORLD)
                            totalDeriv.setSizes((PETSc.DECIDE, nDVs), bsize=1)
                            totalDeriv.setFromOptions()
                            self.calcTotalDeriv(dRdAOA, dFdAOA, self.adjVectors[objFuncName], totalDeriv)
                            # assign the total derivative to self.adjTotalDeriv
                            self.adjTotalDeriv[objFuncName][designVarName] = np.zeros(nDVs, self.dtype)
                            # we need to convert the parallel vec to seq vec
                            totalDerivSeq = PETSc.Vec().createSeq(nDVs, bsize=1, comm=PETSc.COMM_SELF)
                            self.solver.convertMPIVec2SeqVec(totalDeriv, totalDerivSeq)
                            for i in range(nDVs):
                                self.adjTotalDeriv[objFuncName][designVarName][i] = totalDerivSeq[i]

                            totalDeriv.destroy()
                            totalDerivSeq.destroy()
                            dFdAOA.destroy()
                    dRdAOA.destroy()
                elif self.getOption("useAD")["mode"] == "reverse":
                    nDVs = 1
                    # loop over all objectives
                    for objFuncName in objFuncDict:
                        if objFuncName in self.objFuncNames4Adj:
                            # calculate dFdAOA
                            dFdAOA = PETSc.Vec().create(PETSc.COMM_WORLD)
                            dFdAOA.setSizes((PETSc.DECIDE, nDVs), bsize=1)
                            dFdAOA.setFromOptions()
                            self.calcdFdAOAAnalytical(objFuncName, dFdAOA)
                            # Calculate dRAOAT^Psi
                            totalDeriv = PETSc.Vec().create(PETSc.COMM_WORLD)
                            totalDeriv.setSizes((PETSc.DECIDE, nDVs), bsize=1)
                            totalDeriv.setFromOptions()
                            self.solverAD.calcdRdAOATPsiAD(
                                self.xvVec, self.wVec, self.adjVectors[objFuncName], designVarName.encode(), totalDeriv
                            )
                            # totalDeriv = dFdAOA - dRdAOAT*psi
                            totalDeriv.scale(-1.0)
                            totalDeriv.axpy(1.0, dFdAOA)
                            # assign the total derivative to self.adjTotalDeriv
                            self.adjTotalDeriv[objFuncName][designVarName] = np.zeros(nDVs, self.dtype)
                            # we need to convert the parallel vec to seq vec
                            totalDerivSeq = PETSc.Vec().createSeq(nDVs, bsize=1, comm=PETSc.COMM_SELF)
                            self.solver.convertMPIVec2SeqVec(totalDeriv, totalDerivSeq)
                            for i in range(nDVs):
                                self.adjTotalDeriv[objFuncName][designVarName][i] = totalDerivSeq[i]

                            totalDeriv.destroy()
                            totalDerivSeq.destroy()
                            dFdAOA.destroy()
            ################### FFD: FFD points as design variable ###################
            elif designVarDict[designVarName]["designVarType"] == "FFD":
                if self.getOption("useAD")["mode"] == "fd":
                    nDVs = self.setdXvdFFDMat(designVarName)
                    # calculate dRdFFD
                    dRdFFD = PETSc.Mat().create(PETSc.COMM_WORLD)
                    self.solver.calcdRdFFD(self.xvVec, self.wVec, designVarName.encode(), dRdFFD)
                    # loop over all objectives
                    for objFuncName in objFuncDict:
                        if objFuncName in self.objFuncNames4Adj:
                            # calculate dFdFFD
                            dFdFFD = PETSc.Vec().create(PETSc.COMM_WORLD)
                            dFdFFD.setSizes((PETSc.DECIDE, nDVs), bsize=1)
                            dFdFFD.setFromOptions()
                            self.solver.calcdFdFFD(
                                self.xvVec, self.wVec, objFuncName.encode(), designVarName.encode(), dFdFFD
                            )
                            # call the total deriv
                            totalDeriv = PETSc.Vec().create(PETSc.COMM_WORLD)
                            totalDeriv.setSizes((PETSc.DECIDE, nDVs), bsize=1)
                            totalDeriv.setFromOptions()
                            self.calcTotalDeriv(dRdFFD, dFdFFD, self.adjVectors[objFuncName], totalDeriv)
                            # assign the total derivative to self.adjTotalDeriv
                            self.adjTotalDeriv[objFuncName][designVarName] = np.zeros(nDVs, self.dtype)
                            # we need to convert the parallel vec to seq vec
                            totalDerivSeq = PETSc.Vec().createSeq(nDVs, bsize=1, comm=PETSc.COMM_SELF)
                            self.solver.convertMPIVec2SeqVec(totalDeriv, totalDerivSeq)
                            for i in range(nDVs):
                                self.adjTotalDeriv[objFuncName][designVarName][i] = totalDerivSeq[i]
                            totalDeriv.destroy()
                            totalDerivSeq.destroy()
                            dFdFFD.destroy()
                    dRdFFD.destroy()
                elif self.getOption("useAD")["mode"] == "reverse":
                    try:
                        nDVs = len(self.DVGeo.getValues()[designVarName])
                    except Exception:
                        nDVs = 1
                    xvSize = len(self.xv) * 3
                    for objFuncName in objFuncDict:
                        if objFuncName in self.objFuncNames4Adj:
                            # Calculate dFdXv
                            dFdXv = PETSc.Vec().create(PETSc.COMM_WORLD)
                            dFdXv.setSizes((xvSize, PETSc.DECIDE), bsize=1)
                            dFdXv.setFromOptions()
                            self.solverAD.calcdFdXvAD(
                                self.xvVec, self.wVec, objFuncName.encode(), designVarName.encode(), dFdXv
                            )

                            # Calculate dRXvT^Psi
                            totalDerivXv = PETSc.Vec().create(PETSc.COMM_WORLD)
                            totalDerivXv.setSizes((xvSize, PETSc.DECIDE), bsize=1)
                            totalDerivXv.setFromOptions()
                            self.solverAD.calcdRdXvTPsiAD(
                                self.xvVec, self.wVec, self.adjVectors[objFuncName], totalDerivXv
                            )

                            # totalDeriv = dFdXv - dRdXvT*psi
                            totalDerivXv.scale(-1.0)
                            totalDerivXv.axpy(1.0, dFdXv)

                            # write the matrix
                            if "dFdXvTotalDeriv" in self.getOption("writeJacobians") or "all" in self.getOption(
                                "writeJacobians"
                            ):
                                self.writePetscVecMat("dFdXvTotalDeriv_%s" % objFuncName, totalDerivXv)
                                self.writePetscVecMat("dFdXvTotalDeriv_%s" % objFuncName, totalDerivXv, "ASCII")

                            if self.DVGeo is not None and self.DVGeo.getNDV() > 0:
                                dFdFFD = self.mapdXvTodFFD(totalDerivXv)
                                if designVarName in self.getOption("writeSensMap"):
                                    # we can't save the surface sensitivity time with the primal solution
                                    # because surfaceSensMap needs to have its own mesh (design surface only)
                                    sensSolTime = float(solutionTime) / 1000.0
                                    self.writeSurfaceSensitivityMap(objFuncName, designVarName, sensSolTime)

                                # assign the total derivative to self.adjTotalDeriv
                                self.adjTotalDeriv[objFuncName][designVarName] = np.zeros(nDVs, self.dtype)
                                for i in range(nDVs):
                                    self.adjTotalDeriv[objFuncName][designVarName][i] = dFdFFD[designVarName][0][i]

                            totalDerivXv.destroy()
                            dFdXv.destroy()
            ################### ACT: actuator models as design variable ###################
            elif designVarDict[designVarName]["designVarType"] in ["ACTL", "ACTP", "ACTD"]:
                if self.getOption("useAD")["mode"] == "fd":
                    designVarType = designVarDict[designVarName]["designVarType"]
                    nDVTable = {"ACTP": 9, "ACTD": 10, "ACTL": 11}
                    nDVs = nDVTable[designVarType]
                    # calculate dRdACT
                    dRdACT = PETSc.Mat().create(PETSc.COMM_WORLD)
                    self.solver.calcdRdACT(
                        self.xvVec, self.wVec, designVarName.encode(), designVarType.encode(), dRdACT
                    )
                    # loop over all objectives
                    for objFuncName in objFuncDict:
                        if objFuncName in self.objFuncNames4Adj:
                            # calculate dFdACT
                            dFdACT = PETSc.Vec().create(PETSc.COMM_WORLD)
                            dFdACT.setSizes((PETSc.DECIDE, nDVs), bsize=1)
                            dFdACT.setFromOptions()
                            self.solver.calcdFdACT(
                                self.xvVec,
                                self.wVec,
                                objFuncName.encode(),
                                designVarName.encode(),
                                designVarType.encode(),
                                dFdACT,
                            )
                            # call the total deriv
                            totalDeriv = PETSc.Vec().create(PETSc.COMM_WORLD)
                            totalDeriv.setSizes((PETSc.DECIDE, nDVs), bsize=1)
                            totalDeriv.setFromOptions()
                            self.calcTotalDeriv(dRdACT, dFdACT, self.adjVectors[objFuncName], totalDeriv)
                            # assign the total derivative to self.adjTotalDeriv
                            self.adjTotalDeriv[objFuncName][designVarName] = np.zeros(nDVs, self.dtype)
                            # we need to convert the parallel vec to seq vec
                            totalDerivSeq = PETSc.Vec().createSeq(nDVs, bsize=1, comm=PETSc.COMM_SELF)
                            self.solver.convertMPIVec2SeqVec(totalDeriv, totalDerivSeq)
                            for i in range(nDVs):
                                self.adjTotalDeriv[objFuncName][designVarName][i] = totalDerivSeq[i]
                            totalDeriv.destroy()
                            totalDerivSeq.destroy()
                            dFdACT.destroy()
                    dRdACT.destroy()
                elif self.getOption("useAD")["mode"] == "reverse":
                    designVarType = designVarDict[designVarName]["designVarType"]
                    nDVTable = {"ACTP": 9, "ACTD": 10, "ACTL": 11}
                    nDVs = nDVTable[designVarType]
                    # loop over all objectives
                    for objFuncName in objFuncDict:
                        if objFuncName in self.objFuncNames4Adj:
                            # calculate dFdACT
                            dFdACT = PETSc.Vec().create(PETSc.COMM_WORLD)
                            dFdACT.setSizes((PETSc.DECIDE, nDVs), bsize=1)
                            dFdACT.setFromOptions()
                            self.solverAD.calcdFdACTAD(
                                self.xvVec, self.wVec, objFuncName.encode(), designVarName.encode(), dFdACT
                            )
                            # call the total deriv
                            totalDeriv = PETSc.Vec().create(PETSc.COMM_WORLD)
                            totalDeriv.setSizes((PETSc.DECIDE, nDVs), bsize=1)
                            totalDeriv.setFromOptions()
                            # calculate dRdActT*Psi and save it to totalDeriv
                            self.solverAD.calcdRdActTPsiAD(
                                self.xvVec, self.wVec, self.adjVectors[objFuncName], designVarName.encode(), totalDeriv
                            )

                            # totalDeriv = dFdAct - dRdActT*psi
                            totalDeriv.scale(-1.0)
                            totalDeriv.axpy(1.0, dFdACT)

                            # assign the total derivative to self.adjTotalDeriv
                            self.adjTotalDeriv[objFuncName][designVarName] = np.zeros(nDVs, self.dtype)
                            # we need to convert the parallel vec to seq vec
                            totalDerivSeq = PETSc.Vec().createSeq(nDVs, bsize=1, comm=PETSc.COMM_SELF)
                            self.solver.convertMPIVec2SeqVec(totalDeriv, totalDerivSeq)
                            for i in range(nDVs):
                                self.adjTotalDeriv[objFuncName][designVarName][i] = totalDerivSeq[i]
                            totalDeriv.destroy()
                            totalDerivSeq.destroy()
            ################### Field: field variables (e.g., alphaPorosity, betaSA) as design variable ###################
            elif designVarDict[designVarName]["designVarType"] == "Field":
                if self.getOption("useAD")["mode"] == "reverse":

                    xDV = self.DVGeo.getValues()
                    nDVs = len(xDV[designVarName])
                    fieldType = designVarDict[designVarName]["fieldType"]
                    if fieldType == "scalar":
                        fieldComp = 1
                    elif fieldType == "vector":
                        fieldComp = 3
                    nLocalCells = self.solver.getNLocalCells()

                    # loop over all objectives
                    for objFuncName in objFuncDict:
                        if objFuncName in self.objFuncNames4Adj:

                            # calculate dFdField
                            dFdField = PETSc.Vec().create(PETSc.COMM_WORLD)
                            dFdField.setSizes((fieldComp * nLocalCells, PETSc.DECIDE), bsize=1)
                            dFdField.setFromOptions()
                            self.solverAD.calcdFdFieldAD(
                                self.xvVec, self.wVec, objFuncName.encode(), designVarName.encode(), dFdField
                            )

                            # call the total deriv
                            totalDeriv = PETSc.Vec().create(PETSc.COMM_WORLD)
                            totalDeriv.setSizes((fieldComp * nLocalCells, PETSc.DECIDE), bsize=1)
                            totalDeriv.setFromOptions()
                            # calculate dRdFieldT*Psi and save it to totalDeriv
                            self.solverAD.calcdRdFieldTPsiAD(
                                self.xvVec, self.wVec, self.adjVectors[objFuncName], designVarName.encode(), totalDeriv
                            )

                            # totalDeriv = dFdField - dRdFieldT*psi
                            totalDeriv.scale(-1.0)
                            totalDeriv.axpy(1.0, dFdField)

                            # write the matrix
                            if "dFdFieldTotalDeriv" in self.getOption("writeJacobians") or "all" in self.getOption(
                                "writeJacobians"
                            ):
                                self.writePetscVecMat("dFdFieldTotalDeriv_%s" % objFuncName, totalDeriv)
                                self.writePetscVecMat("dFdFieldTotalDeriv_%s" % objFuncName, totalDeriv, "ASCII")

                            # check if we need to save the sensitivity maps
                            if designVarName in self.getOption("writeSensMap"):
                                # we will write the field sensitivity with the primal solution because they
                                # share the same mesh
                                self.writeFieldSensitivityMap(
                                    objFuncName, designVarName, float(solutionTime), fieldType, totalDeriv
                                )

                            # assign the total derivative to self.adjTotalDeriv
                            self.adjTotalDeriv[objFuncName][designVarName] = np.zeros(nDVs, self.dtype)
                            # we need to convert the parallel vec to seq vec
                            totalDerivSeq = PETSc.Vec().createSeq(nDVs, bsize=1, comm=PETSc.COMM_SELF)
                            self.solver.convertMPIVec2SeqVec(totalDeriv, totalDerivSeq)
                            for i in range(nDVs):
                                self.adjTotalDeriv[objFuncName][designVarName][i] = totalDerivSeq[i]
                            totalDeriv.destroy()
                            totalDerivSeq.destroy()
                            dFdField.destroy()
                else:
                    raise Error("For Field design variable type, we only support useAD->mode=reverse")
            else:
                raise Error("designVarType %s not supported!" % designVarDict[designVarName]["designVarType"])

        self.nSolveAdjoints += 1

        # we destroy dRdWTPC only when we need to recompute it next time
        if (self.nSolveAdjoints - 1) % adjPCLag == 0:
            self.dRdWTPC.destroy()

        return

    def mapdXvTodFFD(self, totalDerivXv):
        """
        Map the Xv derivative (volume derivative) to the FFD derivatives (design variables)
        Essentially, we first map the Xv (volume) to Xs (surface) using IDWarp, then, we
        further map Xs (surface) to FFD using pyGeo

        Input:
        ------
        totalDerivXv: total derivative dFdXv vector

        Output:
        ------
        dFdFFD: the mapped total derivative with respect to FFD variables
        """

        xvSize = len(self.xv) * 3

        dFdXvTotalArray = np.zeros(xvSize, self.dtype)

        Istart, Iend = totalDerivXv.getOwnershipRange()

        for idxI in range(Istart, Iend):
            idxRel = idxI - Istart
            dFdXvTotalArray[idxRel] = totalDerivXv[idxI]

        self.mesh.warpDeriv(dFdXvTotalArray)
        dFdXs = self.mesh.getdXs()
        dFdXs = self.mapVector(dFdXs, self.allWallsGroup, self.designSurfacesGroup)
        dFdFFD = self.DVGeo.totalSensitivity(dFdXs, ptSetName=self.ptSetName, comm=self.comm)

        return dFdFFD

    def getThermal(self, varName, groupName=None):
        """
        Return the forces on this processor on the families defined by groupName.
        Parameters
        ----------

        varName : str
            Which variable to get. Can be either temperature or heatFlux

        groupName : str
            Group identifier to get only forces cooresponding to the
            desired group. The group must be a family or a user-supplied
            group of families. The default is None which corresponds to
            design surfaces.

        Returns
        -------
        thermal : array (N)
            The thermal variables (either temperature or heatFlux) on this processor.
            N is the number of faces on design surface patches
            Note that N may be 0, and an empty array of shape (0) can be returned.
        """

        Info("Computing %s" % varName)

        # Calculate number of surface points
        if groupName is None:
            groupName = self.couplingSurfacesGroup

        nPts, nFaces = self._getSurfaceSize(groupName)

        thermalVec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        thermalVec.setSizes((nFaces, PETSc.DECIDE), bsize=1)
        thermalVec.setFromOptions()

        # Compute forces
        self.solver.getThermal(varName.encode(), thermalVec)

        # Copy data from PETSc vectors
        thermal = np.zeros(nFaces)
        thermal[:] = np.copy(thermalVec.getArray())

        # Cleanup PETSc vectors
        thermalVec.destroy()

        # Print total force
        thermalSum = np.sum(thermal[:])

        thermalSum = self.comm.allreduce(thermalSum, op=MPI.SUM)

        Info("Total %s: %e" % (varName, thermalSum))

        # Finally map the vector as required.
        return thermal

    def getForces(self, groupName=None):
        """
        Return the forces on this processor on the families defined by groupName.
        Parameters
        ----------
        groupName : str
            Group identifier to get only forces cooresponding to the
            desired group. The group must be a family or a user-supplied
            group of families. The default is None which corresponds to
            design surfaces.

        Returns
        -------
        forces : array (N,3)
            Forces on this processor. Note that N may be 0, and an
            empty array of shape (0, 3) can be returned.
        """
        Info("Computing surface forces")
        # Calculate number of surface points
        if groupName is None:
            groupName = self.couplingSurfacesGroup
        nPts, _ = self._getSurfaceSize(groupName)

        fX = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        fX.setSizes((nPts, PETSc.DECIDE), bsize=1)
        fX.setFromOptions()

        fY = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        fY.setSizes((nPts, PETSc.DECIDE), bsize=1)
        fY.setFromOptions()

        fZ = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        fZ.setSizes((nPts, PETSc.DECIDE), bsize=1)
        fZ.setFromOptions()

        # Compute forces
        self.solver.getForces(fX, fY, fZ)

        # Copy data from PETSc vectors
        forces = np.zeros((nPts, 3))
        forces[:, 0] = np.copy(fX.getArray())
        forces[:, 1] = np.copy(fY.getArray())
        forces[:, 2] = np.copy(fZ.getArray())

        # Cleanup PETSc vectors
        fX.destroy()
        fY.destroy()
        fZ.destroy()

        # Print total force
        fXSum = np.sum(forces[:, 0])
        fYSum = np.sum(forces[:, 1])
        fZSum = np.sum(forces[:, 2])

        fXTot = self.comm.allreduce(fXSum, op=MPI.SUM)
        fYTot = self.comm.allreduce(fYSum, op=MPI.SUM)
        fZTot = self.comm.allreduce(fZSum, op=MPI.SUM)

        Info("Total force:")
        Info("Fx = %e" % fXTot)
        Info("Fy = %e" % fYTot)
        Info("Fz = %e" % fZTot)

        # Finally map the vector as required.
        return forces

    def getAcousticData(self, groupName=None):
        """
        Return the acoustic data on this processor.
        Parameters
        ----------
        groupName : str
            Group identifier to get only data cooresponding to the
            desired group. The group must be a family or a user-supplied
            group of families. The default is None which corresponds to
            design surfaces.

        Returns
        -------
        position : array (N,3)
            Face positions on this processor. Note that N may be 0, and an
            empty array of shape (0, 3) can be returned.
        normal : array (N,3)
            Face normals on this processor. Note that N may be 0, and an
            empty array of shape (0, 3) can be returned.
        area : array (N)
            Face areas on this processor. Note that N may be 0, and an
            empty array of shape (0) can be returned.
        forces : array (N,3)
            Face forces on this processor. Note that N may be 0, and an
            empty array of shape (0, 3) can be returned.
        """
        Info("Computing surface acoustic data")
        # Calculate number of surface cells
        if groupName is None:
            raise ValueError("Aeroacoustic grouName not set!")
        _, nCls = self._getSurfaceSize(groupName)

        x = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        x.setSizes((nCls, PETSc.DECIDE), bsize=1)
        x.setFromOptions()

        y = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        y.setSizes((nCls, PETSc.DECIDE), bsize=1)
        y.setFromOptions()

        z = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        z.setSizes((nCls, PETSc.DECIDE), bsize=1)
        z.setFromOptions()

        nX = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        nX.setSizes((nCls, PETSc.DECIDE), bsize=1)
        nX.setFromOptions()

        nY = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        nY.setSizes((nCls, PETSc.DECIDE), bsize=1)
        nY.setFromOptions()

        nZ = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        nZ.setSizes((nCls, PETSc.DECIDE), bsize=1)
        nZ.setFromOptions()

        a = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        a.setSizes((nCls, PETSc.DECIDE), bsize=1)
        a.setFromOptions()

        fX = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        fX.setSizes((nCls, PETSc.DECIDE), bsize=1)
        fX.setFromOptions()

        fY = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        fY.setSizes((nCls, PETSc.DECIDE), bsize=1)
        fY.setFromOptions()

        fZ = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        fZ.setSizes((nCls, PETSc.DECIDE), bsize=1)
        fZ.setFromOptions()

        # Compute forces
        self.solver.getAcousticData(x, y, z, nX, nY, nZ, a, fX, fY, fZ, groupName.encode())

        # Copy data from PETSc vectors
        positions = np.zeros((nCls, 3))
        positions[:, 0] = np.copy(x.getArray())
        positions[:, 1] = np.copy(y.getArray())
        positions[:, 2] = np.copy(z.getArray())
        normals = np.zeros((nCls, 3))
        normals[:, 0] = np.copy(nX.getArray())
        normals[:, 1] = np.copy(nY.getArray())
        normals[:, 2] = np.copy(nZ.getArray())
        areas = np.zeros(nCls)
        areas[:] = np.copy(a.getArray())
        forces = np.zeros((nCls, 3))
        forces[:, 0] = np.copy(fX.getArray())
        forces[:, 1] = np.copy(fY.getArray())
        forces[:, 2] = np.copy(fZ.getArray())

        # Cleanup PETSc vectors
        x.destroy()
        y.destroy()
        z.destroy()
        nX.destroy()
        nY.destroy()
        nZ.destroy()
        a.destroy()
        fX.destroy()
        fY.destroy()
        fZ.destroy()

        # Finally map the vector as required.
        return positions, normals, areas, forces

    def calcTotalDeriv(self, dRdX, dFdX, psi, totalDeriv):
        """
        Compute total derivative

        Input:
        ------
        dRdX, dFdX, and psi

        Output:
        ------
        totalDeriv = dFdX - [dRdX]^T * psi
        """

        dRdX.multTranspose(psi, totalDeriv)
        totalDeriv.scale(-1.0)
        totalDeriv.axpy(1.0, dFdX)

    def calcdFdAOAAnalytical(self, objFuncName, dFdAOA):
        """
        This function computes partials derivatives dFdAlpha with alpha being the angle of attack (AOA)
        We use the analytical method:
        CD = Fx * cos(alpha) + Fy * sin(alpha)
        CL = - Fx * sin(alpha) + Fy * cos(alpha)
        So:
        dCD/dAlpha = - Fx * sin(alpha) + Fy * cos(alpha) = CL
        dCL/dAlpha = - Fx * cos(alpha) - Fy * sin(alpha) = - CD
        NOTE: we need to convert the unit from radian to degree
        """

        objFuncDict = self.getOption("objFunc")

        # find the neededMode of this objective function and also find out if it is a force objective
        neededMode = "None"
        isForceObj = 0
        for objFuncPart in objFuncDict[objFuncName]:
            if objFuncDict[objFuncName][objFuncPart]["type"] == "force":
                isForceObj = 1
                if objFuncDict[objFuncName][objFuncPart]["directionMode"] == "fixedDirection":
                    raise Error("AOA derivative does not support directionMode=fixedDirection!")
                elif objFuncDict[objFuncName][objFuncPart]["directionMode"] == "parallelToFlow":
                    neededMode = "normalToFlow"
                    break
                elif objFuncDict[objFuncName][objFuncPart]["directionMode"] == "normalToFlow":
                    neededMode = "parallelToFlow"
                    break
                else:
                    raise Error("directionMode not valid!")

        # if it is a forceObj, use the analytical approach to calculate dFdAOA, otherwise set it to zero
        if isForceObj == 1:
            # loop over all objectives again to find the neededMode
            # Note that if the neededMode == "parallelToFlow", we need to add a minus sign
            for objFuncNameNeeded in objFuncDict:
                for objFuncPart in objFuncDict[objFuncNameNeeded]:
                    if objFuncDict[objFuncNameNeeded][objFuncPart]["type"] == "force":
                        if objFuncDict[objFuncNameNeeded][objFuncPart]["directionMode"] == neededMode:
                            val = self.solver.getObjFuncValue(objFuncNameNeeded.encode())
                            if neededMode == "parallelToFlow":
                                dFdAOA[0] = -val * np.pi / 180.0
                            elif neededMode == "normalToFlow":
                                dFdAOA[0] = val * np.pi / 180.0
                            dFdAOA.assemblyBegin()
                            dFdAOA.assemblyEnd()
                            break
        else:
            dFdAOA.zeroEntries()

    def _initSolver(self):
        """
        Initialize the solvers. This needs to be called before calling any runs
        """

        if self.solverInitialized == 1:
            raise Error("pyDAFoam: self._initSolver has been called! One shouldn't initialize solvers twice!")

        solverName = self.getOption("solverName")
        solverArg = solverName + " -python " + self.parallelFlag
        if solverName in self.solverRegistry["Incompressible"]:

            from .pyDASolverIncompressible import pyDASolvers

            self.solver = pyDASolvers(solverArg.encode(), self.options)

            if self.getOption("useAD")["mode"] == "forward":

                from .pyDASolverIncompressibleADF import pyDASolvers as pyDASolversAD

                self.solverAD = pyDASolversAD(solverArg.encode(), self.options)

            elif self.getOption("useAD")["mode"] == "reverse":

                from .pyDASolverIncompressibleADR import pyDASolvers as pyDASolversAD

                self.solverAD = pyDASolversAD(solverArg.encode(), self.options)

        elif solverName in self.solverRegistry["Compressible"]:

            from .pyDASolverCompressible import pyDASolvers

            self.solver = pyDASolvers(solverArg.encode(), self.options)

            if self.getOption("useAD")["mode"] == "forward":

                from .pyDASolverCompressibleADF import pyDASolvers as pyDASolversAD

                self.solverAD = pyDASolversAD(solverArg.encode(), self.options)

            elif self.getOption("useAD")["mode"] == "reverse":

                from .pyDASolverCompressibleADR import pyDASolvers as pyDASolversAD

                self.solverAD = pyDASolversAD(solverArg.encode(), self.options)

        elif solverName in self.solverRegistry["Solid"]:

            from .pyDASolverSolid import pyDASolvers

            self.solver = pyDASolvers(solverArg.encode(), self.options)

            if self.getOption("useAD")["mode"] == "forward":

                from .pyDASolverSolidADF import pyDASolvers as pyDASolversAD

                self.solverAD = pyDASolversAD(solverArg.encode(), self.options)

            elif self.getOption("useAD")["mode"] == "reverse":

                from .pyDASolverSolidADR import pyDASolvers as pyDASolversAD

                self.solverAD = pyDASolversAD(solverArg.encode(), self.options)
        else:
            raise Error("pyDAFoam: %s not registered! Check _solverRegistry(self)." % solverName)

        self.solver.initSolver()

        if self.getOption("useAD")["mode"] in ["forward", "reverse"]:
            self.solverAD.initSolver()

        if self.getOption("printDAOptions"):
            self.solver.printAllOptions()

        adjMode = self.getOption("unsteadyAdjoint")["mode"]
        if adjMode == "hybridAdjoint" or adjMode == "timeAccurateAdjoint":
            self.initTimeInstanceMats()

        self.solverInitialized = 1

        return

    def runColoring(self):
        """
        Run coloring solver
        """

        Info("\n")
        Info("+--------------------------------------------------------------------------+")
        Info("|                       Running Coloring Solver                            |")
        Info("+--------------------------------------------------------------------------+")

        solverName = self.getOption("solverName")
        if solverName in self.solverRegistry["Incompressible"]:

            from .pyColoringIncompressible import pyColoringIncompressible

            solverArg = "ColoringIncompressible -python " + self.parallelFlag
            solver = pyColoringIncompressible(solverArg.encode(), self.options)
        elif solverName in self.solverRegistry["Compressible"]:

            from .pyColoringCompressible import pyColoringCompressible

            solverArg = "ColoringCompressible -python " + self.parallelFlag
            solver = pyColoringCompressible(solverArg.encode(), self.options)
        elif solverName in self.solverRegistry["Solid"]:

            from .pyColoringSolid import pyColoringSolid

            solverArg = "ColoringSolid -python " + self.parallelFlag
            solver = pyColoringSolid(solverArg.encode(), self.options)
        else:
            raise Error("pyDAFoam: %s not registered! Check _solverRegistry(self)." % solverName)
        solver.run()

        solver = None

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

        if self.comm.rank == 0:
            status = subprocess.call("decomposePar", stdout=sys.stdout, stderr=subprocess.STDOUT, shell=False)
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
        format like 0.00000001, 0.00000002, etc. One can load these intermediate shapes and fields and
        plot them in paraview.
        The way it is implemented is that we sort the solution folder and consider the largest time folder
        as the solution folder and rename it

        Parameters
        ----------
        solIndex: int
            The major interation index
        """

        allSolutions = []
        rootDir = os.getcwd()
        if self.parallel:
            checkPath = os.path.join(rootDir, "processor%d" % self.comm.rank)
        else:
            checkPath = rootDir

        folderNames = os.listdir(checkPath)
        for folderName in folderNames:
            try:
                float(folderName)
                allSolutions.append(folderName)
            except ValueError:
                continue
        allSolutions.sort(reverse=True)
        # choose the latst solution to rename
        solutionTime = allSolutions[0]

        if float(solutionTime) < 1e-4:
            Info("Solution time %g less than 1e-4, not renamed." % float(solutionTime))
            renamed = False
            return solutionTime, renamed

        distTime = "%.8f" % (solIndex / 1e8)

        src = os.path.join(checkPath, solutionTime)
        dst = os.path.join(checkPath, distTime)

        Info("Moving time %s to %s" % (solutionTime, distTime))

        if os.path.isdir(dst):
            raise Error("%s already exists, moving failed!" % dst)
        else:
            try:
                shutil.move(src, dst)
            except Exception:
                raise Error("Can not move %s to %s" % (src, dst))

        renamed = True
        return distTime, renamed

    def calcFFD2XvSeedVec(self):
        """
        Calculate the FFD2XvSeedVec vector:
        Given a FFD seed xDvDot, run pyGeo and IDWarp and propagate the seed to Xv seed xVDot:
            xSDot = \\frac{dX_{S}}{dX_{DV}}\\xDvDot
            xVDot = \\frac{dX_{V}}{dX_{S}}\\xSDot

        Then, we assign this vector to FFD2XvSeedVec in DASolver
        This will be used in forward mode AD runs
        """

        if self.DVGeo is None:
            raise Error("DVGeo not set!")

        dvName = self.getOption("useAD")["dvName"]
        seedIndex = self.getOption("useAD")["seedIndex"]
        # create xDVDot vec and initialize it with zeros
        xDV = self.DVGeo.getValues()

        # create a copy of xDV and set the seed to 1.0
        # the dv and index depends on dvName and seedIndex
        xDvDot = {}
        for key in list(xDV.keys()):
            xDvDot[key] = np.zeros_like(xDV[key], dtype=self.dtype)
        xDvDot[dvName][seedIndex] = 1.0

        # get the original surf coords
        xSDot0 = np.zeros_like(self.xs0, self.dtype)
        xSDot0 = self.mapVector(xSDot0, self.allSurfacesGroup, self.designSurfacesGroup)

        # get xSDot
        xSDot = self.DVGeo.totalSensitivityProd(xDvDot, ptSetName=self.ptSetName).reshape(xSDot0.shape)
        # get xVDot
        xVDot = self.mesh.warpDerivFwd(xSDot)

        seedVec = self.xvVec.duplicate()
        seedVec.zeroEntries()
        Istart, Iend = seedVec.getOwnershipRange()

        # assign xVDot to seedVec
        for idx in range(Istart, Iend):
            idxRel = idx - Istart
            seedVec[idx] = xVDot[idxRel]

        seedVec.assemblyBegin()
        seedVec.assemblyEnd()

        self.solverAD.setFFD2XvSeedVec(seedVec)

    def setdXvdFFDMat(self, designVarName, deltaVPointThreshold=1.0e-16):
        """
        Perturb each design variable and save the delta volume point coordinates
        to a mat, this will be used to calculate dRdFFD and dFdFFD in DAFoam

        Parameters
        ----------
        deltaVPointThreshold: float
            A threshold, any delta volume coordinates smaller than this value will be ignored

        """

        if self.DVGeo is None:
            raise Error("DVGeo not set!")

        # Get the FFD size
        nDVs = -9999
        xDV = self.DVGeo.getValues()
        nDVs = len(xDV[designVarName])

        # get the unperturbed point coordinates
        oldVolPoints = self.mesh.getSolverGrid()
        # get the size of xv, it is the number of points * 3
        nXvs = len(oldVolPoints)
        # get eps
        epsFFD = self.getOption("adjPartDerivFDStep")["FFD"]

        Info("Calculating the dXvdFFD matrix with epsFFD: " + str(epsFFD))

        dXvdFFDMat = PETSc.Mat().create(PETSc.COMM_WORLD)
        dXvdFFDMat.setSizes(((nXvs, None), (None, nDVs)))
        dXvdFFDMat.setFromOptions()
        dXvdFFDMat.setPreallocationNNZ((nDVs, nDVs))
        dXvdFFDMat.setUp()
        Istart, Iend = dXvdFFDMat.getOwnershipRange()

        # for each DV, perturb epsFFD and save the delta vol point coordinates
        for i in range(nDVs):
            # perturb
            xDV[designVarName][i] += epsFFD
            # set the dv to DVGeo
            self.DVGeo.setDesignVars(xDV)
            # update the vol points according to the new DV values
            self.updateVolumePoints()
            # get the new vol points
            newVolPoints = self.mesh.getSolverGrid()
            # assign the delta vol coords to the mat
            for idx in range(Istart, Iend):
                idxRel = idx - Istart
                deltaVal = newVolPoints[idxRel] - oldVolPoints[idxRel]
                if abs(deltaVal) > deltaVPointThreshold:  # a threshold
                    dXvdFFDMat[idx, i] = deltaVal
            # reset the perturbation of the dv
            xDV[designVarName][i] -= epsFFD

        # reset the volume mesh coordinates
        self.DVGeo.setDesignVars(xDV)
        self.updateVolumePoints()

        # assemble
        dXvdFFDMat.assemblyBegin()
        dXvdFFDMat.assemblyEnd()

        # viewer = PETSc.Viewer().createASCII("dXvdFFDMat_%s_%s.dat" % (designVarName, self.comm.size), "w")
        # viewer(dXvdFFDMat)

        self.solver.setdXvdFFDMat(dXvdFFDMat)

        dXvdFFDMat.destroy()

        return nDVs

    def updateVolumePoints(self):
        """
        Update the vol mesh point coordinates based on the current values of design variables
        """

        # update the CFD Coordinates
        if self.DVGeo is not None:
            if self.ptSetName not in self.DVGeo.points:
                xs0 = self.mapVector(self.xs0, self.allSurfacesGroup, self.designSurfacesGroup)
                self.DVGeo.addPointSet(xs0, self.ptSetName)
                self.pointsSet = True

            # set the surface coords
            if not self.DVGeo.pointSetUpToDate(self.ptSetName):
                coords = self.DVGeo.update(self.ptSetName, config=None)
                self.setSurfaceCoordinates(coords, self.designSurfacesGroup)

            # warp the mesh
            self.mesh.warpMesh()

        return

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
        if self.getOption("useAD")["mode"] in ["forward", "reverse"]:
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

    def getPointSetName(self):
        """
        Take the apName and return the mangled point set name.
        """
        return "openFoamCoords"

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

    def _initializeMeshPointVec(self):
        """
        Initialize the mesh point vec: xvVec
        """

        xvSize = len(self.xv) * 3
        self.xvVec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        self.xvVec.setSizes((xvSize, PETSc.DECIDE), bsize=1)
        self.xvVec.setFromOptions()

        self.xv2XvVec(self.xv, self.xvVec)

        # viewer = PETSc.Viewer().createASCII("xvVec", comm=PETSc.COMM_WORLD)
        # viewer(self.xvVec)

        return

    def _initializeStateVec(self):
        """
        Initialize state variable vector
        """

        wSize = self.solver.getNLocalAdjointStates()
        self.wVec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        self.wVec.setSizes((wSize, PETSc.DECIDE), bsize=1)
        self.wVec.setFromOptions()

        self.solver.ofField2StateVec(self.wVec)

        # viewer = PETSc.Viewer().createASCII("wVec", comm=PETSc.COMM_WORLD)
        # viewer(self.wVec)

        # if it is a multipoint case, initialize self.wVecMPList
        if self.getOption("multiPoint") is True:
            nMultiPoints = self.getOption("nMultiPoints")
            self.wVecMPList = [None] * self.getOption("nMultiPoints")
            for i in range(nMultiPoints):
                self.wVecMPList[i] = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
                self.wVecMPList[i].setSizes((wSize, PETSc.DECIDE), bsize=1)
                self.wVecMPList[i].setFromOptions()

        return

    def xv2XvVec(self, xv, xvVec):
        """
        Convert a Nx3 mesh point numpy array to a Petsc xvVec
        """

        xSize = len(xv)

        for i in range(xSize):
            for j in range(3):
                globalIdx = self.solver.getGlobalXvIndex(i, j)
                xvVec[globalIdx] = xv[i][j]

        xvVec.assemblyBegin()
        xvVec.assemblyEnd()

        return

    def xvFlatten2XvVec(self, xv, xvVec):
        """
        Convert a 3Nx1 mesh point numpy array to a Petsc xvVec
        """

        xSize = len(xv)
        xSize = int(xSize // 3)

        counterI = 0
        for i in range(xSize):
            for j in range(3):
                globalIdx = self.solver.getGlobalXvIndex(i, j)
                xvVec[globalIdx] = xv[counterI]
                counterI += 1

        xvVec.assemblyBegin()
        xvVec.assemblyEnd()

        return

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
            'objFunc': [dict, {
                'name': 'CD',
                'direction': [1.0, 0.0, 0.0],
                'scale': 1.0}]
        }

        Then, calling self.setOption('objFunc', {'name': 'CL'}) will give:

        self.options =
        {
            'objFunc': [dict, {
                'name': 'CL',
                'direction': [1.0, 0.0, 0.0],
                'scale': 1.0}]
        }

        INSTEAD OF

        self.options =
        {
            'objFunc': [dict, {'name': 'CL'}]
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

    def setFieldValue4GlobalCellI(self, fieldName, val, globalCellI, compI=0):
        """
        Set the field value based on the global cellI. This is usually
        used if the state variables are design variables, e.g., betaSA
        The reason to use global cell index, instead of local one, is
        because this index is usually provided by the optimizer. Optimizer
        uses global cell index as the design variable

        Parameters
        ----------
        fieldName : str
           Name of the flow field to set, e.g., U, p, nuTilda
        val : float
           The value to set
        globalCellI : int
           The global cell index to set the value
        compI : int
           The component index to set the value (for vectorField only)

        """

        self.solver.setFieldValue4GlobalCellI(fieldName, val, globalCellI, compI)

    def setFieldValue4LocalCellI(self, fieldName, val, localCellI, compI=0):
        """
        Set the field value based on the local cellI.

        Parameters
        ----------
        fieldName : str
           Name of the flow field to set, e.g., U, p, nuTilda
        val : float
           The value to set
        localCellI : int
           The global cell index to set the value
        compI : int
           The component index to set the value (for vectorField only)

        """

        self.solver.setFieldValue4LocalCellI(fieldName, val, localCellI, compI)

    def updateBoundaryConditions(self, fieldName, fieldType):
        """
        Update the boundary condition for a field

        Parameters
        ----------
        fieldName : str
           Name of the flow field to update, e.g., U, p, nuTilda
        fieldType : str
           Type of the flow field: scalar or vector

        """

        self.solver.updateBoundaryConditions(fieldName, fieldType)

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

        if len(self.getOption("fvSource")) > 0:
            self.syncDAOptionToActuatorDVs()

    def syncDAOptionToActuatorDVs(self):
        """
        Synchronize the values in DAOption and actuatorDiskDVs_. We need to synchronize the values
        defined in fvSource from DAOption to actuatorDiskDVs_ in the C++ layer
        NOTE: we need to call this function whenever we change the actuator design variables
        during optimization.
        """

        self.solver.syncDAOptionToActuatorDVs()

        if self.getOption("useAD")["mode"] in ["forward", "reverse"]:
            self.solverAD.syncDAOptionToActuatorDVs()

    def getNLocalAdjointStates(self):
        """
        Get number of local adjoint states
        """
        return self.solver.getNLocalAdjointStates()

    def getDVsCons(self):
        """
        Return the list of design variable names
        NOTE: constraints are not implemented yet
        """
        DVNames = []
        DVSizes = []
        if self.DVGeo is None:
            return None
        else:
            DVs = self.DVGeo.getValues()
            for dvName in DVs:
                try:
                    size = len(DVs[dvName])
                except Exception:
                    size = 1
                DVNames.append(dvName)
                DVSizes.append(size)
            return DVNames, DVSizes

    def getStates(self):
        """
        Return the adjoint state array owns by this processor
        """
        nLocalStateSize = self.solver.getNLocalAdjointStates()
        states = np.zeros(nLocalStateSize, self.dtype)
        Istart, Iend = self.wVec.getOwnershipRange()
        for i in range(Istart, Iend):
            iRel = i - Istart
            states[iRel] = self.wVec[i]

        return states

    def getResiduals(self):
        """
        Return the residual array owns by this processor
        """
        nLocalStateSize = self.solver.getNLocalAdjointStates()
        residuals = np.zeros(nLocalStateSize, self.dtype)
        resVec = self.wVec.duplicate()
        resVec.zeroEntries()

        self.solver.calcResidualVec(resVec)

        Istart, Iend = resVec.getOwnershipRange()
        for i in range(Istart, Iend):
            iRel = i - Istart
            residuals[iRel] = resVec[i]

        return residuals

    def setStates(self, states):
        """
        Set the state to the OpenFOAM field
        """
        Istart, Iend = self.wVec.getOwnershipRange()
        for i in range(Istart, Iend):
            iRel = i - Istart
            self.wVec[i] = states[iRel]

        self.wVec.assemblyBegin()
        self.wVec.assemblyEnd()

        self.solver.updateOFField(self.wVec)

        if self.getOption("useAD")["mode"] in ["forward", "reverse"]:
            self.solverAD.updateOFField(self.wVec)

        return

    def convertMPIVec2SeqArray(self, mpiVec):
        """
        Convert a MPI vector to a seq array
        """
        vecSize = mpiVec.getSize()
        seqVec = PETSc.Vec().createSeq(vecSize, bsize=1, comm=PETSc.COMM_SELF)
        self.solver.convertMPIVec2SeqVec(mpiVec, seqVec)

        array1 = np.zeros(vecSize, self.dtype)
        for i in range(vecSize):
            array1[i] = seqVec[i]

        return array1

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

    def array2VecSeq(self, array1):
        """
        Convert a numpy array to Petsc vector in serial mode
        """
        size = len(array1)

        vec = PETSc.Vec().createSeq(size, bsize=1, comm=PETSc.COMM_SELF)
        vec.zeroEntries()

        for i in range(size):
            vec[i] = array1[i]

        vec.assemblyBegin()
        vec.assemblyEnd()

        return vec

    def vec2ArraySeq(self, vec):
        """
        Convert a Petsc vector to numpy array in serial mode
        """

        size = vec.getSize()
        array1 = np.zeros(size, self.dtype)
        for i in range(size):
            array1[i] = vec[i]
        return array1

    def _printCurrentOptions(self):
        """
        Prints a nicely formatted dictionary of all the current solver
        options to the stdout on the root processor
        """

        Info("+---------------------------------------+")
        Info("|         All DAFoam Options:           |")
        Info("+---------------------------------------+")
        # Need to assemble a temporary dictionary
        tmpDict = {}
        for key in self.options:
            tmpDict[key] = self.getOption(key)
        Info(tmpDict)

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
