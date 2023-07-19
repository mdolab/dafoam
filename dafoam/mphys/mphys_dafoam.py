import sys
from openmdao.api import Group, ImplicitComponent, ExplicitComponent, AnalysisError
import openmdao.api as om
from dafoam import PYDAFOAM
from idwarp import USMesh
from mphys.builder import Builder
import petsc4py
from petsc4py import PETSc
import numpy as np
from mpi4py import MPI
from mphys import MaskedConverter, UnmaskedConverter, MaskedVariableDescription
from mphys.utils.directory_utils import cd

petsc4py.init(sys.argv)


class DAFoamBuilder(Builder):
    """
    DAFoam builder called from runScript.py
    """

    def __init__(
        self,
        options,  # DAFoam options
        mesh_options=None,  # IDWarp options
        scenario="aerodynamic",  # scenario type to configure the groups
        prop_coupling=None,
        run_directory="",  # the directory to run this case in, default is the current directory
    ):

        # options dictionary for DAFoam
        self.options = options

        # mesh warping option. If no design variables are mesh related,
        # e.g., topology optimization, set it to None.
        self.mesh_options = mesh_options
        # flag to determine if the mesh warping component is added
        # in the nonlinear solver loop (e.g. for aerostructural)
        # or as a preprocessing step like the surface mesh coordinates
        # (e.g. for aeropropulsive). This will avoid doing extra work
        # for mesh deformation when the volume mesh does not change
        # during nonlinear iterations
        self.warp_in_solver = False
        # flag for aerostructural coupling variables
        self.struct_coupling = False
        # thermal coupling defaults to false
        self.thermal_coupling = False

        # the directory to run this case in, default is the current directory
        self.run_directory = run_directory

        # flag for aero-propulsive coupling variables
        self.prop_coupling = prop_coupling

        # depending on the scenario we are building for, we adjust a few internal parameters:
        if scenario.lower() == "aerodynamic":
            # default
            pass
        elif scenario.lower() == "aerostructural":
            # volume mesh warping needs to be inside the coupling loop for aerostructural
            self.warp_in_solver = True
            self.struct_coupling = True
        elif scenario.lower() == "aerothermal":
            # volume mesh warping needs to be inside the coupling loop for aerothermal
            self.thermal_coupling = True
        else:
            raise AnalysisError(
                "scenario %s not valid! Options: aerodynamic, aerostructural, and aerothermal" % scenario
            )

    # api level method for all builders
    def initialize(self, comm):

        self.comm = comm

        with cd(self.run_directory):
            # initialize the PYDAFOAM class, defined in pyDAFoam.py
            self.DASolver = PYDAFOAM(options=self.options, comm=comm)
            if self.mesh_options is not None:
                # always set the mesh
                mesh = USMesh(options=self.mesh_options, comm=comm)
                self.DASolver.setMesh(mesh)  # add the design surface family group
                self.DASolver.printFamilyList()

    def get_solver(self):
        # this method is only used by the RLT transfer scheme
        return self.DASolver

    # api level method for all builders
    def get_coupling_group_subsystem(self, scenario_name=None):
        dafoam_group = DAFoamGroup(
            solver=self.DASolver,
            use_warper=self.warp_in_solver,
            struct_coupling=self.struct_coupling,
            prop_coupling=self.prop_coupling,
            thermal_coupling=self.thermal_coupling,
            run_directory=self.run_directory,
        )
        return dafoam_group

    def get_mesh_coordinate_subsystem(self, scenario_name=None):

        # just return the component that outputs the surface mesh.
        return DAFoamMesh(solver=self.DASolver)

    def get_pre_coupling_subsystem(self, scenario_name=None):

        if self.mesh_options is None:
            return None
        else:
            return DAFoamPrecouplingGroup(
                solver=self.DASolver, warp_in_solver=self.warp_in_solver, thermal_coupling=self.thermal_coupling
            )

    def get_post_coupling_subsystem(self, scenario_name=None):
        return DAFoamPostcouplingGroup(solver=self.DASolver)

    def get_number_of_nodes(self, groupName=None):
        # Get number of aerodynamic nodes
        if groupName is None:
            groupName = self.DASolver.couplingSurfacesGroup
        nodes = int(self.DASolver.getSurfaceCoordinates(groupName=groupName).size / 3)

        # Add fictitious nodes to root proc, if they are used
        if self.comm.rank == 0:
            aerostructDict = self.DASolver.getOption("couplingInfo")["aerostructural"]
            fvSourceDict = self.DASolver.getOption("fvSource")
            if aerostructDict["active"] and aerostructDict["propMovement"]:
                if "fvSource" in aerostructDict.keys():
                    # Iterate through Actuator Disks
                    for fvSource, parameters in aerostructDict["fvSource"].items():
                        # Check if Actuator Disk Exists
                        if fvSource not in fvSourceDict:
                            raise RuntimeWarning("Actuator disk {} not found when adding masked nodes".format(fvSource))

                        # Count Nodes
                        nodes += 1 + parameters["nNodes"]
        return nodes


class DAFoamGroup(Group):
    """
    DAFoam solver group
    """

    def initialize(self):
        self.options.declare("solver", recordable=False)
        self.options.declare("struct_coupling", default=False)
        self.options.declare("use_warper", default=True)
        self.options.declare("prop_coupling", default=None)
        self.options.declare("thermal_coupling", default=False)
        self.options.declare("run_directory", default="")

    def setup(self):

        self.DASolver = self.options["solver"]
        self.struct_coupling = self.options["struct_coupling"]
        self.use_warper = self.options["use_warper"]
        self.prop_coupling = self.options["prop_coupling"]
        self.thermal_coupling = self.options["thermal_coupling"]
        self.run_directory = self.options["run_directory"]
        self.discipline = self.DASolver.getOption("discipline")

        if self.prop_coupling is not None:
            if self.prop_coupling not in ["Prop", "Wing"]:
                raise AnalysisError("prop_coupling can be either Wing or Prop, while %s is given!" % self.prop_coupling)

        aerostructDict = self.DASolver.getOption("couplingInfo")["aerostructural"]
        if self.use_warper:
            # Setup node masking
            self.mphys_set_masking()

            # Add propeller movement, if enabled
            if aerostructDict["active"] and aerostructDict["propMovement"]:
                prop_movement = DAFoamActuator(solver=self.DASolver)
                self.add_subsystem("prop_movement", prop_movement, promotes_inputs=["*"], promotes_outputs=["*"])

            # if we dont have geo_disp, we also need to promote the x_a as x_a0 from the deformer component
            self.add_subsystem(
                "deformer",
                DAFoamWarper(
                    solver=self.DASolver,
                ),
                promotes_inputs=[("x_%s" % self.discipline, "x_%s_masked" % self.discipline)],
                promotes_outputs=["%s_vol_coords" % self.discipline],
            )
        elif aerostructDict["active"] and aerostructDict["propMovement"]:
            raise RuntimeError(
                "Propeller movement not possible when the warper is outside of the solver. Check for a valid scenario."
            )

        if self.prop_coupling is not None:
            if self.prop_coupling == "Wing":
                self.add_subsystem(
                    "source",
                    DAFoamFvSource(solver=self.DASolver),
                    promotes_inputs=["*"],
                    promotes_outputs=["*"],
                )

        # add the solver implicit component
        self.add_subsystem(
            "solver",
            DAFoamSolver(solver=self.DASolver, prop_coupling=self.prop_coupling, run_directory=self.run_directory),
            promotes_inputs=["*"],
            promotes_outputs=["%s_states" % self.discipline],
        )

        if self.prop_coupling is not None:
            if self.prop_coupling == "Prop":

                activeProp = None
                nActiveProps = 0

                wingPropNameList = list(self.DASolver.getOption("wingProp").keys())

                for propName in wingPropNameList:
                    if self.DASolver.getOption("wingProp")[propName]["active"]:
                        activeProp = propName
                        nActiveProps += 1

                if nActiveProps != 1:
                    raise RuntimeError(
                        "if prop_coupling = Prop, we should set only one active propeller for wingProp key."
                    )

                self.add_subsystem(
                    activeProp,
                    DAFoamPropForce(solver=self.DASolver, propName=activeProp),
                    promotes_inputs=["*"],
                    promotes_outputs=["*"],
                )

        if self.struct_coupling:
            self.add_subsystem(
                "force",
                DAFoamForces(solver=self.DASolver),
                promotes_inputs=["%s_vol_coords" % self.discipline, "%s_states" % self.discipline],
                promotes_outputs=[("f_aero", "f_aero_masked")],
            )

        if self.thermal_coupling:

            self.add_subsystem(
                "get_%s" % self.discipline,
                DAFoamThermal(solver=self.DASolver),
                promotes_inputs=["*"],
                promotes_outputs=["*"],
            )

        # Setup unmasking
        self.mphys_set_unmasking(forces=self.struct_coupling)

    def mphys_compute_nodes(self):
        aerostructDict = self.DASolver.getOption("couplingInfo")["aerostructural"]
        fvSourceDict = self.DASolver.getOption("fvSource")

        # Check if Actuator Disk Definitions Exist, only add to Root Proc
        nodes_prop = 0
        if self.comm.rank == 0:
            if aerostructDict["active"] and aerostructDict["propMovement"]:
                if "fvSource" in aerostructDict.keys():
                    # Iterate through Actuator Disks
                    for fvSource, parameters in aerostructDict["fvSource"].items():
                        # Check if Actuator Disk Exists
                        if fvSource not in fvSourceDict:
                            raise RuntimeWarning("Actuator disk %s not found when adding masked nodes" % fvSource)

                        # Count Nodes
                        nodes_prop += 1 + parameters["nNodes"]

        # Compute number of aerodynamic nodes
        nodes_aero = int(self.DASolver.getSurfaceCoordinates(groupName=self.DASolver.designSurfacesGroup).size / 3)

        # Sum nodes and return all values
        nodes_total = nodes_aero + nodes_prop
        return nodes_total, nodes_aero, nodes_prop

    def mphys_set_masking(self):
        # Retrieve number of nodes in each category
        nodes_total, nodes_aero, nodes_prop = self.mphys_compute_nodes()

        aerostructDict = self.DASolver.getOption("couplingInfo")["aerostructural"]

        mask = []
        output = []
        promotes_inputs = []
        promotes_outputs = []

        # Mesh Coordinate Mask
        mask.append(np.zeros([(nodes_total) * 3], dtype=bool))
        mask[0][:] = True

        if nodes_prop > 0:
            mask[0][3 * nodes_aero :] = False

        output.append(
            MaskedVariableDescription("x_%s_masked" % self.discipline, shape=(nodes_aero) * 3, tags=["mphys_coupling"])
        )

        promotes_outputs.append("x_%s_masked" % self.discipline)

        # Add Propeller Masks
        if aerostructDict["active"] and aerostructDict["propMovement"]:
            if "fvSource" in aerostructDict.keys():
                i_fvSource = 0
                i_start = 3 * nodes_aero
                for fvSource, parameters in aerostructDict["fvSource"].items():
                    mask.append(np.zeros([(nodes_total) * 3], dtype=bool))
                    mask[i_fvSource + 1][:] = False

                    if self.comm.rank == 0:
                        mask[i_fvSource + 1][i_start : i_start + 3 * (1 + parameters["nNodes"])] = True
                        i_start += 3 * (1 + parameters["nNodes"])

                        output.append(
                            MaskedVariableDescription(
                                "x_prop_%s" % fvSource, shape=((1 + parameters["nNodes"])) * 3, tags=["mphys_coupling"]
                            )
                        )
                    else:
                        output.append(
                            MaskedVariableDescription("x_prop_%s" % fvSource, shape=(0), tags=["mphys_coupling"])
                        )

                    promotes_outputs.append("x_prop_%s" % fvSource)

                    i_fvSource += 1

        # Define Mask
        input = MaskedVariableDescription("x_%s" % self.discipline, shape=(nodes_total) * 3, tags=["mphys_coupling"])
        promotes_inputs.append("x_%s" % self.discipline)
        masker = MaskedConverter(input=input, output=output, mask=mask, distributed=True, init_output=0.0)
        self.add_subsystem("masker", masker, promotes_inputs=promotes_inputs, promotes_outputs=promotes_outputs)

    def mphys_set_unmasking(self, forces=False):
        # Retrieve number of nodes in each category
        nodes_total, nodes_aero, nodes_prop = self.mphys_compute_nodes()

        # If forces are active, generate mask
        if forces:
            aerostructDict = self.DASolver.getOption("couplingInfo")["aerostructural"]

            mask = []
            input = []
            promotes_inputs = []
            promotes_outputs = []

            # Mesh Coordinate Mask
            mask.append(np.zeros([(nodes_total) * 3], dtype=bool))
            mask[0][:] = True
            if nodes_prop > 0:
                mask[0][3 * nodes_aero :] = False
            input.append(MaskedVariableDescription("f_aero_masked", shape=(nodes_aero) * 3, tags=["mphys_coupling"]))
            promotes_inputs.append("f_aero_masked")

            if aerostructDict["active"] and aerostructDict["propMovement"]:
                if "fvSource" in aerostructDict.keys():
                    # Add Propeller Masks
                    i_fvSource = 0
                    i_start = 3 * nodes_aero
                    for fvSource, parameters in aerostructDict["fvSource"].items():
                        mask.append(np.zeros([(nodes_total) * 3], dtype=bool))
                        mask[i_fvSource + 1][:] = False

                        if self.comm.rank == 0:
                            mask[i_fvSource + 1][i_start : i_start + 3 * (1 + parameters["nNodes"])] = True
                            i_start += 3 * (1 + parameters["nNodes"])

                            input.append(
                                MaskedVariableDescription(
                                    "f_prop_%s" % fvSource,
                                    shape=((1 + parameters["nNodes"])) * 3,
                                    tags=["mphys_coordinates"],
                                )
                            )
                        else:
                            input.append(
                                MaskedVariableDescription("f_prop_%s" % fvSource, shape=(0), tags=["mphys_coupling"])
                            )
                        promotes_inputs.append("f_prop_%s" % fvSource)

                        i_fvSource += 1

            # Define Mask
            output = MaskedVariableDescription("f_aero", shape=(nodes_total) * 3, tags=["mphys_coupling"])
            promotes_outputs.append("f_aero")
            unmasker = UnmaskedConverter(input=input, output=output, mask=mask, distributed=True, default_values=0.0)
            self.add_subsystem(
                "force_unmasker", unmasker, promotes_inputs=promotes_inputs, promotes_outputs=promotes_outputs
            )

    def mphys_set_options(self, optionDict):
        # here optionDict should be a dictionary that has a consistent format
        # with the daOptions defined in the run script
        self.solver.set_options(optionDict)


class DAFoamPrecouplingGroup(Group):
    """
    Pre-coupling group that configures any components that happen before the solver and post-processor.
    """

    def initialize(self):
        self.options.declare("solver", default=None, recordable=False)
        self.options.declare("warp_in_solver", default=None, recordable=False)
        self.options.declare("thermal_coupling", default=None, recordable=False)

    def setup(self):
        self.DASolver = self.options["solver"]
        self.warp_in_solver = self.options["warp_in_solver"]
        self.thermal_coupling = self.options["thermal_coupling"]
        self.discipline = self.DASolver.getOption("discipline")

        aerostructDict = self.DASolver.getOption("couplingInfo")["aerostructural"]

        # Return the warper only if it is not in the solver
        if not self.warp_in_solver:
            if aerostructDict["active"] and aerostructDict["propMovement"]:
                raise RuntimeError(
                    "Propeller movement not possible when the warper is outside of the solver. Check for a valid scenario."
                )

            self.add_subsystem(
                "warper",
                DAFoamWarper(solver=self.DASolver),
                promotes_inputs=["x_%s" % self.discipline],
                promotes_outputs=["%s_vol_coords" % self.discipline],
            )

        # If the warper is in the solver, add other pre-coupling groups if desired
        else:
            fvSourceDict = self.DASolver.getOption("fvSource")
            nodes_prop = 0

            # Add propeller nodes and subsystem if needed
            if aerostructDict["active"] and aerostructDict["propMovement"]:
                self.add_subsystem(
                    "prop_nodes", DAFoamPropNodes(solver=self.DASolver), promotes_inputs=["*"], promotes_outputs=["*"]
                )

                # Only add to Root Proc
                if self.comm.rank == 0:
                    if "fvSource" in aerostructDict.keys():
                        # Iterate through Actuator Disks
                        for fvSource, parameters in aerostructDict["fvSource"].items():
                            # Check if Actuator Disk Exists
                            if fvSource not in fvSourceDict:
                                raise RuntimeWarning("Actuator disk %s not found when adding masked nodes" % fvSource)

                            # Count Nodes
                            nodes_prop += 1 + parameters["nNodes"]

            nodes_aero = int(self.DASolver.getSurfaceCoordinates(groupName=self.DASolver.designSurfacesGroup).size / 3)
            nodes_total = nodes_aero + nodes_prop

            mask = []
            input = []
            promotes_inputs = []

            # Mesh Coordinate Mask
            mask.append(np.zeros([(nodes_total) * 3], dtype=bool))
            mask[0][:] = True
            if nodes_prop > 0:
                mask[0][3 * nodes_aero :] = False
            input.append(
                MaskedVariableDescription(
                    "x_%s0_masked" % self.discipline, shape=(nodes_aero) * 3, tags=["mphys_coordinates"]
                )
            )
            promotes_inputs.append("x_%s0_masked" % self.discipline)

            # Add propeller movement nodes mask if needed
            if aerostructDict["active"] and aerostructDict["propMovement"]:
                # Add Propeller Masks
                if "fvSource" in aerostructDict.keys():
                    i_fvSource = 0
                    i_start = 3 * nodes_aero
                    for fvSource, parameters in aerostructDict["fvSource"].items():
                        mask.append(np.zeros([(nodes_total) * 3], dtype=bool))
                        mask[i_fvSource + 1][:] = False

                        if self.comm.rank == 0:
                            mask[i_fvSource + 1][i_start : i_start + 3 * (1 + parameters["nNodes"])] = True
                            i_start += 3 * (1 + parameters["nNodes"])

                            input.append(
                                MaskedVariableDescription(
                                    "x_prop0_nodes_%s" % fvSource,
                                    shape=((1 + parameters["nNodes"])) * 3,
                                    tags=["mphys_coordinates"],
                                )
                            )
                        else:
                            input.append(
                                MaskedVariableDescription(
                                    "x_prop0_nodes_%s" % fvSource, shape=(0), tags=["mphys_coordinates"]
                                )
                            )
                        promotes_inputs.append("x_prop0_nodes_%s" % fvSource)

                        i_fvSource += 1

            output = MaskedVariableDescription(
                "x_%s0" % self.discipline, shape=(nodes_total) * 3, tags=["mphys_coordinates"]
            )

            unmasker = UnmaskedConverter(input=input, output=output, mask=mask, distributed=True, default_values=0.0)
            self.add_subsystem(
                "unmasker", unmasker, promotes_inputs=promotes_inputs, promotes_outputs=["x_%s0" % self.discipline]
            )

        if self.thermal_coupling:
            self.add_subsystem(
                "%s_xs" % self.discipline,
                DAFoamFaceCoords(solver=self.DASolver, groupName=self.DASolver.couplingSurfacesGroup),
                promotes_inputs=["*"],
                promotes_outputs=["*"],
            )


class DAFoamPostcouplingGroup(Group):
    """
    Post-coupling group that configures any components that happen in the post-processor.
    """

    def initialize(self):
        self.options.declare("solver", default=None, recordable=False)

    def setup(self):
        self.DASolver = self.options["solver"]

        # Add Functionals
        self.add_subsystem("functionals", DAFoamFunctions(solver=self.DASolver), promotes=["*"])

        # Add Acoustics Data
        couplingInfo = self.DASolver.getOption("couplingInfo")
        if couplingInfo["aeroacoustic"]["active"]:
            for groupName in couplingInfo["aeroacoustic"]["couplingSurfaceGroups"]:
                self.add_subsystem(
                    groupName, DAFoamAcoustics(solver=self.DASolver, groupName=groupName), promotes_inputs=["*"]
                )

    def mphys_add_funcs(self):
        self.functionals.mphys_add_funcs()

    def add_dv_func(self, dvName, dv_func):
        self.functionals.add_dv_func(dvName, dv_func)

    def mphys_set_options(self, options):
        self.functionals.mphys_set_options(options)


class DAFoamSolver(ImplicitComponent):
    """
    OpenMDAO component that wraps the DAFoam flow and adjoint solvers
    """

    def initialize(self):
        self.options.declare("solver", recordable=False)
        self.options.declare("prop_coupling", recordable=False)
        self.options.declare("run_directory", default="")

    def setup(self):
        # NOTE: the setup function will be called everytime a new scenario is created.

        self.prop_coupling = self.options["prop_coupling"]

        self.DASolver = self.options["solver"]
        DASolver = self.DASolver

        self.run_directory = self.options["run_directory"]

        self.discipline = self.DASolver.getOption("discipline")

        self.solution_counter = 1

        # by default, we will not have a separate optionDict attached to this
        # solver. But if we do multipoint optimization, we need to use the
        # optionDict for each point because each point may have different
        # objFunc and primalBC options
        self.optionDict = None

        # Initialize the design variable functions, e.g., aoa, actuator
        self.dv_funcs = {}

        # initialize the dRdWT matrix-free matrix in DASolver
        DASolver.solverAD.initializedRdWTMatrixFree(DASolver.xvVec, DASolver.wVec)

        # create the adjoint vector
        self.psi = self.DASolver.wVec.duplicate()
        self.psi.zeroEntries()

        # if true, we need to compute the coloring
        if DASolver.getOption("adjEqnSolMethod") == "fixedPoint":
            self.runColoring = False
        else:
            self.runColoring = True

        # determine which function to compute the adjoint
        self.evalFuncs = []
        DASolver.setEvalFuncs(self.evalFuncs)

        local_state_size = DASolver.getNLocalAdjointStates()

        designVariables = DASolver.getOption("designVar")

        # get the dvType dict
        self.dvType = {}
        for dvName in list(designVariables.keys()):
            self.dvType[dvName] = designVariables[dvName]["designVarType"]

        # setup input and output for the solver
        # we need to add states for all cases
        self.add_output(
            "%s_states" % self.discipline, distributed=True, shape=local_state_size, tags=["mphys_coupling"]
        )

        couplingInfo = DASolver.getOption("couplingInfo")
        if couplingInfo["aerothermal"]["active"]:
            nCells, nFaces = self.DASolver._getSurfaceSize(self.DASolver.couplingSurfacesGroup)
            # NOTE: here we create two duplicated surface center coords, so the size is nFaces * 2
            # one is for transferring near wall temperature, the other is for transferring k/d coefficients
            if self.discipline == "aero":
                self.add_input("T_convect", distributed=True, shape=2 * nFaces, tags=["mphys_coupling"])
            if self.discipline == "thermal":
                self.add_input("q_conduct", distributed=True, shape=2 * nFaces, tags=["mphys_coupling"])

        # now loop over the design variable keys to determine which other variables we need to add
        shapeVarAdded = False
        for dvName in list(designVariables.keys()):
            dvType = self.dvType[dvName]
            if dvType == "FFD":  # add shape variables
                if shapeVarAdded is False:  # we add the shape variable only once
                    # NOTE: for shape variables, we add dafoam_vol_coords as the input name
                    # the specific name for this shape variable will be added in the geometry component (DVGeo)
                    self.add_input(
                        "%s_vol_coords" % self.discipline, distributed=True, shape_by_conn=True, tags=["mphys_coupling"]
                    )
                    shapeVarAdded = True
            elif dvType == "AOA":  # add angle of attack variable
                self.add_input(dvName, distributed=False, shape_by_conn=True, tags=["mphys_coupling"])
            elif dvType == "BC":  # add boundary conditions
                self.add_input(dvName, distributed=False, shape_by_conn=True, tags=["mphys_coupling"])
            elif dvType == "ACTD":  # add actuator parameter variables
                nACTDVars = 10
                if "comps" in list(designVariables[dvName].keys()):
                    nACTDVars = len(designVariables[dvName]["comps"])
                self.add_input(dvName, distributed=False, shape=nACTDVars, tags=["mphys_coupling"])
            elif dvType == "Field":  # add field variables
                self.add_input(dvName, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
            else:
                raise AnalysisError("designVarType %s not supported! " % dvType)

    def add_dv_func(self, dvName, dv_func):
        # add a design variable function to self.dv_func
        # we need to call this function in runScript.py everytime we define a new dv_func, e.g., aoa, actuator
        # no need to call this for the shape variables because they will be handled in the geometry component
        # the dv_func should have two inputs, (dvVal, DASolver)
        if dvName in self.dv_funcs:
            raise AnalysisError("dvName %s is already in self.dv_funcs! " % dvName)
        else:
            self.dv_funcs[dvName] = dv_func

    def set_options(self, optionDict):
        # here optionDict should be a dictionary that has a consistent format
        # with the daOptions defined in the run script
        self.optionDict = optionDict

    def apply_options(self, optionDict):
        if optionDict is not None:
            # This is a multipoint optimization. We need to replace the
            # daOptions with optionDict
            for key in optionDict.keys():
                self.DASolver.setOption(key, optionDict[key])
            self.DASolver.updateDAOption()

    # calculate the residual
    def apply_nonlinear(self, inputs, outputs, residuals):
        DASolver = self.DASolver
        DASolver.setStates(outputs["%s_states" % self.discipline])

        # get flow residuals from DASolver
        residuals["%s_states" % self.discipline] = DASolver.getResiduals()

    # solve the flow
    def solve_nonlinear(self, inputs, outputs):

        with cd(self.run_directory):

            DASolver = self.DASolver

            # set the runStatus, this is useful when the actuator term is activated
            DASolver.setOption("runStatus", "solvePrimal")

            # assign the optionDict to the solver
            self.apply_options(self.optionDict)

            # now call the dv_funcs to update the design variables
            for dvName in self.dv_funcs:
                func = self.dv_funcs[dvName]
                dvVal = inputs[dvName]
                func(dvVal, DASolver)

            DASolver.updateDAOption()

            couplingInfo = DASolver.getOption("couplingInfo")
            if couplingInfo["aerothermal"]["active"]:
                if self.discipline == "aero":
                    T_convect = inputs["T_convect"]
                    DASolver.solver.setThermal(T_convect)
                elif self.discipline == "thermal":
                    q_conduct = inputs["q_conduct"]
                    DASolver.solver.setThermal(q_conduct)
                else:
                    raise AnalysisError("discipline not valid!")

            # solve the flow with the current design variable
            DASolver()

            # get the objective functions
            funcs = {}
            DASolver.evalFunctions(funcs, evalFuncs=self.evalFuncs)

            # assign the computed flow states to outputs
            outputs["%s_states" % self.discipline] = DASolver.getStates()

            # if the primal solution fail, we return analysisError and let the optimizer handle it
            fail = funcs["fail"]
            if fail:
                raise AnalysisError("Primal solution failed!")

    def linearize(self, inputs, outputs, residuals):
        # NOTE: we do not do any computation in this function, just print some information

        self.DASolver.setStates(outputs["%s_states" % self.discipline])

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        # compute the matrix vector products for states and volume mesh coordinates
        # i.e., dRdWT*psi, dRdXvT*psi

        # we do not support forward mode
        if mode == "fwd":
            om.issue_warning(
                " mode = %s, but the forward mode functions are not implemented for DAFoam!" % mode,
                prefix="",
                stacklevel=2,
                category=om.OpenMDAOWarning,
            )
            return

        DASolver = self.DASolver

        # assign the optionDict to the solver
        self.apply_options(self.optionDict)

        # now call the dv_funcs to update the design variables
        for dvName in self.dv_funcs:
            func = self.dv_funcs[dvName]
            dvVal = inputs[dvName]
            func(dvVal, DASolver)

        # assign the states in outputs to the OpenFOAM flow fields
        DASolver.setStates(outputs["%s_states" % self.discipline])

        designVariables = DASolver.getOption("designVar")

        if "%s_states" % self.discipline in d_residuals:

            # get the reverse mode AD seed from d_residuals
            resBar = d_residuals["%s_states" % self.discipline]
            # convert the seed array to Petsc vector
            resBarVec = DASolver.array2Vec(resBar)

            # this computes [dRdW]^T*Psi using reverse mode AD
            if "%s_states" % self.discipline in d_outputs:
                prodVec = DASolver.wVec.duplicate()
                prodVec.zeroEntries()
                DASolver.solverAD.calcdRdWTPsiAD(DASolver.xvVec, DASolver.wVec, resBarVec, prodVec)
                wBar = DASolver.vec2Array(prodVec)
                d_outputs["%s_states" % self.discipline] += wBar

            # loop over all d_inputs keys and compute the matrix-vector products accordingly
            for inputName in list(d_inputs.keys()):
                # this computes [dRdXv]^T*Psi using reverse mode AD
                if inputName == "%s_vol_coords" % self.discipline:
                    prodVec = DASolver.xvVec.duplicate()
                    prodVec.zeroEntries()
                    DASolver.solverAD.calcdRdXvTPsiAD(DASolver.xvVec, DASolver.wVec, resBarVec, prodVec)
                    xVBar = DASolver.vec2Array(prodVec)
                    d_inputs["%s_vol_coords" % self.discipline] += xVBar
                elif inputName == "q_conduct":
                    # calculate [dRdQ]^T*Psi for thermal
                    volCoords = inputs["%s_vol_coords" % self.discipline]
                    states = outputs["%s_states" % self.discipline]
                    thermal = inputs["q_conduct"]
                    product = np.zeros_like(thermal)
                    DASolver.solverAD.calcdRdThermalTPsiAD(volCoords, states, thermal, resBar, product)
                    d_inputs["q_conduct"] += product
                elif inputName == "T_convect":
                    # calculate [dRdT]^T*Psi for aero
                    volCoords = inputs["%s_vol_coords" % self.discipline]
                    states = outputs["%s_states" % self.discipline]
                    thermal = inputs["T_convect"]
                    product = np.zeros_like(thermal)
                    DASolver.solverAD.calcdRdThermalTPsiAD(volCoords, states, thermal, resBar, product)
                    d_inputs["T_convect"] += product
                else:  # now we deal with general input output names
                    # compute [dRdAOA]^T*Psi using reverse mode AD
                    if self.dvType[inputName] == "AOA":
                        prodVec = PETSc.Vec().create(self.comm)
                        prodVec.setSizes((PETSc.DECIDE, 1), bsize=1)
                        prodVec.setFromOptions()
                        DASolver.solverAD.calcdRdAOATPsiAD(
                            DASolver.xvVec, DASolver.wVec, resBarVec, inputName.encode(), prodVec
                        )
                        # The aoaBar variable will be length 1 on the root proc, but length 0 an all slave procs.
                        # The value on the root proc must be broadcast across all procs.
                        if self.comm.rank == 0:
                            aoaBar = DASolver.vec2Array(prodVec)[0]
                        else:
                            aoaBar = 0.0

                        d_inputs[inputName] += self.comm.bcast(aoaBar, root=0)

                    # compute [dRdBC]^T*Psi using reverse mode AD
                    elif self.dvType[inputName] == "BC":
                        prodVec = PETSc.Vec().create(self.comm)
                        prodVec.setSizes((PETSc.DECIDE, 1), bsize=1)
                        prodVec.setFromOptions()
                        DASolver.solverAD.calcdRdBCTPsiAD(
                            DASolver.xvVec, DASolver.wVec, resBarVec, inputName.encode(), prodVec
                        )
                        # The BCBar variable will be length 1 on the root proc, but length 0 an all slave procs.
                        # The value on the root proc must be broadcast across all procs.
                        if self.comm.rank == 0:
                            BCBar = DASolver.vec2Array(prodVec)[0]
                        else:
                            BCBar = 0.0

                        d_inputs[inputName] += self.comm.bcast(BCBar, root=0)

                    # compute [dRdActD]^T*Psi using reverse mode AD
                    elif self.dvType[inputName] == "ACTD":
                        prodVec = PETSc.Vec().create(self.comm)
                        prodVec.setSizes((PETSc.DECIDE, 10), bsize=1)
                        prodVec.setFromOptions()
                        DASolver.solverAD.calcdRdActTPsiAD(
                            DASolver.xvVec, DASolver.wVec, resBarVec, inputName.encode(), prodVec
                        )
                        # we will convert the MPI prodVec to seq array for all procs
                        ACTDBar = DASolver.convertMPIVec2SeqArray(prodVec)
                        if "comps" in list(designVariables[inputName].keys()):
                            nACTDVars = len(designVariables[inputName]["comps"])
                            ACTDBarSub = np.zeros(nACTDVars, "d")
                            for i in range(nACTDVars):
                                comp = designVariables[inputName]["comps"][i]
                                ACTDBarSub[i] = ACTDBar[comp]
                            d_inputs[inputName] += ACTDBarSub
                        else:
                            d_inputs[inputName] += ACTDBar

                    # compute dRdFieldT*Psi using reverse mode AD
                    elif self.dvType[inputName] == "Field":
                        nLocalCells = self.DASolver.solver.getNLocalCells()
                        fieldType = DASolver.getOption("designVar")[inputName]["fieldType"]
                        fieldComp = 1
                        if fieldType == "vector":
                            fieldComp = 3
                        nLocalSize = nLocalCells * fieldComp
                        prodVec = PETSc.Vec().create(self.comm)
                        prodVec.setSizes((nLocalSize, PETSc.DECIDE), bsize=1)
                        prodVec.setFromOptions()
                        DASolver.solverAD.calcdRdFieldTPsiAD(
                            DASolver.xvVec, DASolver.wVec, resBarVec, inputName.encode(), prodVec
                        )
                        fieldBar = DASolver.vec2Array(prodVec)
                        d_inputs[inputName] += fieldBar

                    else:
                        raise AnalysisError("designVarType %s not supported! " % self.dvType[inputName])

    def solve_linear(self, d_outputs, d_residuals, mode):
        # solve the adjoint equation [dRdW]^T * Psi = dFdW

        # we do not support forward mode
        if mode == "fwd":
            om.issue_warning(
                " mode = %s, but the forward mode functions are not implemented for DAFoam!" % mode,
                prefix="",
                stacklevel=2,
                category=om.OpenMDAOWarning,
            )
            return

        with cd(self.run_directory):

            DASolver = self.DASolver

            # set the runStatus, this is useful when the actuator term is activated
            DASolver.setOption("runStatus", "solveAdjoint")
            DASolver.updateDAOption()

            adjEqnSolMethod = DASolver.getOption("adjEqnSolMethod")

            # right hand side array from d_outputs
            dFdWArray = d_outputs["%s_states" % self.discipline]
            # convert the array to vector
            dFdW = DASolver.array2Vec(dFdWArray)

            # run coloring
            if self.DASolver.getOption("adjUseColoring") and self.runColoring:
                self.DASolver.runColoring()
                self.runColoring = False

            if adjEqnSolMethod == "Krylov":
                # solve the adjoint equation using the Krylov method

                # if writeMinorIterations=True, we rename the solution in pyDAFoam.py. So we don't recompute the PC
                if DASolver.getOption("writeMinorIterations"):
                    if DASolver.dRdWTPC is None or DASolver.ksp is None:
                        DASolver.dRdWTPC = PETSc.Mat().create(self.comm)
                        DASolver.solver.calcdRdWT(DASolver.xvVec, DASolver.wVec, 1, DASolver.dRdWTPC)
                        DASolver.ksp = PETSc.KSP().create(self.comm)
                        DASolver.solverAD.createMLRKSPMatrixFree(DASolver.dRdWTPC, DASolver.ksp)
                # otherwise, we need to recompute the PC mat based on adjPCLag
                else:
                    # NOTE: this function will be called multiple times (one time for one obj func) in each opt iteration
                    # so we don't want to print the total info and recompute PC for each obj, we need to use renamed
                    # to check if a recompute is needed. In other words, we only recompute the PC for the first obj func
                    # adjoint solution
                    solutionTime, renamed = DASolver.renameSolution(self.solution_counter)
                    if renamed:
                        # write the deformed FFD for post-processing
                        # DASolver.writeDeformedFFDs(self.solution_counter)
                        # print the solution counter
                        if self.comm.rank == 0:
                            print("Driver total derivatives for iteration: %d" % self.solution_counter)
                            print("---------------------------------------------")
                        self.solution_counter += 1

                    # compute the preconditioner matrix for the adjoint linear equation solution
                    # and initialize the ksp object. We reinitialize them every adjPCLag
                    adjPCLag = DASolver.getOption("adjPCLag")
                    if DASolver.dRdWTPC is None or DASolver.ksp is None or (self.solution_counter - 1) % adjPCLag == 0:
                        if renamed:
                            # calculate the PC mat
                            if DASolver.dRdWTPC is not None:
                                DASolver.dRdWTPC.destroy()
                            DASolver.dRdWTPC = PETSc.Mat().create(self.comm)
                            DASolver.solver.calcdRdWT(DASolver.xvVec, DASolver.wVec, 1, DASolver.dRdWTPC)
                            # reset the KSP
                            if DASolver.ksp is not None:
                                DASolver.ksp.destroy()
                            DASolver.ksp = PETSc.KSP().create(self.comm)
                            DASolver.solverAD.createMLRKSPMatrixFree(DASolver.dRdWTPC, DASolver.ksp)

                # if useNonZeroInitGuess is False, we will manually reset self.psi to zero
                # this is important because we need the correct psi to update the KSP tolerance
                # in the next line
                if not self.DASolver.getOption("adjEqnOption")["useNonZeroInitGuess"]:
                    self.psi.set(0)

                if self.DASolver.getOption("adjEqnOption")["dynAdjustTol"]:
                    # if we want to dynamically adjust the tolerance, call this function. This is mostly used
                    # in the block Gauss-Seidel method in two discipline coupling
                    # update the KSP tolerances the coupled adjoint before solving
                    self._updateKSPTolerances(self.psi, dFdW, DASolver.ksp)

                # actually solving the adjoint linear equation using Petsc
                fail = DASolver.solverAD.solveLinearEqn(DASolver.ksp, dFdW, self.psi)
            elif adjEqnSolMethod == "fixedPoint":
                solutionTime, renamed = DASolver.renameSolution(self.solution_counter)
                if renamed:
                    # write the deformed FFD for post-processing
                    # DASolver.writeDeformedFFDs(self.solution_counter)
                    # print the solution counter
                    if self.comm.rank == 0:
                        print("Driver total derivatives for iteration: %d" % self.solution_counter)
                        print("---------------------------------------------")
                    self.solution_counter += 1
                # solve the adjoint equation using the fixed-point adjoint approach
                fail = DASolver.solverAD.runFPAdj(DASolver.xvVec, DASolver.wVec, dFdW, self.psi)
            else:
                raise RuntimeError("adjEqnSolMethod=%s not valid! Options are: Krylov or fixedPoint" % adjEqnSolMethod)

            # convert the solution vector to array and assign it to d_residuals
            d_residuals["%s_states" % self.discipline] = DASolver.vec2Array(self.psi)

            # if the adjoint solution fail, we return analysisError and let the optimizer handle it
            if fail:
                raise AnalysisError("Adjoint solution failed!")

    def _updateKSPTolerances(self, psi, dFdW, ksp):
        # Here we need to manually update the KSP tolerances because the default
        # relative tolerance will always want to converge the adjoint to a fixed
        # tolerance during the LINGS adjoint solution. However, what we want is
        # to converge just a few orders of magnitude. Here we need to bypass the
        # rTol in Petsc and manually calculate the aTol.

        DASolver = self.DASolver
        # calculate the initial residual for the adjoint before solving
        rVec = self.DASolver.wVec.duplicate()
        rVec.zeroEntries()
        DASolver.solverAD.calcdRdWTPsiAD(DASolver.xvVec, DASolver.wVec, psi, rVec)
        rVec.axpy(-1.0, dFdW)
        rNorm = rVec.norm()
        # read the rTol and aTol from DAOption
        rTol0 = self.DASolver.getOption("adjEqnOption")["gmresRelTol"]
        aTol0 = self.DASolver.getOption("adjEqnOption")["gmresAbsTol"]
        # calculate the new absolute tolerance that gives you rTol residual drop
        aTolNew = rNorm * rTol0
        # if aTolNew is smaller than aTol0, assign aTol0 to aTolNew
        if aTolNew < aTol0:
            aTolNew = aTol0
        # assign the atolNew and distable rTol
        ksp.setTolerances(rtol=0.0, atol=aTolNew, divtol=None, max_it=None)


class DAFoamMeshGroup(Group):
    def initialize(self):
        self.options.declare("solver", recordable=False)

    def setup(self):
        DASolver = self.options["solver"]

        self.discipline = self.DASolver.getOption("discipline")

        self.add_subsystem("surface_mesh", DAFoamMesh(solver=DASolver), promotes=["*"])
        self.add_subsystem(
            "volume_mesh",
            DAFoamWarper(solver=DASolver),
            promotes_inputs=[("x_%s_masked" % self.discipline, "x_%s0" % self.discipline)],
            promotes_outputs=["%s_vol_coords" % self.discipline],
        )

    def mphys_add_coordinate_input(self):
        # just pass through the call
        return self.surface_mesh.mphys_add_coordinate_input()

    def mphys_get_triangulated_surface(self):
        # just pass through the call
        return self.surface_mesh.mphys_get_triangulated_surface()


class DAFoamMesh(ExplicitComponent):
    """
    Component to get the partitioned initial surface mesh coordinates
    """

    def initialize(self):
        self.options.declare("solver", recordable=False)

    def setup(self):

        self.DASolver = self.options["solver"]

        self.discipline = self.DASolver.getOption("discipline")

        # design surface coordinates
        self.x_a0 = self.DASolver.getSurfaceCoordinates(self.DASolver.designSurfacesGroup).flatten(order="C")

        # add output
        coord_size = self.x_a0.size
        self.add_output(
            "x_%s0" % self.discipline,
            distributed=True,
            shape=coord_size,
            desc="initial aerodynamic surface node coordinates",
            tags=["mphys_coordinates"],
        )

    def mphys_add_coordinate_input(self):
        self.add_input(
            "x_%s0_points" % self.discipline,
            distributed=True,
            shape_by_conn=True,
            desc="aerodynamic surface with geom changes",
        )

        # return the promoted name and coordinates
        return "x_%s0_points" % self.discipline, self.x_a0

    def mphys_get_surface_mesh(self):
        return self.x_a0

    def mphys_get_triangulated_surface(self, groupName=None):
        # this is a list of lists of 3 points
        # p0, v1, v2

        return self.DASolver.getTriangulatedMeshSurface()

    def mphys_get_surface_size(self, groupName):
        return self.DASolver._getSurfaceSize(groupName)

    def compute(self, inputs, outputs):
        # just assign the surface mesh coordinates
        if "x_%s0_points" % self.discipline in inputs:
            outputs["x_%s0" % self.discipline] = inputs["x_%s0_points" % self.discipline]
        else:
            outputs["x_%s0" % self.discipline] = self.x_a0

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        # we do not support forward mode AD
        if mode == "fwd":
            om.issue_warning(
                " mode = %s, but the forward mode functions are not implemented for DAFoam!" % mode,
                prefix="",
                stacklevel=2,
                category=om.OpenMDAOWarning,
            )
            return

        # just assign the matrix-vector product
        if "x_%s0_points" % self.discipline in d_inputs:
            d_inputs["x_%s0_points" % self.discipline] += d_outputs["x_%s0" % self.discipline]


class DAFoamFunctions(ExplicitComponent):
    """
    DAFoam objective and constraint functions component
    """

    def initialize(self):
        self.options.declare("solver", recordable=False)

        # a list that contains all function names, e.g., CD, CL
        self.funcs = None

    def setup(self):

        self.DASolver = self.options["solver"]

        self.discipline = self.DASolver.getOption("discipline")

        # init the dv_funcs
        self.dv_funcs = {}

        self.optionDict = None

        # get the dvType dict
        designVariables = self.DASolver.getOption("designVar")
        self.dvType = {}
        for dvName in list(designVariables.keys()):
            self.dvType[dvName] = designVariables[dvName]["designVarType"]

        # setup input and output for the function
        # we need to add states for all cases
        self.add_input("%s_states" % self.discipline, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])

        # now loop over the design variable keys to determine which other variables we need to add
        shapeVarAdded = False
        for dvName in list(designVariables.keys()):
            dvType = self.dvType[dvName]
            if dvType == "FFD":  # add shape variables
                if shapeVarAdded is False:  # we add the shape variable only once
                    # NOTE: for shape variables, we add dafoam_vol_coords as the input name
                    # the specific name for this shape variable will be added in the geometry component (DVGeo)
                    self.add_input(
                        "%s_vol_coords" % self.discipline, distributed=True, shape_by_conn=True, tags=["mphys_coupling"]
                    )
                    shapeVarAdded = True
            elif dvType == "AOA":  # add angle of attack variable
                self.add_input(dvName, distributed=False, shape_by_conn=True, tags=["mphys_coupling"])
            elif dvType == "BC":  # add boundary conditions
                self.add_input(dvName, distributed=False, shape_by_conn=True, tags=["mphys_coupling"])
            elif dvType == "ACTD":  # add actuator parameter variables
                nACTDVars = 10
                if "comps" in list(designVariables[dvName].keys()):
                    nACTDVars = len(designVariables[dvName]["comps"])
                self.add_input(dvName, distributed=False, shape=nACTDVars, tags=["mphys_coupling"])
            elif dvType == "Field":  # add field variables
                self.add_input(dvName, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
            else:
                raise AnalysisError("designVarType %s not supported! " % dvType)

    def mphys_add_funcs(self):
        # add the function names to this component, called from runScript.py

        # it is called objFunc in DAOptions but it contains both objective and constraint functions
        objFuncs = self.DASolver.getOption("objFunc")

        self.funcs = []

        for objFunc in objFuncs:
            self.funcs.append(objFunc)

        # loop over the functions here and create the output
        for f_name in self.funcs:
            self.add_output(f_name, distributed=False, shape=1, units=None, tags=["mphys_result"])

    def add_dv_func(self, dvName, dv_func):
        # add a design variable function to self.dv_func
        # we need to call this function in runScript.py everytime we define a new dv_func, e.g., aoa, actuator
        # no need to call this for the shape variables because they will be handled in the geometry component
        # the dv_func should have two inputs, (dvVal, DASolver)
        if dvName in self.dv_funcs:
            raise AnalysisError("dvName %s is already in self.dv_funcs! " % dvName)
        else:
            self.dv_funcs[dvName] = dv_func

    def mphys_set_options(self, optionDict):
        # here optionDict should be a dictionary that has a consistent format
        # with the daOptions defined in the run script
        self.optionDict = optionDict

    def apply_options(self, optionDict):
        if optionDict is not None:
            # This is a multipoint optimization. We need to replace the
            # daOptions with optionDict
            for key in optionDict.keys():
                self.DASolver.setOption(key, optionDict[key])
            self.DASolver.updateDAOption()

    # get the objective function from DASolver
    def compute(self, inputs, outputs):

        DASolver = self.DASolver

        DASolver.setStates(inputs["%s_states" % self.discipline])

        funcs = {}

        if self.funcs is not None:
            DASolver.evalFunctions(funcs, evalFuncs=self.funcs)
            for f_name in self.funcs:
                if f_name in funcs:
                    outputs[f_name] = funcs[f_name]

    # compute the partial derivatives of functions
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        DASolver = self.DASolver

        # set the runStatus, this is useful when the actuator term is activated
        DASolver.setOption("runStatus", "solveAdjoint")
        DASolver.updateDAOption()

        designVariables = DASolver.getOption("designVar")

        # assign the optionDict to the solver
        self.apply_options(self.optionDict)
        # now call the dv_funcs to update the design variables
        for dvName in self.dv_funcs:
            func = self.dv_funcs[dvName]
            dvVal = inputs[dvName]
            func(dvVal, DASolver)
        DASolver.setStates(inputs["%s_states" % self.discipline])

        # we do not support forward mode AD
        if mode == "fwd":
            om.issue_warning(
                " mode = %s, but the forward mode functions are not implemented for DAFoam!" % mode,
                prefix="",
                stacklevel=2,
                category=om.OpenMDAOWarning,
            )
            return

        funcsBar = {}

        # assign value to funcsBar. NOTE: we only assign seed if d_outputs has
        # non-zero values!
        if self.funcs is None:
            raise AnalysisError("functions not set! Forgot to call mphys_add_funcs?")
        else:
            for func_name in self.funcs:
                if func_name in d_outputs and d_outputs[func_name] != 0.0:
                    funcsBar[func_name] = d_outputs[func_name][0]

        # if self.comm.rank == 0:
        #     print(funcsBar)

        if self.comm.rank == 0:
            print("Computing partials for ", list(funcsBar.keys()))

        # loop over all d_inputs keys and compute the partials accordingly
        for objFuncName in list(funcsBar.keys()):

            fBar = funcsBar[objFuncName]

            for inputName in list(d_inputs.keys()):

                # compute dFdW * fBar
                if inputName == "%s_states" % self.discipline:
                    dFdW = DASolver.wVec.duplicate()
                    dFdW.zeroEntries()
                    DASolver.solverAD.calcdFdWAD(DASolver.xvVec, DASolver.wVec, objFuncName.encode(), dFdW)
                    wBar = DASolver.vec2Array(dFdW)
                    d_inputs["%s_states" % self.discipline] += wBar * fBar

                # compute dFdW * fBar
                elif inputName == "%s_vol_coords" % self.discipline:
                    dFdXv = DASolver.xvVec.duplicate()
                    dFdXv.zeroEntries()
                    DASolver.solverAD.calcdFdXvAD(
                        DASolver.xvVec, DASolver.wVec, objFuncName.encode(), "dummy".encode(), dFdXv
                    )
                    xVBar = DASolver.vec2Array(dFdXv)
                    d_inputs["%s_vol_coords" % self.discipline] += xVBar * fBar

                # now we deal with general input input names
                else:
                    # compute dFdAOA
                    if self.dvType[inputName] == "AOA":
                        dFdAOA = PETSc.Vec().create(self.comm)
                        dFdAOA.setSizes((PETSc.DECIDE, 1), bsize=1)
                        dFdAOA.setFromOptions()
                        DASolver.calcdFdAOAAnalytical(objFuncName, dFdAOA)

                        # The aoaBar variable will be length 1 on the root proc, but length 0 an all slave procs.
                        # The value on the root proc must be broadcast across all procs.
                        if self.comm.rank == 0:
                            aoaBar = DASolver.vec2Array(dFdAOA)[0] * fBar
                        else:
                            aoaBar = 0.0

                        d_inputs[inputName] += self.comm.bcast(aoaBar, root=0)

                    # compute dFdBC
                    elif self.dvType[inputName] == "BC":
                        dFdBC = PETSc.Vec().create(self.comm)
                        dFdBC.setSizes((PETSc.DECIDE, 1), bsize=1)
                        dFdBC.setFromOptions()
                        DASolver.solverAD.calcdFdBCAD(
                            DASolver.xvVec, DASolver.wVec, objFuncName.encode(), inputName.encode(), dFdBC
                        )
                        # The BCBar variable will be length 1 on the root proc, but length 0 an all slave procs.
                        # The value on the root proc must be broadcast across all procs.
                        if self.comm.rank == 0:
                            BCBar = DASolver.vec2Array(dFdBC)[0] * fBar
                        else:
                            BCBar = 0.0

                        d_inputs[inputName] += self.comm.bcast(BCBar, root=0)

                    # compute dFdActD
                    elif self.dvType[inputName] == "ACTD":
                        dFdACTD = PETSc.Vec().create(self.comm)
                        dFdACTD.setSizes((PETSc.DECIDE, 10), bsize=1)
                        dFdACTD.setFromOptions()
                        DASolver.solverAD.calcdFdACTAD(
                            DASolver.xvVec, DASolver.wVec, objFuncName.encode(), inputName.encode(), dFdACTD
                        )
                        # we will convert the MPI dFdACTD to seq array for all procs
                        ACTDBar = DASolver.convertMPIVec2SeqArray(dFdACTD)
                        if "comps" in list(designVariables[inputName].keys()):
                            nACTDVars = len(designVariables[inputName]["comps"])
                            ACTDBarSub = np.zeros(nACTDVars, "d")
                            for i in range(nACTDVars):
                                comp = designVariables[inputName]["comps"][i]
                                ACTDBarSub[i] = ACTDBar[comp]
                            d_inputs[inputName] += ACTDBarSub * fBar
                        else:
                            d_inputs[inputName] += ACTDBar * fBar

                    # compute dFdField
                    elif self.dvType[inputName] == "Field":
                        nLocalCells = self.DASolver.solver.getNLocalCells()
                        fieldType = DASolver.getOption("designVar")[inputName]["fieldType"]
                        fieldComp = 1
                        if fieldType == "vector":
                            fieldComp = 3
                        nLocalSize = nLocalCells * fieldComp
                        dFdField = PETSc.Vec().create(self.comm)
                        dFdField.setSizes((nLocalSize, PETSc.DECIDE), bsize=1)
                        dFdField.setFromOptions()
                        DASolver.solverAD.calcdFdFieldAD(
                            DASolver.xvVec, DASolver.wVec, objFuncName.encode(), inputName.encode(), dFdField
                        )
                        fieldBar = DASolver.vec2Array(dFdField)
                        d_inputs[inputName] += fieldBar * fBar

                    else:
                        raise AnalysisError("designVarType %s not supported! " % self.dvType[inputName])


class DAFoamWarper(ExplicitComponent):
    """
    OpenMDAO component that wraps the warping.
    """

    def initialize(self):
        self.options.declare("solver", recordable=False)

    def setup(self):

        self.DASolver = self.options["solver"]
        DASolver = self.DASolver

        self.discipline = self.DASolver.getOption("discipline")

        # state inputs and outputs
        local_volume_coord_size = DASolver.mesh.getSolverGrid().size

        self.add_input("x_%s" % self.discipline, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_output(
            "%s_vol_coords" % self.discipline, distributed=True, shape=local_volume_coord_size, tags=["mphys_coupling"]
        )

    def compute(self, inputs, outputs):
        # given the new surface mesh coordinates, compute the new volume mesh coordinates
        # the mesh warping will be called in getSolverGrid()
        DASolver = self.DASolver

        x_a = inputs["x_%s" % self.discipline].reshape((-1, 3))
        DASolver.setSurfaceCoordinates(x_a, DASolver.designSurfacesGroup)
        DASolver.mesh.warpMesh()
        solverGrid = DASolver.mesh.getSolverGrid()
        # actually change the mesh in the C++ layer by setting xvVec
        DASolver.xvFlatten2XvVec(solverGrid, DASolver.xvVec)
        outputs["%s_vol_coords" % self.discipline] = solverGrid

    # compute the mesh warping products in IDWarp
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        # we do not support forward mode AD
        if mode == "fwd":
            om.issue_warning(
                " mode = %s, but the forward mode functions are not implemented for DAFoam!" % mode,
                prefix="",
                stacklevel=2,
                category=om.OpenMDAOWarning,
            )
            return

        # compute dXv/dXs such that we can propagate the partials (e.g., dF/dXv) to Xs
        # then the partial will be further propagated to XFFD in pyGeo
        if "%s_vol_coords" % self.discipline in d_outputs:
            if "x_%s" % self.discipline in d_inputs:
                dxV = d_outputs["%s_vol_coords" % self.discipline]
                self.DASolver.mesh.warpDeriv(dxV)
                dxS = self.DASolver.mesh.getdXs()
                dxS = self.DASolver.mapVector(dxS, self.DASolver.allWallsGroup, self.DASolver.designSurfacesGroup)
                d_inputs["x_%s" % self.discipline] += dxS.flatten()


class DAFoamThermal(ExplicitComponent):
    """
    OpenMDAO component that wraps conjugate heat transfer integration

    """

    def initialize(self):
        self.options.declare("solver", recordable=False)

    def setup(self):

        self.DASolver = self.options["solver"]

        self.discipline = self.DASolver.getOption("discipline")

        self.nCouplingFaces = self.DASolver.solver.getNCouplingFaces()

        self.add_input("%s_vol_coords" % self.discipline, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_input("%s_states" % self.discipline, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])

        if self.discipline == "thermal":
            self.add_output("T_conduct", distributed=True, shape=self.nCouplingFaces * 2, tags=["mphys_coupling"])
        elif self.discipline == "aero":
            self.add_output("q_convect", distributed=True, shape=self.nCouplingFaces * 2, tags=["mphys_coupling"])
        else:
            raise AnalysisError("%s not supported! Options are: aero or thermal" % self.discipline)

    def compute(self, inputs, outputs):

        self.DASolver.setStates(inputs["%s_states" % self.discipline])

        vol_coords = inputs["%s_vol_coords" % self.discipline]
        states = inputs["%s_states" % self.discipline]

        thermal = np.zeros(self.nCouplingFaces * 2)

        if self.discipline == "thermal":

            self.DASolver.solver.getThermal(vol_coords, states, thermal)

            outputs["T_conduct"] = thermal

        elif self.discipline == "aero":

            self.DASolver.solver.getThermal(vol_coords, states, thermal)

            outputs["q_convect"] = thermal

        else:
            raise AnalysisError("%s not supported! Options are: aero or thermal" % self.discipline)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        if mode == "fwd":
            om.issue_warning(
                " mode = %s, but the forward mode functions are not implemented for DAFoam!" % mode,
                prefix="",
                stacklevel=2,
                category=om.OpenMDAOWarning,
            )
            return

        DASolver = self.DASolver

        vol_coords = inputs["%s_vol_coords" % self.discipline]
        states = inputs["%s_states" % self.discipline]

        if "T_conduct" in d_outputs:
            seeds = d_outputs["T_conduct"]

            if "%s_states" % self.discipline in d_inputs:
                product = np.zeros_like(d_inputs["%s_states" % self.discipline])
                DASolver.solverAD.getThermalAD("states", vol_coords, states, seeds, product)
                d_inputs["%s_states" % self.discipline] += product

            if "%s_vol_coords" % self.discipline in d_inputs:
                product = np.zeros_like(d_inputs["%s_vol_coords" % self.discipline])
                DASolver.solverAD.getThermalAD("volCoords", vol_coords, states, seeds, product)
                d_inputs["%s_vol_coords" % self.discipline] += product

        if "q_convect" in d_outputs:
            seeds = d_outputs["q_convect"]

            if "%s_states" % self.discipline in d_inputs:
                product = np.zeros_like(d_inputs["%s_states" % self.discipline])
                DASolver.solverAD.getThermalAD("states", vol_coords, states, seeds, product)
                d_inputs["%s_states" % self.discipline] += product

            if "%s_vol_coords" % self.discipline in d_inputs:
                product = np.zeros_like(d_inputs["%s_vol_coords" % self.discipline])
                DASolver.solverAD.getThermalAD("volCoords", vol_coords, states, seeds, product)
                d_inputs["%s_vol_coords" % self.discipline] += product


class DAFoamFaceCoords(ExplicitComponent):
    """
    Calculate coupling surface coordinates based on volume coordinates

    """

    def initialize(self):
        self.options.declare("solver", recordable=False)
        self.options.declare("groupName", recordable=False)

    def setup(self):

        self.DASolver = self.options["solver"]
        self.discipline = self.DASolver.getOption("discipline")
        groupName = self.options["groupName"]

        self.add_input("%s_vol_coords" % self.discipline, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])

        nPts, self.nFaces = self.DASolver._getSurfaceSize(groupName)
        # NOTE: here we create two duplicated surface center coordinates, so the size is nFaces * 6
        # one is for transferring near wall temperature, the other is for transferring k/d coefficients
        self.add_output(
            "x_%s_surface0" % self.discipline, distributed=True, shape=self.nFaces * 6, tags=["mphys_coupling"]
        )

    def compute(self, inputs, outputs):

        volCoords = inputs["%s_vol_coords" % self.discipline]

        nCouplingFaces = self.DASolver.solver.getNCouplingFaces()
        surfCoords = np.zeros(nCouplingFaces * 6)
        self.DASolver.solver.calcCouplingFaceCoords(volCoords, surfCoords)

        outputs["x_%s_surface0" % self.discipline] = surfCoords

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        if mode == "fwd":
            om.issue_warning(
                " mode = %s, but the forward mode functions are not implemented for DAFoam!" % mode,
                prefix="",
                stacklevel=2,
                category=om.OpenMDAOWarning,
            )
            return

        DASolver = self.DASolver

        if "x_%s_surface0" % self.discipline in d_outputs:
            seeds = d_outputs["x_%s_surface0" % self.discipline]

            if "%s_vol_coords" % self.discipline in d_inputs:
                volCoords = inputs["%s_vol_coords" % self.discipline]
                product = np.zeros_like(volCoords)
                DASolver.solverAD.calcCouplingFaceCoordsAD(volCoords, seeds, product)
                d_inputs["%s_vol_coords" % self.discipline] += product


class DAFoamForces(ExplicitComponent):
    """
    OpenMDAO component that wraps force integration

    """

    def initialize(self):
        self.options.declare("solver", recordable=False)

    def setup(self):

        self.DASolver = self.options["solver"]

        self.discipline = self.DASolver.getOption("discipline")

        self.add_input("%s_vol_coords" % self.discipline, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_input("%s_states" % self.discipline, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])

        local_surface_coord_size = self.DASolver.getSurfaceCoordinates(self.DASolver.couplingSurfacesGroup).size
        self.add_output("f_aero", distributed=True, shape=local_surface_coord_size, tags=["mphys_coupling"])

    def compute(self, inputs, outputs):

        self.DASolver.setStates(inputs["%s_states" % self.discipline])

        outputs["f_aero"] = self.DASolver.getForces().flatten(order="C")

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        DASolver = self.DASolver

        if mode == "fwd":
            om.issue_warning(
                " mode = %s, but the forward mode functions are not implemented for DAFoam!" % mode,
                prefix="",
                stacklevel=2,
                category=om.OpenMDAOWarning,
            )
            return

        if "f_aero" in d_outputs:
            fBar = d_outputs["f_aero"]
            fBarVec = DASolver.array2Vec(fBar)

            if "%s_vol_coords" % self.discipline in d_inputs:
                dForcedXv = DASolver.xvVec.duplicate()
                dForcedXv.zeroEntries()
                DASolver.solverAD.calcdForcedXvAD(DASolver.xvVec, DASolver.wVec, fBarVec, dForcedXv)
                xVBar = DASolver.vec2Array(dForcedXv)
                d_inputs["%s_vol_coords" % self.discipline] += xVBar
            if "%s_states" % self.discipline in d_inputs:
                dForcedW = DASolver.wVec.duplicate()
                dForcedW.zeroEntries()
                DASolver.solverAD.calcdForcedWAD(DASolver.xvVec, DASolver.wVec, fBarVec, dForcedW)
                wBar = DASolver.vec2Array(dForcedW)
                d_inputs["%s_states" % self.discipline] += wBar


class DAFoamAcoustics(ExplicitComponent):
    """
    OpenMDAO component that wraps acoustic coupling

    """

    def initialize(self):
        self.options.declare("solver", recordable=False)
        self.options.declare("groupName", recordable=False)

    def setup(self):

        self.DASolver = self.options["solver"]
        self.groupName = self.options["groupName"]

        self.discipline = self.DASolver.getOption("discipline")

        self.add_input("%s_vol_coords" % self.discipline, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_input("%s_states" % self.discipline, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])

        _, nCls = self.DASolver._getSurfaceSize(self.groupName)
        self.add_output("xAcou", distributed=True, shape=nCls * 3)
        self.add_output("nAcou", distributed=True, shape=nCls * 3)
        self.add_output("aAcou", distributed=True, shape=nCls)
        self.add_output("fAcou", distributed=True, shape=nCls * 3)

    def compute(self, inputs, outputs):

        self.DASolver.setStates(inputs["%s_states" % self.discipline])

        positions, normals, areas, forces = self.DASolver.getAcousticData(self.groupName)

        outputs["xAcou"] = positions.flatten(order="C")
        outputs["nAcou"] = normals.flatten(order="C")
        outputs["aAcou"] = areas.flatten(order="C")
        outputs["fAcou"] = forces.flatten(order="C")

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        DASolver = self.DASolver

        if mode == "fwd":
            om.issue_warning(
                " mode = %s, but the forward mode functions are not implemented for DAFoam!" % mode,
                prefix="",
                stacklevel=2,
                category=om.OpenMDAOWarning,
            )
            return

        for varName in ["xAcou", "nAcou", "aAcou", "fAcou"]:
            if varName in d_outputs:
                fBar = d_outputs[varName]
                fBarVec = DASolver.array2Vec(fBar)
                if "%s_vol_coords" % self.discipline in d_inputs:
                    dAcoudXv = DASolver.xvVec.duplicate()
                    dAcoudXv.zeroEntries()
                    DASolver.solverAD.calcdAcousticsdXvAD(
                        DASolver.xvVec, DASolver.wVec, fBarVec, dAcoudXv, varName.encode(), self.groupName.encode()
                    )
                    xVBar = DASolver.vec2Array(dAcoudXv)
                    d_inputs["%s_vol_coords" % self.discipline] += xVBar
                if "%s_states" % self.discipline in d_inputs:
                    dAcoudW = DASolver.wVec.duplicate()
                    dAcoudW.zeroEntries()
                    DASolver.solverAD.calcdAcousticsdWAD(
                        DASolver.xvVec, DASolver.wVec, fBarVec, dAcoudW, varName.encode(), self.groupName.encode()
                    )
                    wBar = DASolver.vec2Array(dAcoudW)
                    d_inputs["%s_states" % self.discipline] += wBar


class DAFoamPropForce(ExplicitComponent):
    """
    DAFoam component that computes the propeller force and radius profile based on the CFD surface states
    """

    def initialize(self):
        self.options.declare("solver", recordable=False)
        self.options.declare("propName", recordable=False)

    def setup(self):

        self.DASolver = self.options["solver"]
        self.propName = self.options["propName"]

        self.discipline = self.DASolver.getOption("discipline")

        # inputs
        self.nForceSections = self.DASolver.getOption("wingProp")[self.propName]["nForceSections"]
        self.add_input("%s_states" % self.discipline, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_input("%s_vol_coords" % self.discipline, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])

        # outputs
        self.add_output("axial_force", distributed=False, shape=self.nForceSections, tags=["mphys_coupling"])
        self.add_output("tangential_force", distributed=False, shape=self.nForceSections, tags=["mphys_coupling"])
        self.add_output("radial_location", distributed=False, shape=self.nForceSections, tags=["mphys_coupling"])
        self.add_output("integral_force", distributed=False, shape=2, tags=["mphys_coupling"])

    def compute(self, inputs, outputs):

        DASolver = self.DASolver

        DASolver.setStates(inputs["%s_states" % self.discipline])

        # initialize output to zeros
        axialForceProfileVec = PETSc.Vec().createSeq(self.nForceSections, bsize=1, comm=PETSc.COMM_SELF)
        axialForceProfileVec.zeroEntries()
        tangentialForceProfileVec = PETSc.Vec().createSeq(self.nForceSections, bsize=1, comm=PETSc.COMM_SELF)
        tangentialForceProfileVec.zeroEntries()
        radialLocationVec = PETSc.Vec().createSeq(self.nForceSections, bsize=1, comm=PETSc.COMM_SELF)
        radialLocationVec.zeroEntries()
        integralForceVec = PETSc.Vec().createSeq(2, bsize=1, comm=PETSc.COMM_SELF)
        integralForceVec.zeroEntries()

        DASolver.solver.calcForceProfile(
            self.propName.encode(),
            axialForceProfileVec,
            tangentialForceProfileVec,
            radialLocationVec,
            integralForceVec,
        )

        outputs["axial_force"] = DASolver.vec2ArraySeq(axialForceProfileVec)
        outputs["tangential_force"] = DASolver.vec2ArraySeq(tangentialForceProfileVec)
        outputs["radial_location"] = DASolver.vec2ArraySeq(radialLocationVec)
        outputs["integral_force"] = DASolver.vec2ArraySeq(integralForceVec)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        DASolver = self.DASolver

        if mode == "fwd":
            om.issue_warning(
                " mode = %s, but the forward mode functions are not implemented for DAFoam!" % mode,
                prefix="",
                stacklevel=2,
                category=om.OpenMDAOWarning,
            )
            return

        dafoam_states = inputs["%s_states" % self.discipline]
        dafoam_xv = inputs["%s_vol_coords" % self.discipline]

        stateVec = DASolver.array2Vec(dafoam_states)
        xvVec = DASolver.array2Vec(dafoam_xv)

        if "axial_force" in d_outputs:
            aFBar = d_outputs["axial_force"]
            aFBarVec = DASolver.array2VecSeq(aFBar)

            if "%s_states" % self.discipline in d_inputs:
                prodVec = stateVec.duplicate()
                prodVec.zeroEntries()
                DASolver.solverAD.calcdForceProfiledXvWAD(
                    self.propName.encode(),
                    "aForce".encode(),
                    "state".encode(),
                    xvVec,
                    stateVec,
                    aFBarVec,
                    prodVec,
                )
                sBar = DASolver.vec2Array(prodVec)
                # sBar = self.comm.allreduce(sBar, op=MPI.SUM)
                d_inputs["%s_states" % self.discipline] += sBar

            if "%s_vol_coords" % self.discipline in d_inputs:
                prodVec = xvVec.duplicate()
                prodVec.zeroEntries()
                DASolver.solverAD.calcdForceProfiledXvWAD(
                    self.propName.encode(),
                    "aForce".encode(),
                    "mesh".encode(),
                    xvVec,
                    stateVec,
                    aFBarVec,
                    prodVec,
                )
                vBar = DASolver.vec2Array(prodVec)
                # vBar = self.comm.allreduce(vBar, op=MPI.SUM)
                d_inputs["%s_vol_coords" % self.discipline] += vBar

        if "tangential_force" in d_outputs:
            tFBar = d_outputs["tangential_force"]
            tFBarVec = DASolver.array2VecSeq(tFBar)

            if "%s_states" % self.discipline in d_inputs:
                prodVec = stateVec.duplicate()
                prodVec.zeroEntries()
                DASolver.solverAD.calcdForceProfiledXvWAD(
                    self.propName.encode(),
                    "tForce".encode(),
                    "state".encode(),
                    xvVec,
                    stateVec,
                    tFBarVec,
                    prodVec,
                )
                sBar = DASolver.vec2Array(prodVec)
                # sBar = self.comm.allreduce(sBar, op=MPI.SUM)
                d_inputs["%s_states" % self.discipline] += sBar

            if "%s_vol_coords" % self.discipline in d_inputs:
                prodVec = xvVec.duplicate()
                prodVec.zeroEntries()
                DASolver.solverAD.calcdForceProfiledXvWAD(
                    self.propName.encode(),
                    "tForce".encode(),
                    "mesh".encode(),
                    xvVec,
                    stateVec,
                    tFBarVec,
                    prodVec,
                )
                vBar = DASolver.vec2Array(prodVec)
                # vBar = self.comm.allreduce(vBar, op=MPI.SUM)
                d_inputs["%s_vol_coords" % self.discipline] += vBar

        if "radial_location" in d_inputs:
            rDBar = d_outputs["radial_location"]
            rDBarVec = DASolver.array2VecSeq(rDBar)
            if "%s_states" % self.discipline in d_inputs:
                prodVec = stateVec.duplicate()
                prodVec.zeroEntries()
                DASolver.solverAD.calcdForceProfiledXvWAD(
                    self.propName.encode(),
                    "rDist".encode(),
                    "state".encode(),
                    xvVec,
                    stateVec,
                    rDBarVec,
                    prodVec,
                )
                sBar = DASolver.vec2Array(prodVec)
                # sBar = self.comm.allreduce(sBar, op=MPI.SUM)
                d_inputs["%s_states" % self.discipline] += sBar

            if "%s_vol_coords" % self.discipline in d_inputs:
                prodVec = xvVec.duplicate()
                prodVec.zeroEntries()
                DASolver.solverAD.calcdForceProfiledXvWAD(
                    self.propName.encode(),
                    "rDist".encode(),
                    "mesh".encode(),
                    xvVec,
                    stateVec,
                    rDBarVec,
                    prodVec,
                )
                vBar = DASolver.vec2Array(prodVec)
                # vBar = self.comm.allreduce(vBar, op=MPI.SUM)
                d_inputs["%s_vol_coords" % self.discipline] += vBar

        if "integral_force" in d_outputs:
            iFBar = d_outputs["integral_force"]
            iFBarVec = DASolver.array2VecSeq(iFBar)

            if "%s_states" % self.discipline in d_inputs:
                prodVec = stateVec.duplicate()
                prodVec.zeroEntries()
                DASolver.solverAD.calcdForceProfiledXvWAD(
                    self.propName.encode(),
                    "integralForce".encode(),
                    "state".encode(),
                    xvVec,
                    stateVec,
                    iFBarVec,
                    prodVec,
                )
                sBar = DASolver.vec2Array(prodVec)
                # sBar = self.comm.allreduce(sBar, op=MPI.SUM)
                d_inputs["%s_states" % self.discipline] += sBar

            if "%s_vol_coords" % self.discipline in d_inputs:
                prodVec = xvVec.duplicate()
                prodVec.zeroEntries()
                DASolver.solverAD.calcdForceProfiledXvWAD(
                    self.propName.encode(),
                    "integralForce".encode(),
                    "mesh".encode(),
                    xvVec,
                    stateVec,
                    iFBarVec,
                    prodVec,
                )
                vBar = DASolver.vec2Array(prodVec)
                # vBar = self.comm.allreduce(vBar, op=MPI.SUM)
                d_inputs["%s_vol_coords" % self.discipline] += vBar

class DAFoamFvSource(ExplicitComponent):
    """
    DAFoam component that computes the actuator source term based on force and radius profiles and prop center
    """

    def initialize(self):
        self.options.declare("solver", recordable=False)

    def setup(self):

        self.DASolver = self.options["solver"]

        # loop over all the propNames and check if any of them is active. If yes, add inputs for this prop
        for propName in list(self.DASolver.getOption("wingProp").keys()):
            if self.DASolver.getOption("wingProp")[propName]["active"]:
                self.nForceSections = self.DASolver.getOption("wingProp")[propName]["nForceSections"]
                self.add_input(
                    propName + "_axial_force", distributed=False, shape=self.nForceSections, tags=["mphys_coupling"]
                )
                self.add_input(
                    propName + "_tangential_force",
                    distributed=False,
                    shape=self.nForceSections,
                    tags=["mphys_coupling"],
                )
                self.add_input(
                    propName + "_radial_location",
                    distributed=False,
                    shape=self.nForceSections,
                    tags=["mphys_coupling"],
                )
                self.add_input(propName + "_integral_force", distributed=False, shape=2, tags=["mphys_coupling"])
                self.add_input(propName + "_prop_center", distributed=False, shape=3, tags=["mphys_coupling"])

        # we have only one output
        self.nLocalCells = self.DASolver.solver.getNLocalCells()
        self.add_output("fvSource", distributed=True, shape=self.nLocalCells * 3, tags=["mphys_coupling"])

    def compute(self, inputs, outputs):

        DASolver = self.DASolver

        # initialize output to zeros
        fvSourceVec = PETSc.Vec().create(self.comm)
        fvSourceVec.setSizes((self.nLocalCells * 3, PETSc.DECIDE), bsize=1)
        fvSourceVec.setFromOptions()
        fvSourceVec.zeroEntries()

        outputs["fvSource"] = DASolver.vec2Array(fvSourceVec)

        # we call calcFvSource multiple times and add contributions from all the propellers
        for propName in list(self.DASolver.getOption("wingProp").keys()):
            if self.DASolver.getOption("wingProp")[propName]["active"]:

                axial_force = inputs[propName + "_axial_force"]
                tangential_force = inputs[propName + "_tangential_force"]
                radial_location = inputs[propName + "_radial_location"]
                integral_force = inputs[propName + "_integral_force"]
                prop_center = inputs[propName + "_prop_center"]

                axial_force_vec = DASolver.array2VecSeq(axial_force)
                tangential_force_vec = DASolver.array2VecSeq(tangential_force)
                radial_location_vec = DASolver.array2VecSeq(radial_location)
                integral_force_vec = DASolver.array2VecSeq(integral_force)
                prop_center_vec = DASolver.array2VecSeq(prop_center)

                fvSourceVec.zeroEntries()

                DASolver.solver.calcFvSource(
                    propName.encode(),
                    axial_force_vec,
                    tangential_force_vec,
                    radial_location_vec,
                    integral_force_vec,
                    prop_center_vec,
                    fvSourceVec,
                )

                outputs["fvSource"] += DASolver.vec2Array(fvSourceVec)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        DASolver = self.DASolver

        if mode == "fwd":
            om.issue_warning(
                " mode = %s, but the forward mode functions are not implemented for DAFoam!" % mode,
                prefix="",
                stacklevel=2,
                category=om.OpenMDAOWarning,
            )
            return

        if "fvSource" in d_outputs:
            sBar = d_outputs["fvSource"]
            sBarVec = DASolver.array2Vec(sBar)

            for propName in list(self.DASolver.getOption("wingProp").keys()):
                if self.DASolver.getOption("wingProp")[propName]["active"]:

                    a = inputs[propName + "_axial_force"]
                    t = inputs[propName + "_tangential_force"]
                    r = inputs[propName + "_radial_location"]
                    f = inputs[propName + "_integral_force"]
                    c = inputs[propName + "_prop_center"]

                    aVec = DASolver.array2VecSeq(a)
                    tVec = DASolver.array2VecSeq(t)
                    rVec = DASolver.array2VecSeq(r)
                    fVec = DASolver.array2VecSeq(f)
                    cVec = DASolver.array2VecSeq(c)

                    if propName + "_axial_force" in d_inputs:
                        prodVec = PETSc.Vec().createSeq(self.nForceSections, bsize=1, comm=PETSc.COMM_SELF)
                        prodVec.zeroEntries()
                        DASolver.solverAD.calcdFvSourcedInputsTPsiAD(
                            propName.encode(), "aForce".encode(), aVec, tVec, rVec, fVec, cVec, sBarVec, prodVec
                        )
                        aBar = DASolver.vec2ArraySeq(prodVec)
                        # aBar = self.comm.allreduce(aBar, op=MPI.SUM)
                        d_inputs[propName + "_axial_force"] += aBar

                    if propName + "_tangential_force" in d_inputs:
                        prodVec = PETSc.Vec().createSeq(self.nForceSections, bsize=1, comm=PETSc.COMM_SELF)
                        prodVec.zeroEntries()
                        DASolver.solverAD.calcdFvSourcedInputsTPsiAD(
                            propName.encode(), "tForce".encode(), aVec, tVec, rVec, fVec, cVec, sBarVec, prodVec
                        )
                        tBar = DASolver.vec2ArraySeq(prodVec)
                        # tBar = self.comm.allreduce(tBar, op=MPI.SUM)
                        d_inputs[propName + "_tangential_force"] += tBar

                    if propName + "_radial_location" in d_inputs:
                        prodVec = PETSc.Vec().createSeq(self.nForceSections, bsize=1, comm=PETSc.COMM_SELF)
                        prodVec.zeroEntries()
                        DASolver.solverAD.calcdFvSourcedInputsTPsiAD(
                            propName.encode(), "rDist".encode(), aVec, tVec, rVec, fVec, cVec, sBarVec, prodVec
                        )
                        rBar = DASolver.vec2ArraySeq(prodVec)
                        # rBar = self.comm.allreduce(rBar, op=MPI.SUM)
                        d_inputs[propName + "_radial_location"] += rBar

                    if propName + "_integral_force" in d_inputs:
                        prodVec = PETSc.Vec().createSeq(2, bsize=1, comm=PETSc.COMM_SELF)
                        prodVec.zeroEntries()
                        DASolver.solverAD.calcdFvSourcedInputsTPsiAD(
                            propName.encode(), "targetForce".encode(), aVec, tVec, rVec, fVec, cVec, sBarVec, prodVec
                        )
                        fBar = DASolver.vec2ArraySeq(prodVec)
                        # fBar = self.comm.allreduce(fBar, op=MPI.SUM)
                        d_inputs[propName + "_integral_force"] += fBar

                    if propName + "_prop_center" in d_inputs:
                        prodVec = PETSc.Vec().createSeq(3, bsize=1, comm=PETSc.COMM_SELF)
                        prodVec.zeroEntries()
                        DASolver.solverAD.calcdFvSourcedInputsTPsiAD(
                            propName.encode(), "center".encode(), aVec, tVec, rVec, fVec, cVec, sBarVec, prodVec
                        )
                        cBar = DASolver.vec2ArraySeq(prodVec)
                        # cBar = self.comm.allreduce(cBar, op=MPI.SUM)
                        d_inputs[propName + "_prop_center"] += cBar


class DAFoamPropNodes(ExplicitComponent):
    """
    Component that computes propeller aero-node locations that link with structural nodes in aerostructural cases.
    """

    def initialize(self):
        self.options.declare("solver", default=None, recordable=False)

    def setup(self):
        self.DASolver = self.options["solver"]

        self.aerostructDict = self.DASolver.getOption("couplingInfo")["aerostructural"]
        self.fvSourceDict = self.DASolver.getOption("fvSource")

        if "fvSource" in self.aerostructDict.keys():
            # Iterate through Actuator Disks
            for fvSource, parameters in self.aerostructDict["fvSource"].items():
                # Check if Actuator Disk Exists
                if fvSource not in self.fvSourceDict:
                    raise RuntimeWarning("Actuator disk %s not found when adding masked nodes" % fvSource)

                # Add Input
                self.add_input("x_prop0_%s" % fvSource, shape=3, distributed=False, tags=["mphys_coordinates"])

                # Add Output
                if self.comm.rank == 0:
                    self.add_output(
                        "x_prop0_nodes_%s" % fvSource,
                        shape=(1 + parameters["nNodes"]) * 3,
                        distributed=True,
                        tags=["mphys_coordinates"],
                    )
                    self.add_output(
                        "f_prop_%s" % fvSource,
                        shape=(1 + parameters["nNodes"]) * 3,
                        distributed=True,
                        tags=["mphys_coordinates"],
                    )
                else:
                    self.add_output(
                        "x_prop0_nodes_%s" % fvSource, shape=(0), distributed=True, tags=["mphys_coordinates"]
                    )
                    self.add_output("f_prop_%s" % fvSource, shape=(0), distributed=True, tags=["mphys_coordinates"])

    def compute(self, inputs, outputs):
        # Loop over all actuator disks to generate ring of nodes for each
        for fvSource, parameters in self.aerostructDict["fvSource"].items():
            # Nodes should only be on root proc
            if self.comm.rank == 0:
                center = inputs["x_prop0_%s" % fvSource]

                # Compute local coordinate frame for ring of nodes
                direction = self.fvSourceDict[fvSource]["direction"]
                direction = direction / np.linalg.norm(direction, 2)
                temp_vec = np.array([1.0, 0.0, 0.0])
                y_local = np.cross(direction, temp_vec)
                if np.linalg.norm(y_local, 2) < 1e-5:
                    temp_vec = np.array([0.0, 1.0, 0.0])
                    y_local = np.cross(direction, temp_vec)
                y_local = y_local / np.linalg.norm(y_local, 2)
                z_local = np.cross(direction, y_local)
                z_local = z_local / np.linalg.norm(z_local, 2)

                n_theta = parameters["nNodes"]
                radial_loc = parameters["radialLoc"]

                # Set ring of nodes location and force values
                nodes_x = np.zeros((n_theta + 1, 3))
                nodes_x[0, :] = center
                nodes_f = np.zeros((n_theta + 1, 3))
                if n_theta == 0:
                    nodes_f[0, :] = -self.fvSourceDict[fvSource]["targetThrust"] * direction
                else:
                    nodes_f[0, :] = 0.0
                    for i in range(n_theta):
                        theta = i / n_theta * 2 * np.pi
                        nodes_x[i + 1, :] = (
                            center + radial_loc * y_local * np.cos(theta) + radial_loc * z_local * np.sin(theta)
                        )
                        nodes_f[i + 1, :] = -self.fvSourceDict[fvSource]["targetThrust"] * direction / n_theta

                outputs["x_prop0_nodes_%s" % fvSource] = nodes_x.flatten()
                outputs["f_prop_%s" % fvSource] = nodes_f.flatten()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            om.issue_warning(
                " mode = %s, but the forward mode functions are not implemented for DAFoam!" % mode,
                prefix="",
                stacklevel=2,
                category=om.OpenMDAOWarning,
            )
            return

        for fvSource, parameters in self.aerostructDict["fvSource"].items():
            if "x_prop0_%s" % fvSource in d_inputs:
                if "x_prop0_nodes_%s" % fvSource in d_outputs:
                    temp = np.zeros((parameters["nNodes"] + 1) * 3)
                    # Take ring of node seeds, broadcast them, and add them to all procs
                    if self.comm.rank == 0:
                        temp[:] = d_outputs["x_prop0_nodes_%s" % fvSource]
                    self.comm.Bcast(temp, root=0)
                    for i in range(parameters["nNodes"]):
                        d_inputs["x_prop0_%s" % fvSource] += temp[3 * i : 3 * i + 3]


class DAFoamActuator(ExplicitComponent):
    """
    Component that updates actuator disk definition variables when actuator disks are displaced in an aerostructural case.
    """

    def initialize(self):
        self.options.declare("solver", recordable=False)

    def setup(self):
        self.DASolver = self.options["solver"]

        self.aerostructDict = self.DASolver.getOption("couplingInfo")["aerostructural"]
        self.fvSourceDict = self.DASolver.getOption("fvSource")

        for fvSource, _ in self.aerostructDict["fvSource"].items():
            self.add_input("dv_actuator_%s" % fvSource, shape=(7), distributed=False, tags=["mphys_coupling"])
            self.add_input("x_prop_%s" % fvSource, shape_by_conn=True, distributed=True, tags=["mphys_coupling"])

            self.add_output("actuator_%s" % fvSource, shape_by_conn=(10), distributed=False, tags=["mphys_coupling"])

    def compute(self, inputs, outputs):
        # Loop over all actuator disks
        for fvSource, _ in self.aerostructDict["fvSource"].items():
            actuator = np.zeros(10)
            # Update variables on root proc
            if self.comm.rank == 0:
                actuator[3:] = inputs["dv_actuator_%s" % fvSource][:]
                actuator[:3] = inputs["x_prop_%s" % fvSource][:3]

            # Broadcast variables to all procs and set as output
            self.comm.Bcast(actuator, root=0)
            outputs["actuator_%s" % fvSource] = actuator

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            om.issue_warning(
                " mode = %s, but the forward mode functions are not implemented for DAFoam!" % mode,
                prefix="",
                stacklevel=2,
                category=om.OpenMDAOWarning,
            )
            return

        # Loop over all actuator disks
        for fvSource, _ in self.aerostructDict["fvSource"].items():
            if "actuator_%s" % fvSource in d_outputs:
                if "dv_actuator_%s" % fvSource in d_inputs:
                    # Add non-location seeds to all procs
                    d_inputs["dv_actuator_%s" % fvSource][:] += d_outputs["actuator_%s" % fvSource][3:]
                if "x_prop_%s" % fvSource in d_inputs:
                    # Add location seeds to only root proc
                    if self.comm.rank == 0:
                        d_inputs["x_prop_%s" % fvSource][:3] += d_outputs["actuator_%s" % fvSource][:3]


class OptFuncs(object):
    """
    Some utility functions
    """

    def __init__(self, daOptions, om_prob):
        """
        daOptions: dict or list
            The daOptions dict from runScript.py. Support more than two dicts

        om_prob:
            The om.Problem() object
        """

        self.daOptions = daOptions
        self.om_prob = om_prob
        self.comm = MPI.COMM_WORLD

        # we need to check if the design variable set in the OM model is also set
        # in the designVar key in DAOptions. If they are not consistent, we print
        # an error and exit because it will produce wrong gradient

        modelDesignVars = self.om_prob.model.get_design_vars()

        isList = isinstance(self.daOptions, list)
        if isList:
            DADesignVars = []
            for subDict in self.daOptions:
                for key in list(subDict["designVar"].keys()):
                    DADesignVars.append(key)
        else:
            DADesignVars = list(self.daOptions["designVar"].keys())
        for modelDV in modelDesignVars:
            dvFound = False
            for dv in DADesignVars:
                # NOTE. modelDV has format like dvs.shape, so we have to use "in" to check
                if dv in modelDV:
                    dvFound = True
            if dvFound is not True:
                raise RuntimeError(
                    "Design variable %s is defined in the model but not found in designVar in DAOptions! " % modelDV
                )

    def findFeasibleDesign(
        self,
        constraints,
        designVars,
        targets,
        constraintsComp=None,
        designVarsComp=None,
        epsFD=None,
        maxIter=10,
        tol=1e-4,
        maxNewtonStep=None,
    ):
        """
        Find the design variables that meet the prescribed constraints. This can be used to get a
        feasible design to start the optimization. For example, finding the angle of attack and
        tail rotation angle that give the target lift and pitching moment. The sizes of cons and
        designvars have to be the same.
        NOTE: we use the Newton method to find the feasible design.
        """

        if self.comm.rank == 0:
            print("Finding a feasible design using the Newton method. ")
            print("Constraints: ", constraints)
            print("Design Vars: ", designVars)
            print("Target: ", targets)

        if len(constraints) != len(designVars):
            raise RuntimeError("Sizes of the constraints and designVars lists need to be the same! ")

        size = len(constraints)

        # if the component is empty, set it to 0
        if constraintsComp is None:
            constraintsComp = size * [0]
        if designVarsComp is None:
            designVarsComp = size * [0]
        # if the FD step size is None, set it to 1e-3
        if epsFD is None:
            epsFD = size * [1e-3]
        # if the max Newton step is None, set it to a very large value
        if maxNewtonStep is None:
            maxNewtonStep = size * [1e16]

        # main Newton loop
        for n in range(maxIter):

            # Newton Jacobian
            jacMat = np.zeros((size, size))

            # run the primal for the reference dvs
            self.om_prob.run_model()

            # get the reference design vars and constraints values
            dv0 = np.zeros(size)
            for i in range(size):
                dvName = designVars[i]
                comp = designVarsComp[i]
                val = self.om_prob.get_val(dvName)
                dv0[i] = val[comp]
            con0 = np.zeros(size)
            for i in range(size):
                conName = constraints[i]
                comp = constraintsComp[i]
                val = self.om_prob.get_val(conName)
                con0[i] = val[comp]

            # calculate the residual. Constraints - Targets
            res = con0 - targets

            # compute the residual norm
            norm = np.linalg.norm(res / targets)

            if self.comm.rank == 0:
                print("FindFeasibleDesign Iter: ", n)
                print("DesignVars: ", dv0)
                print("Constraints: ", con0)
                print("Residual Norm: ", norm)

            # break the loop if residual is already smaller than the tolerance
            if norm < tol:
                if self.comm.rank == 0:
                    print("FindFeasibleDesign Converged! ")
                break

            # perturb design variables and compute the Jacobian matrix
            for i in range(size):
                dvName = designVars[i]
                comp = designVarsComp[i]
                # perturb  +step
                dvP = dv0[i] + epsFD[i]
                self.om_prob.set_val(dvName, dvP, indices=comp)
                # run the primal
                self.om_prob.run_model()
                # reset the perturbation
                self.om_prob.set_val(dvName, dv0[i], indices=comp)

                # get the perturb constraints and compute the Jacobian
                for j in range(size):
                    conName = constraints[j]
                    comp = constraintsComp[j]
                    val = self.om_prob.get_val(conName)
                    conP = val[comp]

                    deriv = (conP - con0[j]) / epsFD[i]
                    jacMat[j][i] = deriv

            # calculate the deltaDV using the Newton method
            deltaDV = -np.linalg.inv(jacMat).dot(res)

            # we can bound the delta change to ensure a more robust Newton solver.
            for i in range(size):
                if abs(deltaDV[i]) > abs(maxNewtonStep[i]):
                    if deltaDV[i] > 0:
                        deltaDV[i] = abs(maxNewtonStep[i])
                    else:
                        deltaDV[i] = -abs(maxNewtonStep[i])

            # update the dv
            dv1 = dv0 + deltaDV
            for i in range(size):
                dvName = designVars[i]
                comp = designVarsComp[i]
                self.om_prob.set_val(dvName, dv1[i], indices=comp)
