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
            groupName = self.DASolver.designSurfacesGroup
        nodes = int(self.DASolver.getSurfaceCoordinates(groupName=groupName).size / 3)

        return nodes

    def get_ndof(self):
        # The number of degrees of freedom used at each output location.
        return -1


class DAFoamGroup(Group):
    """
    DAFoam solver group
    """

    def initialize(self):
        self.options.declare("solver", recordable=False)
        self.options.declare("struct_coupling", default=False)
        self.options.declare("use_warper", default=True)
        self.options.declare("thermal_coupling", default=False)
        self.options.declare("run_directory", default="")

    def setup(self):

        self.DASolver = self.options["solver"]
        self.struct_coupling = self.options["struct_coupling"]
        self.use_warper = self.options["use_warper"]
        self.thermal_coupling = self.options["thermal_coupling"]
        self.run_directory = self.options["run_directory"]
        self.discipline = self.DASolver.getOption("discipline")

        if self.use_warper:

            # if we dont have geo_disp, we also need to promote the x_a as x_a0 from the deformer component
            self.add_subsystem(
                "deformer",
                DAFoamWarper(
                    solver=self.DASolver,
                ),
                promotes_inputs=["x_%s" % self.discipline],
                promotes_outputs=["%s_vol_coords" % self.discipline],
            )

        # add the solver implicit component
        self.add_subsystem(
            "solver",
            DAFoamSolver(solver=self.DASolver, run_directory=self.run_directory),
            promotes_inputs=["*"],
            promotes_outputs=["%s_states" % self.discipline],
        )

        if self.struct_coupling:
            self.add_subsystem(
                "force",
                DAFoamForces(solver=self.DASolver),
                promotes_inputs=["%s_vol_coords" % self.discipline, "%s_states" % self.discipline],
                promotes_outputs=["f_aero"],
            )

        if self.thermal_coupling:
            self.add_subsystem(
                "get_%s" % self.discipline,
                DAFoamThermal(solver=self.DASolver),
                promotes_inputs=["*"],
                promotes_outputs=["*"],
            )


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

        # Return the warper only if it is not in the solver
        if not self.warp_in_solver:
            self.add_subsystem(
                "warper",
                DAFoamWarper(solver=self.DASolver),
                promotes_inputs=["x_%s" % self.discipline],
                promotes_outputs=["%s_vol_coords" % self.discipline],
            )

        if self.thermal_coupling:
            self.add_subsystem(
                "%s_xs" % self.discipline,
                DAFoamFaceCoords(solver=self.DASolver),
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


class DAFoamSolver(ImplicitComponent):
    """
    OpenMDAO component that wraps the DAFoam flow and adjoint solvers
    """

    def initialize(self):
        self.options.declare("solver", recordable=False)
        self.options.declare("run_directory", default="")

    def setup(self):
        # NOTE: the setup function will be called everytime a new scenario is created.

        self.DASolver = self.options["solver"]
        DASolver = self.DASolver

        self.run_directory = self.options["run_directory"]

        self.discipline = self.DASolver.getOption("discipline")

        self.solution_counter = 1

        # pointer to the DVGeo object
        self.DVGeo = None

        # pointer to the DVCon object
        self.DVCon = None

        # setup some names
        self.stateName = "%s_states" % self.discipline
        self.residualName = "%s_residuals" % self.discipline
        self.volCoordName = "%s_vol_coords" % self.discipline

        # initialize the dRdWT matrix-free matrix in DASolver
        DASolver.solverAD.initializedRdWTMatrixFree()

        # create the adjoint vector
        self.localAdjSize = DASolver.getNLocalAdjointStates()
        self.psi = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        self.psi.setSizes((self.localAdjSize, PETSc.DECIDE), bsize=1)
        self.psi.setFromOptions()
        self.psi.zeroEntries()

        # if true, we need to compute the coloring
        if DASolver.getOption("adjEqnSolMethod") == "fixedPoint":
            self.runColoring = False
        else:
            self.runColoring = True

        # setup input and output for the solver
        # we need to add states as outputs for all cases
        local_state_size = DASolver.getNLocalAdjointStates()
        self.add_output(self.stateName, distributed=True, shape=local_state_size, tags=["mphys_coupling"])

        # now loop over the solver input keys to determine which other variables we need to add as inputs
        inputDict = DASolver.getOption("inputInfo")
        for inputName in list(inputDict.keys()):
            # this input is attached to solver comp
            if "solver" in inputDict[inputName]["components"]:
                inputType = inputDict[inputName]["type"]
                inputSize = DASolver.solver.getInputSize(inputName, inputType)
                inputDistributed = DASolver.solver.getInputDistributed(inputName, inputType)
                self.add_input(inputName, distributed=inputDistributed, shape=inputSize, tags=["mphys_coupling"])

    def add_dvgeo(self, DVGeo):
        self.DVGeo = DVGeo

    def add_dvcon(self, DVCon):
        self.DVCon = DVCon

    # calculate the residual
    # def apply_nonlinear(self, inputs, outputs, residuals):
    #     DASolver = self.DASolver
    #     # NOTE: we do not pass the states from inputs to the OF layer.
    #     # this can cause potential convergence issue because the initial states
    #     # in the inputs are set to all ones. So passing this all-ones states
    #     # into the OF layer may diverge the primal solver. Here we can always
    #     # use the states from the OF layer to compute the residuals.
    #     # DASolver.setStates(outputs["%s_states" % self.discipline])
    #     # get flow residuals from DASolver
    #     residuals[self.stateName] = DASolver.getResiduals()

    # solve the flow
    def solve_nonlinear(self, inputs, outputs):

        with cd(self.run_directory):

            DASolver = self.DASolver

            # set the solver input, including mesh, boundary etc.
            DASolver.set_solver_input(inputs, self.DVGeo)

            # before running the primal, we need to check if the mesh
            # quality is good
            meshOK = DASolver.solver.checkMesh()

            # if the mesh is not OK, do not run the primal
            if meshOK != 1:
                DASolver.solver.writeFailedMesh()
                raise AnalysisError("Mesh quality error!")
                return

            # call the primal
            DASolver()

            # if the primal fails, do not set states and return
            if DASolver.primalFail != 0:
                raise AnalysisError("Primal solution failed!")
                return

            # we can use step-average state variables, this can be useful when the
            # primal has LCO
            if DASolver.getOption("useMeanStates"):
                DASolver.solver.meanStatesToStates()

            # after solving the primal, we need to print its residual info
            if DASolver.getOption("useAD")["mode"] == "forward":
                DASolver.solverAD.calcPrimalResidualStatistics("print")
            else:
                DASolver.solver.calcPrimalResidualStatistics("print")

            # assign the computed flow states to outputs
            states = DASolver.getStates()
            outputs[self.stateName] = states

            # set states
            DASolver.setStates(states)

            # We also need to just calculate the residual for the AD mode to initialize vars like URes
            # We do not print the residual for AD, though
            DASolver.solverAD.calcPrimalResidualStatistics("calc")

    def linearize(self, inputs, outputs, residuals):
        # NOTE: we do not do any computation in this function, just print some information

        self.DASolver.setStates(outputs[self.stateName])

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

        # assign the states in outputs to the OpenFOAM flow fields
        # NOTE: this is not quite necessary because setStates have been called before in the solve_nonlinear
        # here we call it just be on the safe side
        DASolver.setStates(outputs[self.stateName])

        if self.stateName in d_residuals:

            # get the reverse mode AD seed from d_residuals
            seed = d_residuals[self.stateName]

            # this computes [dRdW]^T*Psi using reverse mode AD
            if self.stateName in d_outputs:
                product = np.zeros(self.localAdjSize)
                jacInput = outputs[self.stateName]
                DASolver.solverAD.calcJacTVecProduct(
                    self.stateName,
                    "stateVar",
                    jacInput,
                    self.residualName,
                    "residual",
                    seed,
                    product,
                )
                d_outputs[self.stateName] += product

            # loop over all inputs keys and compute the matrix-vector products accordingly
            inputDict = DASolver.getOption("inputInfo")
            for inputName in list(inputs.keys()):
                inputType = inputDict[inputName]["type"]
                jacInput = inputs[inputName]
                product = np.zeros_like(jacInput)
                DASolver.solverAD.calcJacTVecProduct(
                    inputName,
                    inputType,
                    jacInput,
                    self.residualName,
                    "residual",
                    seed,
                    product,
                )
                d_inputs[inputName] += product

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

            adjEqnSolMethod = DASolver.getOption("adjEqnSolMethod")

            # right hand side array from d_outputs
            dFdWArray = d_outputs[self.stateName]
            # convert the array to vector
            dFdW = DASolver.array2Vec(dFdWArray)

            # run coloring
            if self.DASolver.getOption("adjUseColoring") and self.runColoring:
                self.DASolver.solver.runColoring()
                self.runColoring = False

            if adjEqnSolMethod == "Krylov":
                # solve the adjoint equation using the Krylov method

                # if writeMinorIterations=True, we rename the solution in pyDAFoam.py. So we don't recompute the PC
                if DASolver.getOption("writeMinorIterations"):
                    if DASolver.dRdWTPC is None or DASolver.ksp is None:
                        DASolver.dRdWTPC = PETSc.Mat().create(self.comm)
                        DASolver.solver.calcdRdWT(1, DASolver.dRdWTPC)
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
                        if DASolver.getOption("writeDeformedFFDs"):
                            if self.DVGeo is None:
                                raise RuntimeError(
                                    "writeDeformedFFDs is set but no DVGeo object found! Please call add_dvgeo in the run script!"
                                )
                            else:
                                self.DVGeo.writeTecplot("deformedFFDs_%d.dat" % self.solution_counter)
                                self.DVGeo.writePlot3d("deformedFFDs_%d.xyz" % self.solution_counter)

                        # write the deformed constraints for post-processing
                        if DASolver.getOption("writeDeformedConstraints"):
                            if self.DVCon is None:
                                raise RuntimeError(
                                    "writeDeformedConstraints is set but no DVCon object found! Please call add_dvcon in the run script!"
                                )
                            else:
                                self.DVCon.writeTecplot("deformedConstraints_%d.dat" % self.solution_counter)

                        # print the solution counter
                        if self.comm.rank == 0:
                            print("Driver total derivatives for iteration: %d" % self.solution_counter, flush=True)
                            print("---------------------------------------------", flush=True)
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
                            DASolver.solver.calcdRdWT(1, DASolver.dRdWTPC)
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
                else:
                    # if useNonZeroInitGuess is True, we will assign the OM's psi to self.psi
                    self.psi = DASolver.array2Vec(d_residuals[self.stateName].copy())

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
                        print("Driver total derivatives for iteration: %d" % self.solution_counter, flush=True)
                        print("---------------------------------------------", flush=True)
                    self.solution_counter += 1
                # solve the adjoint equation using the fixed-point adjoint approach
                fail = DASolver.solverAD.runFPAdj(dFdW, self.psi)
            else:
                raise RuntimeError("adjEqnSolMethod=%s not valid! Options are: Krylov or fixedPoint" % adjEqnSolMethod)

            # optionally write the adjoint vector as OpenFOAM field format for post-processing
            psi_array = DASolver.vec2Array(self.psi)
            solTimeFloat = (self.solution_counter - 1) / 1e4
            DASolver.writeAdjointFields("function", solTimeFloat, psi_array)

            # convert the solution vector to array and assign it to d_residuals
            d_residuals[self.stateName] = DASolver.vec2Array(self.psi)

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
        rArray = np.zeros(self.localAdjSize)
        jacInput = DASolver.getStates()
        seed = DASolver.vec2Array(psi)
        DASolver.solverAD.calcJacTVecProduct(
            self.stateName,
            "stateVar",
            jacInput,
            self.residualName,
            "residual",
            seed,
            rArray,
        )
        rVec = DASolver.array2Vec(rArray)
        rVec.axpy(-1.0, dFdW)
        # NOTE, this is the norm for the global vec
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
        DASolver = self.DASolver

        self.discipline = self.DASolver.getOption("discipline")

        # setup some names
        self.stateName = "%s_states" % self.discipline
        self.volCoordName = "%s_vol_coords" % self.discipline

        # setup input and output for the function
        # we need to add states for all cases
        self.add_input(self.stateName, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])

        # now loop over the solver input keys to determine which other variables we need to add as inputs
        inputDict = DASolver.getOption("inputInfo")
        for inputName in list(inputDict.keys()):
            # this input is attached to function comp
            if "function" in inputDict[inputName]["components"]:
                inputType = inputDict[inputName]["type"]
                inputSize = DASolver.solver.getInputSize(inputName, inputType)
                inputDistributed = DASolver.solver.getInputDistributed(inputName, inputType)
                self.add_input(inputName, distributed=inputDistributed, shape=inputSize, tags=["mphys_coupling"])

        # add outputs
        functions = self.DASolver.getOption("function")
        # loop over the functions here and create the output
        for f_name in list(functions.keys()):
            self.add_output(f_name, distributed=False, shape=1, units=None)

    # get the objective function from DASolver
    def compute(self, inputs, outputs):

        # TODO. We should have added a call to assign inputs to the OF layer.
        # This will not cause a problem for now because DAFoamFunctions is usually
        # called right after the primal run.

        DASolver = self.DASolver

        DASolver.setStates(inputs[self.stateName])

        funcs = {}
        DASolver.evalFunctions(funcs)
        for f_name in list(outputs.keys()):
            outputs[f_name] = funcs[f_name]

    # compute the partial derivatives of functions
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        DASolver = self.DASolver

        # TODO. this may not be needed
        DASolver.setStates(inputs[self.stateName])

        # we do not support forward mode AD
        if mode == "fwd":
            om.issue_warning(
                " mode = %s, but the forward mode functions are not implemented for DAFoam!" % mode,
                prefix="",
                stacklevel=2,
                category=om.OpenMDAOWarning,
            )
            return

        # loop over all d_inputs keys and compute the partials accordingly
        inputDict = DASolver.getOption("inputInfo")
        for functionName in list(d_outputs.keys()):

            seed = d_outputs[functionName]

            # if the seed is zero, do not compute
            if abs(seed) < 1e-12:
                continue

            for inputName in list(d_inputs.keys()):
                # compute dFdW * seed
                if inputName == self.stateName:
                    jacInput = inputs[self.stateName]
                    product = np.zeros_like(jacInput)
                    DASolver.solverAD.calcJacTVecProduct(
                        self.stateName,
                        "stateVar",
                        jacInput,
                        functionName,
                        "function",
                        seed,
                        product,
                    )
                    d_inputs[self.stateName] += product
                else:
                    inputType = inputDict[inputName]["type"]
                    jacInput = inputs[inputName]
                    product = np.zeros_like(jacInput)
                    DASolver.solverAD.calcJacTVecProduct(
                        inputName,
                        inputType,
                        jacInput,
                        functionName,
                        "function",
                        seed,
                        product,
                    )
                    d_inputs[inputName] += product


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
        DASolver = self.DASolver

        self.discipline = self.DASolver.getOption("discipline")

        self.stateName = "%s_states" % self.discipline
        self.volCoordName = "%s_vol_coords" % self.discipline

        self.add_input(self.volCoordName, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_input(self.stateName, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])

        # now loop over the solver input keys to determine which other variables we need to add as inputs
        outputDict = DASolver.getOption("outputInfo")
        for outputName in list(outputDict.keys()):
            # this input is attached to the DAFoamThermal comp
            if "thermalCoupling" in outputDict[outputName]["components"]:
                self.outputName = outputName
                self.outputType = outputDict[outputName]["type"]
                self.outputSize = DASolver.solver.getOutputSize(outputName, self.outputType)
                outputDistributed = DASolver.solver.getOutputDistributed(outputName, self.outputType)
                self.add_output(
                    outputName, distributed=outputDistributed, shape=self.outputSize, tags=["mphys_coupling"]
                )
                break

    def compute(self, inputs, outputs):

        self.DASolver.setStates(inputs[self.stateName])
        self.DASolver.setVolCoords(inputs[self.volCoordName])

        thermal = np.zeros(self.outputSize)

        self.DASolver.solver.calcOutput(self.outputName, self.outputType, thermal)

        outputs[self.outputName] = thermal

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

        for outputName in list(d_outputs.keys()):
            seeds = d_outputs[outputName]

            if self.stateName in d_inputs:
                jacInput = inputs[self.stateName]
                product = np.zeros_like(jacInput)
                DASolver.solverAD.calcJacTVecProduct(
                    self.stateName,
                    "stateVar",
                    jacInput,
                    outputName,
                    "thermalCouplingOutput",
                    seeds,
                    product,
                )
                d_inputs[self.stateName] += product

            if self.volCoordName in d_inputs:
                jacInput = inputs[self.volCoordName]
                product = np.zeros_like(jacInput)
                DASolver.solverAD.calcJacTVecProduct(
                    self.volCoordName,
                    "volCoord",
                    jacInput,
                    outputName,
                    "thermalCouplingOutput",
                    seeds,
                    product,
                )
                d_inputs[self.volCoordName] += product


class DAFoamFaceCoords(ExplicitComponent):
    """
    Calculate coupling surface coordinates based on volume coordinates

    """

    def initialize(self):
        self.options.declare("solver", recordable=False)

    def setup(self):

        self.DASolver = self.options["solver"]
        self.discipline = self.DASolver.getOption("discipline")
        self.volCoordName = "%s_vol_coords" % self.discipline
        self.surfCoordName = "x_%s_surface0" % self.discipline

        DASolver = self.DASolver

        self.add_input(self.volCoordName, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])

        # now loop over the solver input keys to determine which other variables we need to add as inputs
        self.nSurfCoords = None
        outputDict = DASolver.getOption("outputInfo")
        for outputName in list(outputDict.keys()):
            # this input is attached to the DAFoamThermal comp
            if "thermalCoupling" in outputDict[outputName]["components"]:
                outputType = outputDict[outputName]["type"]
                outputSize = DASolver.solver.getOutputSize(outputName, outputType)
                # NOTE: here x_surface0 is the surface coordinate, which is 3 times the number of faces
                self.nSurfCoords = outputSize * 3
                self.add_output(self.surfCoordName, distributed=True, shape=self.nSurfCoords, tags=["mphys_coupling"])
                break

        if self.nSurfCoords is None:
            raise AnalysisError("no thermalCoupling output found!")

    def compute(self, inputs, outputs):

        volCoords = inputs[self.volCoordName]
        surfCoords = np.zeros(self.nSurfCoords)
        self.DASolver.solver.calcCouplingFaceCoords(volCoords, surfCoords)

        outputs[self.surfCoordName] = surfCoords

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        # there is no need to compute the jacvec_product because FUN2FEM assumes the surfCoords will not updated during optimization
        # so it will set a zero seed for this anyway
        pass


class DAFoamForces(ExplicitComponent):
    """
    OpenMDAO component that wraps force integration

    """

    def initialize(self):
        self.options.declare("solver", recordable=False)

    def setup(self):

        self.DASolver = self.options["solver"]

        self.discipline = self.DASolver.getOption("discipline")

        self.stateName = "%s_states" % self.discipline
        self.volCoordName = "%s_vol_coords" % self.discipline

        self.add_input(self.volCoordName, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_input(self.stateName, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])

        # now loop over the solver input keys to determine which other variables we need to add as inputs
        outputDict = self.DASolver.getOption("outputInfo")
        for outputName in list(outputDict.keys()):
            # this input is attached to the DAFoamThermal comp
            if "forceCoupling" in outputDict[outputName]["components"]:
                self.outputName = outputName
                self.outputType = outputDict[outputName]["type"]
                outputSize = self.DASolver.solver.getOutputSize(self.outputName, self.outputType)
                self.add_output("f_aero", distributed=True, shape=outputSize, tags=["mphys_coupling"])
                break

    def compute(self, inputs, outputs):

        self.DASolver.setStates(inputs[self.stateName])
        self.DASolver.setVolCoords(inputs[self.volCoordName])

        forces = np.zeros_like(outputs["f_aero"])

        self.DASolver.solver.calcOutput(self.outputName, self.outputType, forces)

        outputs["f_aero"] = forces

        # print out the total forces. They shoud be consistent with the primal's print out
        forcesV = forces.reshape((-1, 3))
        fXSum = np.sum(forcesV[:, 0])
        fYSum = np.sum(forcesV[:, 1])
        fZSum = np.sum(forcesV[:, 2])
        fXTot = self.comm.allreduce(fXSum, op=MPI.SUM)
        fYTot = self.comm.allreduce(fYSum, op=MPI.SUM)
        fZTot = self.comm.allreduce(fZSum, op=MPI.SUM)

        if self.comm.rank == 0:
            print("Total force:", flush=True)
            print("Fx = %f" % fXTot, flush=True)
            print("Fy = %f" % fYTot, flush=True)
            print("Fz = %f" % fZTot, flush=True)

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
            seeds = d_outputs["f_aero"]

            if self.stateName in d_inputs:
                jacInput = inputs[self.stateName]
                product = np.zeros_like(jacInput)
                DASolver.solverAD.calcJacTVecProduct(
                    self.stateName,
                    "stateVar",
                    jacInput,
                    self.outputName,
                    "forceCouplingOutput",
                    seeds,
                    product,
                )
                d_inputs[self.stateName] += product

            if self.volCoordName in d_inputs:
                jacInput = inputs[self.volCoordName]
                product = np.zeros_like(jacInput)
                DASolver.solverAD.calcJacTVecProduct(
                    self.volCoordName,
                    "volCoord",
                    jacInput,
                    self.outputName,
                    "forceCouplingOutput",
                    seeds,
                    product,
                )
                d_inputs[self.volCoordName] += product


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
                print("FindFeasibleDesign Iter: ", n, flush=True)
                print("DesignVars: ", dv0, flush=True)
                print("Constraints: ", con0, flush=True)
                print("Residual Norm: ", norm, flush=True)

            # break the loop if residual is already smaller than the tolerance
            if norm < tol:
                if self.comm.rank == 0:
                    print("FindFeasibleDesign Converged! ", flush=True)
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


class DAFoamBuilderUnsteady(Group):

    def initialize(self):
        self.options.declare("solver_options")
        self.options.declare("mesh_options", default=None)
        self.options.declare("run_directory", default="")

    def setup(self):
        self.run_directory = self.options["run_directory"]
        self.solver_options = self.options["solver_options"]
        self.mesh_options = self.options["mesh_options"]

        with cd(self.run_directory):
            # initialize the PYDAFOAM class, defined in pyDAFoam.py
            self.DASolver = PYDAFOAM(options=self.solver_options, comm=self.comm)
            if self.mesh_options is not None:
                # always set the mesh
                mesh = USMesh(options=self.mesh_options, comm=self.comm)
                self.DASolver.setMesh(mesh)  # add the design surface family group
                self.DASolver.printFamilyList()

            self.x_a0 = self.DASolver.getSurfaceCoordinates(self.DASolver.designSurfacesGroup).flatten(order="C")

        # if we have volume coords as the input, add the warper comp here
        inputDict = self.DASolver.getOption("inputInfo")
        for inputName in list(inputDict.keys()):
            # this input is attached to solver comp
            if "solver" in inputDict[inputName]["components"]:
                inputType = inputDict[inputName]["type"]
                if inputType == "volCoord":
                    self.add_subsystem("warper", DAFoamWarper(solver=self.DASolver), promotes=["*"])
                    break

        # add the solver comp
        self.add_subsystem("solver", DAFoamSolverUnsteady(solver=self.DASolver), promotes=["*"])

    def get_surface_mesh(self):
        return self.x_a0


class DAFoamSolverUnsteady(ExplicitComponent):

    def initialize(self):
        self.options.declare("solver")
        self.options.declare("run_directory", default="")

    def setup(self):
        self.DVGeo = None
        self.DASolver = self.options["solver"]
        self.run_directory = self.options["run_directory"]

        DASolver = self.DASolver

        self.dRdWTPC = None

        self.adjEqnSolMethod = DASolver.getOption("adjEqnSolMethod")
        if self.adjEqnSolMethod not in ["Krylov", "fixedPoint"]:
            raise AnalysisError("adjEqnSolMethod is not valid")

        inputDict = DASolver.getOption("inputInfo")
        for inputName in list(inputDict.keys()):
            # this input is attached to solver comp
            if "solver" in inputDict[inputName]["components"]:
                inputType = inputDict[inputName]["type"]
                inputSize = DASolver.solver.getInputSize(inputName, inputType)
                inputDistributed = DASolver.solver.getInputDistributed(inputName, inputType)
                self.add_input(inputName, distributed=inputDistributed, shape=inputSize)

        # here we define a set of unsteady component output, they can be a linear combination
        # of the functions defined in DAOption->function
        self.unsteadyCompOutput = DASolver.getOption("unsteadyCompOutput")
        if len(self.unsteadyCompOutput) == 0:
            raise AnalysisError("unsteadyCompOutput is not defined for unsteady cases")

        functions = DASolver.getOption("function")
        for outputName in list(self.unsteadyCompOutput.keys()):
            # add the output
            self.add_output(outputName, distributed=0, shape=1)
            # also verify if the function names defined in the unsteadyCompOutput subdict
            # is found in DAOption->function
            for funcName in self.unsteadyCompOutput[outputName]:
                if funcName not in functions:
                    raise AnalysisError("%f is not found in DAOption-function" % funcName)

    def add_dvgeo(self, DVGeo):
        self.DVGeo = DVGeo

    def compute(self, inputs, outputs):

        with cd(self.run_directory):

            DASolver = self.DASolver

            # if readZeroFields, we need to read in the states from the 0 folder every time
            # we start the primal here we read in all time levels. If readZeroFields is not set,
            # we will use the latest flow fields (from a previous primal call) as the init conditions
            readZeroFields = DASolver.getOption("unsteadyAdjoint")["readZeroFields"]
            if readZeroFields:
                DASolver.solver.setTime(0.0, 0)
                deltaT = DASolver.solver.getDeltaT()
                DASolver.readStateVars(0.0, deltaT)

            # set the solver inputs.
            # NOTE: we need to set input after we read the zero fields for forward mode
            DASolver.set_solver_input(inputs, self.DVGeo)
            # if dyamic mesh is used, we need to deform the mesh points and save them to disk
            DASolver.deformDynamicMesh()

            # before running the primal, we need to check if the mesh
            # quality is good
            meshOK = DASolver.solver.checkMesh()

            # solve the flow with the current design variable
            # if the mesh is not OK, do not run the primal
            if meshOK:
                # solve the primal
                DASolver()
            else:
                # if the mesh fails, return
                raise AnalysisError("Primal mesh failed!")
                return

            # if the primal solution fails, return
            if DASolver.primalFail != 0:
                raise AnalysisError("Primal solution failed!")
                return

            # get the objective functions
            funcs = {}
            DASolver.evalFunctions(funcs)

            # now we can print the residual for the endTime state
            DASolver.solver.calcPrimalResidualStatistics("print")

            for outputName in list(self.unsteadyCompOutput.keys()):
                outputs[outputName] = 0.0
                # add all the function values for this output
                for funcName in self.unsteadyCompOutput[outputName]:
                    outputs[outputName] += funcs[funcName]

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

        # run coloring
        if self.adjEqnSolMethod == "Krylov":
            DASolver.solver.runColoring()

            PCMatPrecomputeInterval = DASolver.getOption("unsteadyAdjoint")["PCMatPrecomputeInterval"]
            PCMatUpdateInterval = DASolver.getOption("unsteadyAdjoint")["PCMatUpdateInterval"]

        # NOTE: this step is critical because we need to compute the residual for
        # self.solverAD once to get the proper oldTime level for unsteady adjoint
        DASolver.solverAD.updateStateBoundaryConditions()
        DASolver.solverAD.calcPrimalResidualStatistics("calc")

        # calc the total number of time instances
        # we assume the adjoint is for deltaT to endTime
        # but users can also prescribed a custom time range
        deltaT = DASolver.solver.getDeltaT()

        endTime = DASolver.solver.getEndTime()
        endTimeIndex = round(endTime / deltaT)

        localAdjSize = DASolver.getNLocalAdjointStates()

        ddtSchemeOrder = DASolver.solver.getDdtSchemeOrder()

        # read the latest solution
        DASolver.solver.setTime(endTime, endTimeIndex)
        DASolver.solverAD.setTime(endTime, endTimeIndex)
        # now we can read the variables
        DASolver.readStateVars(endTime, deltaT)
        # if it is dynamic mesh, read the mesh points
        if DASolver.getOption("dynamicMesh")["active"]:
            DASolver.readDynamicMeshPoints(endTime, deltaT, endTimeIndex, ddtSchemeOrder)

        # now we can print the residual for the endTime state
        DASolver.solverAD.calcPrimalResidualStatistics("print")

        # init dRdWTMF
        if self.adjEqnSolMethod == "Krylov":
            DASolver.solverAD.initializedRdWTMatrixFree()

        # precompute the KSP preconditioner Mat and save them to the self.dRdWTPC dict
        if self.dRdWTPC is None and self.adjEqnSolMethod == "Krylov":

            self.dRdWTPC = {}

            # always calculate the PC mat for the endTime
            DASolver.solver.setTime(endTime, endTimeIndex)
            DASolver.solverAD.setTime(endTime, endTimeIndex)
            # now we can read the variables
            DASolver.readStateVars(endTime, deltaT)
            # if it is dynamic mesh, read the mesh points
            if DASolver.getOption("dynamicMesh")["active"]:
                DASolver.readDynamicMeshPoints(endTime, deltaT, endTimeIndex, ddtSchemeOrder)
            # calc the preconditioner mat for endTime
            if self.comm.rank == 0:
                print("Pre-Computing preconditiner mat for t = %f" % endTime, flush=True)

            dRdWTPC1 = PETSc.Mat().create(PETSc.COMM_WORLD)
            DASolver.solver.calcdRdWT(1, dRdWTPC1)
            # always update the PC mat values using OpenFOAM's fvMatrix
            # DASolver.solver.calcPCMatWithFvMatrix(dRdWTPC1)
            self.dRdWTPC[str(endTime)] = dRdWTPC1

            # if we define some extra PCMat in PCMatPrecomputeInterval, calculate them here
            # and set them to the self.dRdWTPC dict
            for timeIndex in range(endTimeIndex - 1, 0, -1):
                if timeIndex % PCMatPrecomputeInterval == 0:
                    t = timeIndex * deltaT
                    if self.comm.rank == 0:
                        print("Pre-Computing preconditiner mat for t = %f" % t, flush=True)
                    # read the latest solution
                    DASolver.solver.setTime(t, timeIndex)
                    DASolver.solverAD.setTime(t, timeIndex)
                    # now we can read the variables
                    DASolver.readStateVars(t, deltaT)
                    # if it is dynamic mesh, read the mesh points
                    if DASolver.getOption("dynamicMesh")["active"]:
                        DASolver.readDynamicMeshPoints(t, deltaT, timeIndex, ddtSchemeOrder)
                    # calc the preconditioner mat
                    dRdWTPC1 = PETSc.Mat().create(PETSc.COMM_WORLD)
                    DASolver.solver.calcdRdWT(1, dRdWTPC1)
                    # always update the PC mat values using OpenFOAM's fvMatrix
                    # DASolver.solver.calcPCMatWithFvMatrix(dRdWTPC1)
                    self.dRdWTPC[str(t)] = dRdWTPC1

        if self.adjEqnSolMethod == "Krylov":
            # Initialize the KSP object using the PCMat from the endTime
            PCMat = self.dRdWTPC[str(endTime)]
            ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
            DASolver.solverAD.createMLRKSPMatrixFree(PCMat, ksp)

        inputDict = DASolver.getOption("inputInfo")

        # init the dFdW vec
        dFdW = PETSc.Vec().create(PETSc.COMM_WORLD)
        dFdW.setSizes((localAdjSize, PETSc.DECIDE), bsize=1)
        dFdW.setFromOptions()
        dFdW.zeroEntries()

        # init the adjoint vector
        psi = dFdW.duplicate()
        psi.zeroEntries()

        # initialize the adjoint vecs
        dRdW0TPsi = np.zeros(localAdjSize)
        dRdW00TPsi = np.zeros(localAdjSize)
        dRdW00TPsiBuffer = np.zeros(localAdjSize)
        dFdWArray = np.zeros(localAdjSize)
        psiArray = np.zeros(localAdjSize)
        tempdFdWArray = np.zeros(localAdjSize)

        # loop over all function, calculate dFdW, and solve the adjoint
        # for functionName in list(d_outputs.keys()):
        for outputName in list(self.unsteadyCompOutput.keys()):

            # we need to zero the total derivative for each function
            totals = {}
            for inputName in list(inputs.keys()):
                totals[inputName] = np.zeros_like(inputs[inputName])

            seed = d_outputs[outputName]

            # if the seed is zero, do not compute the adjoint and pass to
            # the next funnction
            if np.linalg.norm(seed) < 1e-12:
                continue

            # zero the vecs
            dRdW0TPsi[:] = 0.0
            dRdW00TPsi[:] = 0.0
            dRdW00TPsiBuffer[:] = 0.0
            dFdW.zeroEntries()

            # loop over all time steps and solve the adjoint and accumulate the totals
            adjointFail = 0
            for n in range(endTimeIndex, 0, -1):
                timeVal = n * deltaT

                if self.comm.rank == 0:
                    print("---- Solving unsteady adjoint for %s. t = %f ----" % (outputName, timeVal), flush=True)

                # set the time value and index in the OpenFOAM layer. Note: this is critical
                # because if timeIndex < 2, OpenFOAM will not use the oldTime.oldTime for 2nd
                # ddtScheme and mess up the totals. Check backwardDdtScheme.C
                DASolver.solver.setTime(timeVal, n)
                DASolver.solverAD.setTime(timeVal, n)
                # now we can read the variables
                # read the state, state.oldTime, etc and update self.wVec for this time instance
                DASolver.readStateVars(timeVal, deltaT)
                # if it is dynamic mesh, read the mesh points
                if DASolver.getOption("dynamicMesh")["active"]:
                    DASolver.readDynamicMeshPoints(timeVal, deltaT, n, ddtSchemeOrder)

                # calculate dFd? scaling, if time index is within the unsteady objective function
                # index range, prescribed in unsteadyAdjointDict, we calculate dFdW
                # otherwise, we use dFdW=0 because the unsteady obj does not depend
                # on the state at this time index.
                # NOTE: we just use the first function in the output for dFScaling and
                # assume all the functions to have the same timeOp for this output
                firstFunctionName = self.unsteadyCompOutput[outputName][0]
                dFScaling = DASolver.solver.getdFScaling(firstFunctionName, n - 1)

                # loop over all function for this output, compute their dFdW, and add them up to
                # get the dFdW for this output
                dFdWArray[:] = 0.0
                for functionName in self.unsteadyCompOutput[outputName]:

                    # calculate dFdW
                    jacInput = DASolver.getStates()
                    DASolver.solverAD.calcJacTVecProduct(
                        "states",
                        "stateVar",
                        jacInput,
                        functionName,
                        "function",
                        seed,
                        tempdFdWArray,  # NOTE: we just use tempdFdWArray to hold a temp dFdW for this function
                    )

                    dFdWArray += tempdFdWArray * dFScaling

                # do dFdW - dRdW0TPsi - dRdW00TPsi
                if ddtSchemeOrder == 1:
                    dFdWArray = dFdWArray - dRdW0TPsi
                elif ddtSchemeOrder == 2:
                    dFdWArray = dFdWArray - dRdW0TPsi - dRdW00TPsi
                    # now copy the buffer vec dRdW00TPsiBuffer to dRdW00TPsi for the next time step
                    dRdW00TPsi[:] = dRdW00TPsiBuffer
                else:
                    print("ddtSchemeOrder not valid!" % ddtSchemeOrder)

                # check if we need to update the PC Mat vals or use the pre-computed PC matrix
                if self.adjEqnSolMethod == "Krylov":
                    if str(timeVal) in list(self.dRdWTPC.keys()):
                        if self.comm.rank == 0:
                            print("Using pre-computed KSP PC mat for %f" % timeVal, flush=True)
                        PCMat = self.dRdWTPC[str(timeVal)]
                        DASolver.solverAD.updateKSPPCMat(PCMat, ksp)
                    if n % PCMatUpdateInterval == 0 and n < endTimeIndex:
                        # udpate part of the PC mat
                        if self.comm.rank == 0:
                            print("Updating dRdWTPC mat value using OF fvMatrix", flush=True)
                        DASolver.solver.calcPCMatWithFvMatrix(PCMat)

                # now solve the adjoint eqn
                DASolver.arrayVal2Vec(dFdWArray, dFdW)

                if self.adjEqnSolMethod == "Krylov":
                    adjointFail = DASolver.solverAD.solveLinearEqn(ksp, dFdW, psi)
                elif self.adjEqnSolMethod == "fixedPoint":
                    adjointFail = DASolver.solverAD.solveAdjointFP(dFdW, psi)

                # if one adjoint solution fails, return immediate without solving for the rest of steps.
                if adjointFail > 0:
                    break

                # loop over all inputs and compute total derivs
                for inputName in list(inputs.keys()):

                    # calculate dFdX
                    inputType = inputDict[inputName]["type"]
                    jacInput = inputs[inputName]
                    dFdX = np.zeros_like(jacInput)
                    tempdFdX = np.zeros_like(jacInput)

                    # loop over all function for this output, compute their dFdX, and add them up to
                    # get the dFdX for this output
                    for functionName in self.unsteadyCompOutput[outputName]:
                        DASolver.solverAD.calcJacTVecProduct(
                            inputName,
                            inputType,
                            jacInput,
                            functionName,
                            "function",
                            seed,
                            tempdFdX,
                        )
                        # we need to scale the dFdX for unsteady adjoint too
                        dFdX += tempdFdX * dFScaling

                    # calculate dRdX^T * psi
                    dRdXTPsi = np.zeros_like(jacInput)
                    DASolver.vecVal2Array(psi, psiArray)
                    DASolver.solverAD.calcJacTVecProduct(
                        inputName,
                        inputType,
                        jacInput,
                        "residual",
                        "residual",
                        psiArray,
                        dRdXTPsi,
                    )

                    # total derivative
                    totals[inputName] += dFdX - dRdXTPsi

                # we need to calculate dRdW0TPsi for the previous time step
                if ddtSchemeOrder == 1:
                    DASolver.solverAD.calcdRdWOldTPsiAD(1, psiArray, dRdW0TPsi)
                elif ddtSchemeOrder == 2:
                    # do the same for the previous previous step, but we need to save it to a buffer vec
                    # because dRdW00TPsi will be used 2 steps before
                    DASolver.solverAD.calcdRdWOldTPsiAD(1, psiArray, dRdW0TPsi)
                    DASolver.solverAD.calcdRdWOldTPsiAD(2, psiArray, dRdW00TPsiBuffer)

            for inputName in list(inputs.keys()):
                d_inputs[inputName] += totals[inputName]

        # once the adjoint is done, we will assign OF fields with the endTime solution
        # so, if the next primal does not read fields from the 0 time, we will continue
        # to use the latest solutions from the previous design as the initial field
        DASolver.solver.setTime(endTime, endTimeIndex)
        DASolver.solverAD.setTime(endTime, endTimeIndex)
        DASolver.readStateVars(endTime, deltaT)
