import sys
from openmdao.api import Group, ImplicitComponent, ExplicitComponent, AnalysisError
from dafoam import PYDAFOAM
from idwarp import USMesh
from mphys.builder import Builder
import petsc4py
from petsc4py import PETSc

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
    ):

        # options dictionary for DAFoam
        self.options = options

        # mesh warping option
        if mesh_options is None:
            raise AnalysisError("mesh_options not found!")
        else:
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

        # depending on the scenario we are building for, we adjust a few internal parameters:
        if scenario.lower() == "aerodynamic":
            # default
            pass
        elif scenario.lower() == "aerostructural":
            # volume mesh warping needs to be inside the coupling loop for aerostructural
            self.warp_in_solver = True
            self.struct_coupling = True
        else:
            raise AnalysisError("scenario %s not valid! Options: aerodynamic, aerostructural" % scenario)

    # api level method for all builders
    def initialize(self, comm):
        # initialize the PYDAFOAM class, defined in pyDAFoam.py
        self.DASolver = PYDAFOAM(options=self.options, comm=comm)
        # always set the mesh
        mesh = USMesh(options=self.mesh_options, comm=comm)
        self.DASolver.setMesh(mesh)
        # add the design surface family group
        self.DASolver.addFamilyGroup(
            self.DASolver.getOption("designSurfaceFamily"), self.DASolver.getOption("designSurfaces")
        )
        self.DASolver.printFamilyList()

    def get_solver(self):
        # this method is only used by the RLT transfer scheme
        return self.DASolver

    # api level method for all builders
    def get_coupling_group_subsystem(self, scenario_name=None):
        dafoam_group = DAFoamGroup(
            solver=self.DASolver, use_warper=self.warp_in_solver, struct_coupling=self.struct_coupling
        )
        return dafoam_group

    def get_mesh_coordinate_subsystem(self, scenario_name=None):

        # just return the component that outputs the surface mesh.
        return DAFoamMesh(solver=self.DASolver)

    def get_pre_coupling_subsystem(self, scenario_name=None):
        # we warp as a pre-processing step
        if self.warp_in_solver:
            # if we warp in the solver, then we wont have any pre-coupling systems
            return None
        else:
            # we warp as a pre-processing step
            return DAFoamWarper(solver=self.DASolver)

    def get_post_coupling_subsystem(self, scenario_name=None):
        return DAFoamFunctions(solver=self.DASolver)

    # TODO the get_nnodes is deprecated. will remove
    def get_nnodes(self, groupName=None):
        if groupName is None:
            groupName = self.DASolver.designFamilyGroup
        return int(self.DASolver.getSurfaceCoordinates(groupName=groupName).size / 3)

    def get_number_of_nodes(self, groupName=None):
        if groupName is None:
            groupName = self.DASolver.designFamilyGroup
        return int(self.DASolver.getSurfaceCoordinates(groupName=groupName).size / 3)


class DAFoamGroup(Group):
    """
    DAFoam solver group
    """

    def initialize(self):
        self.options.declare("solver", recordable=False)
        self.options.declare("struct_coupling", default=False)
        self.options.declare("use_warper", default=True)

    def setup(self):

        self.DASolver = self.options["solver"]
        self.struct_coupling = self.options["struct_coupling"]
        self.use_warper = self.options["use_warper"]

        if self.use_warper:
            # if we dont have geo_disp, we also need to promote the x_a as x_a0 from the deformer component
            self.add_subsystem(
                "deformer",
                DAFoamWarper(
                    solver=self.DASolver,
                ),
                promotes_inputs=["x_aero"],
                promotes_outputs=["dafoam_vol_coords"],
            )

        # add the solver implicit component
        self.add_subsystem(
            "solver",
            DAFoamSolver(solver=self.DASolver),
            promotes_inputs=["*"],
            promotes_outputs=["dafoam_states"],
        )

        if self.struct_coupling:
            self.add_subsystem(
                "force",
                DAFoamForces(solver=self.DASolver),
                promotes_inputs=["dafoam_vol_coords", "dafoam_states"],
                promotes_outputs=["f_aero"],
            )

    def mphys_set_options(self, optionDict):
        # here optionDict should be a dictionary that has a consistent format
        # with the daOptions defined in the run script
        self.solver.set_options(optionDict)


class DAFoamSolver(ImplicitComponent):
    """
    OpenMDAO component that wraps the DAFoam flow and adjoint solvers
    """

    def initialize(self):
        self.options.declare("solver", recordable=False)

    def setup(self):
        # NOTE: the setup function will be called everytime a new scenario is created.

        self.DASolver = self.options["solver"]
        DASolver = self.DASolver

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

        # run coloring
        if self.DASolver.getOption("adjUseColoring"):
            self.DASolver.runColoring()

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
        self.add_output("dafoam_states", distributed=True, shape=local_state_size, tags=["mphys_coupling"])

        # now loop over the design variable keys to determine which other variables we need to add
        shapeVarAdded = False
        for dvName in list(designVariables.keys()):
            dvType = self.dvType[dvName]
            if dvType == "FFD":  # add shape variables
                if shapeVarAdded is False:  # we add the shape variable only once
                    # NOTE: for shape variables, we add dafoam_vol_coords as the input name
                    # the specific name for this shape variable will be added in the geometry component (DVGeo)
                    self.add_input("dafoam_vol_coords", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
                    shapeVarAdded = True
            elif dvType == "AOA":  # add angle of attack variable
                self.add_input(dvName, distributed=False, shape_by_conn=True, tags=["mphys_coupling"])
            elif dvType == "BC":  # add boundary conditions
                self.add_input(dvName, distributed=False, shape_by_conn=True, tags=["mphys_coupling"])
            elif dvType == "ACTD":  # add actuator parameter variables
                self.add_input(dvName, distributed=False, shape=9, tags=["mphys_coupling"])
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
        DASolver.setStates(outputs["dafoam_states"])

        # get flow residuals from DASolver
        residuals["dafoam_states"] = DASolver.getResiduals()

    # solve the flow
    def solve_nonlinear(self, inputs, outputs):

        DASolver = self.DASolver
        if self.comm.rank == 0:
            print("\n", flush=True)
            print("+------------------------------------------------------+", flush=True)
            print("|          Evaluating Objective Functions %03d          |" % DASolver.nSolvePrimals, flush=True)
            print("+------------------------------------------------------+", flush=True)

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

        # solve the flow with the current design variable
        DASolver()

        # get the objective functions
        funcs = {}
        DASolver.evalFunctions(funcs, evalFuncs=self.evalFuncs)

        # assign the computed flow states to outputs
        outputs["dafoam_states"] = DASolver.getStates()

        # if the primal solution fail, we return analysisError and let the optimizer handle it
        fail = funcs["fail"]
        if fail:
            raise AnalysisError("Primal solution failed!")

    def linearize(self, inputs, outputs, residuals):
        # NOTE: we do not do any computation in this function, just print some information

        DASolver = self.DASolver

        if self.comm.rank == 0:
            print("\n", flush=True)
            print("+------------------------------------------------------+", flush=True)
            print("|    Evaluating Objective Function Sensitivities %03d   |" % DASolver.nSolveAdjoints, flush=True)
            print("+------------------------------------------------------+", flush=True)

        # move the solution folder to 0.000000x
        DASolver.renameSolution(DASolver.nSolveAdjoints)

        # set the runStatus, this is useful when the actuator term is activated
        DASolver.setOption("runStatus", "solveAdjoint")
        DASolver.updateDAOption()

        DASolver.setStates(outputs["dafoam_states"])

        DASolver.nSolveAdjoints += 1

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        # compute the matrix vector products for states and volume mesh coordinates
        # i.e., dRdWT*psi, dRdXvT*psi

        # we do not support forward mode
        if mode == "fwd":
            raise AnalysisError("fwd mode not implemented!")

        DASolver = self.DASolver

        # assign the optionDict to the solver
        self.apply_options(self.optionDict)
        
        # now call the dv_funcs to update the design variables
        for dvName in self.dv_funcs:
            func = self.dv_funcs[dvName]
            dvVal = inputs[dvName]
            func(dvVal, DASolver)

        # assign the states in outputs to the OpenFOAM flow fields
        DASolver.setStates(outputs["dafoam_states"])

        if "dafoam_states" in d_residuals:

            # get the reverse mode AD seed from d_residuals
            resBar = d_residuals["dafoam_states"]
            # convert the seed array to Petsc vector
            resBarVec = DASolver.array2Vec(resBar)

            # this computes [dRdW]^T*Psi using reverse mode AD
            if "dafoam_states" in d_outputs:
                prodVec = DASolver.wVec.duplicate()
                prodVec.zeroEntries()
                DASolver.solverAD.calcdRdWTPsiAD(DASolver.xvVec, DASolver.wVec, resBarVec, prodVec)
                wBar = DASolver.vec2Array(prodVec)
                d_outputs["dafoam_states"] += wBar

            # loop over all d_inputs keys and compute the matrix-vector products accordingly
            for inputName in list(d_inputs.keys()):
                # this computes [dRdXv]^T*Psi using reverse mode AD
                if inputName == "dafoam_vol_coords":
                    prodVec = DASolver.xvVec.duplicate()
                    prodVec.zeroEntries()
                    DASolver.solverAD.calcdRdXvTPsiAD(DASolver.xvVec, DASolver.wVec, resBarVec, prodVec)
                    xVBar = DASolver.vec2Array(prodVec)
                    d_inputs["dafoam_vol_coords"] += xVBar

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
                        prodVec.setSizes((PETSc.DECIDE, 9), bsize=1)
                        prodVec.setFromOptions()
                        DASolver.solverAD.calcdRdActTPsiAD(
                            DASolver.xvVec, DASolver.wVec, resBarVec, inputName.encode(), prodVec
                        )
                        # we will convert the MPI prodVec to seq array for all procs
                        ACTDBar = DASolver.convertMPIVec2SeqArray(prodVec)
                        d_inputs[inputName] += ACTDBar

                    # compute dRdFieldT*Psi using reverse mode AD
                    elif self.dvType[inputName] == "Field":
                        nLocalCells = self.solver.getNLocalCells()
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
            raise AnalysisError("fwd mode not implemented!")

        DASolver = self.DASolver

        # compute the preconditioiner matrix for the adjoint linear equation solution
        # NOTE: we compute this only once and will reuse it during optimization
        # similarly, we will create the ksp once and reuse
        if DASolver.dRdWTPC is None:
            DASolver.cdRoot()
            DASolver.dRdWTPC = PETSc.Mat().create(self.comm)
            DASolver.solver.calcdRdWT(DASolver.xvVec, DASolver.wVec, 1, DASolver.dRdWTPC)

        # NOTE: here we reuse the KSP object defined in pyDAFoam.py
        if DASolver.ksp is None:
            DASolver.ksp = PETSc.KSP().create(self.comm)
            DASolver.solverAD.createMLRKSPMatrixFree(DASolver.dRdWTPC, DASolver.ksp)

        # right hand side array from d_outputs
        dFdWArray = d_outputs["dafoam_states"]
        # convert the array to vector
        dFdW = DASolver.array2Vec(dFdWArray)
        # update the KSP tolerances the coupled adjoint before solving
        self._updateKSPTolerances(self.psi, dFdW, DASolver.ksp)
        # actually solving the adjoint linear equation using Petsc
        fail = DASolver.solverAD.solveLinearEqn(DASolver.ksp, dFdW, self.psi)
        # convert the solution vector to array and assign it to d_residuals
        d_residuals["dafoam_states"] = DASolver.vec2Array(self.psi)

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

        self.add_subsystem("surface_mesh", DAFoamMesh(solver=DASolver), promotes=["*"])
        self.add_subsystem(
            "volume_mesh",
            DAFoamWarper(solver=DASolver),
            promotes_inputs=[("x_aero", "x_aero0")],
            promotes_outputs=["dafoam_vol_coords"],
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

        # design surface coordinates
        self.x_a0 = self.DASolver.getSurfaceCoordinates(self.DASolver.designFamilyGroup).flatten(order="C")

        # add output
        coord_size = self.x_a0.size
        self.add_output(
            "x_aero0",
            distributed=True,
            shape=coord_size,
            desc="initial aerodynamic surface node coordinates",
            tags=["mphys_coordinates"],
        )

    def mphys_add_coordinate_input(self):
        self.add_input(
            "x_aero0_points", distributed=True, shape_by_conn=True, desc="aerodynamic surface with geom changes"
        )

        # return the promoted name and coordinates
        return "x_aero0_points", self.x_a0

    def mphys_get_surface_mesh(self):
        return self.x_a0

    def mphys_get_triangulated_surface(self, groupName=None):
        # this is a list of lists of 3 points
        # p0, v1, v2

        return self.DASolver.getTriangulatedMeshSurface()

    def compute(self, inputs, outputs):
        # just assign the surface mesh coordinates
        if "x_aero0_points" in inputs:
            outputs["x_aero0"] = inputs["x_aero0_points"]
        else:
            outputs["x_aero0"] = self.x_a0

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        # we do not support forward mode AD
        if mode == "fwd":
            raise AnalysisError("fwd mode not implemented!")

        # just assign the matrix-vector product
        if "x_aero0_points" in d_inputs:
            d_inputs["x_aero0_points"] += d_outputs["x_aero0"]


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

        # init the dv_funcs
        self.dv_funcs = {}

        self.optionDict = None

        self.solution_counter = 0

        # get the dvType dict
        designVariables = self.DASolver.getOption("designVar")
        self.dvType = {}
        for dvName in list(designVariables.keys()):
            self.dvType[dvName] = designVariables[dvName]["designVarType"]

        # setup input and output for the function
        # we need to add states for all cases
        self.add_input("dafoam_states", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])

        # now loop over the design variable keys to determine which other variables we need to add
        shapeVarAdded = False
        for dvName in list(designVariables.keys()):
            dvType = self.dvType[dvName]
            if dvType == "FFD":  # add shape variables
                if shapeVarAdded is False:  # we add the shape variable only once
                    # NOTE: for shape variables, we add dafoam_vol_coords as the input name
                    # the specific name for this shape variable will be added in the geometry component (DVGeo)
                    self.add_input("dafoam_vol_coords", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
                    shapeVarAdded = True
            elif dvType == "AOA":  # add angle of attack variable
                self.add_input(dvName, distributed=False, shape_by_conn=True, tags=["mphys_coupling"])
            elif dvType == "BC":  # add boundary conditions
                self.add_input(dvName, distributed=False, shape_by_conn=True, tags=["mphys_coupling"])
            elif dvType == "ACTD":  # add actuator parameter variables
                self.add_input(dvName, distributed=False, shape=9, tags=["mphys_coupling"])
            elif dvType == "Field":  # add field variables
                self.add_input(dvName, distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
            else:
                raise AnalysisError("designVarType %s not supported! " % dvType)

    # add the function names to this component, called from runScript.py
    def mphys_add_funcs(self, funcs):

        self.funcs = funcs

        # loop over the functions here and create the output
        for f_name in funcs:
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

        DASolver.setStates(inputs["dafoam_states"])

        funcs = {}

        if self.funcs is not None:
            DASolver.evalFunctions(funcs, evalFuncs=self.funcs)
            for f_name in self.funcs:
                if f_name in funcs:
                    outputs[f_name] = funcs[f_name]

    # compute the partial derivatives of functions
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        DASolver = self.DASolver

        # assign the optionDict to the solver
        self.apply_options(self.optionDict)
        # now call the dv_funcs to update the design variables
        for dvName in self.dv_funcs:
            func = self.dv_funcs[dvName]
            dvVal = inputs[dvName]
            func(dvVal, DASolver)
        DASolver.setStates(inputs["dafoam_states"])

        # we do not support forward mode AD
        if mode == "fwd":
            raise AnalysisError("fwd not implemented!")

        funcsBar = {}

        # assign value to funcsBar. NOTE: we only assign seed if d_outputs has
        # non-zero values!
        if self.funcs is None:
            raise AnalysisError("functions not set! Forgot to call mphys_add_funcs?")
        else:
            for func_name in self.funcs:
                if func_name in d_outputs and d_outputs[func_name] != 0.0:
                    funcsBar[func_name] = d_outputs[func_name][0]

        # funcsBar should have only one seed for which we need to compute partials
        # if self.comm.rank == 0:
        #     print(funcsBar)

        # get the name of the functions we need to compute partials for
        objFuncName = list(funcsBar.keys())[0]

        # loop over all d_inputs keys and compute the partials accordingly
        for inputName in list(d_inputs.keys()):

            # compute dFdW
            if inputName == "dafoam_states":
                dFdW = DASolver.wVec.duplicate()
                dFdW.zeroEntries()
                DASolver.solverAD.calcdFdWAD(DASolver.xvVec, DASolver.wVec, objFuncName.encode(), dFdW)
                wBar = DASolver.vec2Array(dFdW)
                d_inputs["dafoam_states"] += wBar

            # compute dFdXv
            elif inputName == "dafoam_vol_coords":
                dFdXv = DASolver.xvVec.duplicate()
                dFdXv.zeroEntries()
                DASolver.solverAD.calcdFdXvAD(
                    DASolver.xvVec, DASolver.wVec, objFuncName.encode(), "dummy".encode(), dFdXv
                )
                xVBar = DASolver.vec2Array(dFdXv)
                d_inputs["dafoam_vol_coords"] += xVBar

            else:  # now we deal with general input input names

                # compute dFdAOA
                if self.dvType[inputName] == "AOA":
                    dFdAOA = PETSc.Vec().create(self.comm)
                    dFdAOA.setSizes((PETSc.DECIDE, 1), bsize=1)
                    dFdAOA.setFromOptions()
                    DASolver.calcdFdAOAAnalytical(objFuncName, dFdAOA)
                    # The aoaBar variable will be length 1 on the root proc, but length 0 an all slave procs.
                    # The value on the root proc must be broadcast across all procs.
                    if self.comm.rank == 0:
                        aoaBar = DASolver.vec2Array(dFdAOA)[0]
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
                        BCBar = DASolver.vec2Array(dFdBC)[0]
                    else:
                        BCBar = 0.0

                    d_inputs[inputName] += self.comm.bcast(BCBar, root=0)

                # compute dFdActD
                elif self.dvType[inputName] == "ACTD":
                    dFdACTD = PETSc.Vec().create(self.comm)
                    dFdACTD.setSizes((PETSc.DECIDE, 9), bsize=1)
                    dFdACTD.setFromOptions()
                    DASolver.solverAD.calcdFdACTAD(
                        DASolver.xvVec, DASolver.wVec, objFuncName.encode(), inputName.encode(), dFdACTD
                    )
                    # we will convert the MPI dFdACTD to seq array for all procs
                    ACTDBar = DASolver.convertMPIVec2SeqArray(dFdACTD)
                    d_inputs[inputName] += ACTDBar

                # compute dFdField
                elif self.dvType[inputName] == "Field":
                    nLocalCells = self.solver.getNLocalCells()
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
                    d_inputs[inputName] += fieldBar

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

        # state inputs and outputs
        local_volume_coord_size = DASolver.mesh.getSolverGrid().size

        self.add_input("x_aero", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_output("dafoam_vol_coords", distributed=True, shape=local_volume_coord_size, tags=["mphys_coupling"])

    def compute(self, inputs, outputs):
        # given the new surface mesh coordinates, compute the new volume mesh coordinates
        # the mesh warping will be called in getSolverGrid()
        DASolver = self.DASolver

        x_a = inputs["x_aero"].reshape((-1, 3))
        DASolver.setSurfaceCoordinates(x_a, DASolver.designFamilyGroup)
        DASolver.mesh.warpMesh()
        solverGrid = DASolver.mesh.getSolverGrid()
        # actually change the mesh in the C++ layer by setting xvVec
        DASolver.xvFlatten2XvVec(solverGrid, DASolver.xvVec)
        outputs["dafoam_vol_coords"] = solverGrid

    # compute the mesh warping products in IDWarp
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        # we do not support forward mode AD
        if mode == "fwd":
            raise AnalysisError("fwd not implemented!")

        # compute dXv/dXs such that we can propagate the partials (e.g., dF/dXv) to Xs
        # then the partial will be further propagated to XFFD in pyGeo
        if "dafoam_vol_coords" in d_outputs:
            if "x_aero" in d_inputs:
                dxV = d_outputs["dafoam_vol_coords"]
                self.DASolver.mesh.warpDeriv(dxV)
                dxS = self.DASolver.mesh.getdXs()
                dxS = self.DASolver.mapVector(dxS, self.DASolver.meshFamilyGroup, self.DASolver.designFamilyGroup)
                d_inputs["x_aero"] += dxS.flatten()


class DAFoamForces(ExplicitComponent):
    """
    OpenMDAO component that wraps force integration

    """

    def initialize(self):
        self.options.declare("solver", recordable=False)

    def setup(self):

        self.DASolver = self.options["solver"]

        self.add_input("dafoam_vol_coords", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_input("dafoam_states", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])

        local_surface_coord_size = self.DASolver.mesh.getSurfaceCoordinates().size
        self.add_output("f_aero", distributed=True, shape=local_surface_coord_size, tags=["mphys_coupling"])

    def compute(self, inputs, outputs):

        self.DASolver.setStates(inputs["dafoam_states"])

        outputs["f_aero"] = self.DASolver.getForces().flatten(order="C")

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        DASolver = self.DASolver

        if mode == "fwd":
            raise AnalysisError("fwd not implemented!")

        if "f_aero" in d_outputs:
            fBar = d_outputs["f_aero"]
            fBarVec = DASolver.array2Vec(fBar)

            if "dafoam_vol_coords" in d_inputs:
                dForcedXv = DASolver.xvVec.duplicate()
                dForcedXv.zeroEntries()
                DASolver.solverAD.calcdForcedXvAD(DASolver.xvVec, DASolver.wVec, fBarVec, dForcedXv)
                xVBar = DASolver.vec2Array(dForcedXv)
                d_inputs["dafoam_vol_coords"] += xVBar
            if "dafoam_states" in d_inputs:
                dForcedW = DASolver.wVec.duplicate()
                dForcedW.zeroEntries()
                DASolver.solverAD.calcdForcedWAD(DASolver.xvVec, DASolver.wVec, fBarVec, dForcedW)
                wBar = DASolver.vec2Array(dForcedW)
                d_inputs["dafoam_states"] += wBar
