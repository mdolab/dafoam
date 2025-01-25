
# distutils: language = c++
# distutils: sources = DASolvers.C

"""
    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Cython wrapper functions that call OpenFOAM libraries defined
        in the *.C and *.H files. The python naming convention is to
        add "py" before the C++ class name
"""

# for using Petsc
from petsc4py.PETSc cimport Vec, PetscVec, Mat, PetscMat, KSP, PetscKSP
cimport numpy as np
np.import_array() # initialize C API to call PyArray_SimpleNewFromData

cdef public api CPointerToPyArray(const double* data, int size) with gil:
    if not (data and size >= 0): raise ValueError
    cdef np.npy_intp dims = size
    return np.PyArray_SimpleNewFromData(1, &dims, np.NPY_DOUBLE, <void*>data)

ctypedef void (*pyComputeInterface)(const double *, int, double *, int, void *)
ctypedef void (*pyJacVecProdInterface)(const double *, double *, int, const double *, const double *, int, void *)
ctypedef void (*pySetCharInterface)(const char *, void *)

cdef void pyCalcBetaCallBack(const double* inputs, int n, double* outputs, int m, void *func):
    inputs_data = CPointerToPyArray(inputs, n)
    outputs_data = CPointerToPyArray(outputs, m)
    (<object>func)(inputs_data, n, outputs_data, m)

cdef void pyCalcBetaJacVecProdCallBack(const double* inputs, double* inputs_b, int n, const double* outputs, const double* outputs_b, int m, void *func):
    inputs_data = CPointerToPyArray(inputs, n)
    inputs_b_data = CPointerToPyArray(inputs_b, n)
    outputs_data = CPointerToPyArray(outputs, m)
    outputs_b_data = CPointerToPyArray(outputs_b, m)
    (<object>func)(inputs_data, inputs_b_data, n, outputs_data, outputs_b_data, m)

cdef void pySetModelNameCallBack(const char* modelName, void *func):
    (<object>func)(modelName)

# declare cpp functions
cdef extern from "DASolvers.H" namespace "Foam":
    cppclass DASolvers:
        DASolvers(char *, object) except +
        void initSolver()
        int solvePrimal()
        void runColoring()
        void calcJacTVecProduct(char *, char *, double *, char *, char *, double *, double *)
        int getInputSize(char *, char *)
        int getOutputSize(char *, char *)
        void calcOutput(char *, char *, double *)
        int getInputDistributed(char *, char *)
        int getOutputDistributed(char *, char *)
        void setSolverInput(char *, char *, int, double *, double *)
        void calcdRdWT(int, PetscMat)
        void calcdRdWTPsiAD(PetscVec, PetscVec, PetscVec, PetscVec)
        void initializedRdWTMatrixFree()
        void destroydRdWTMatrixFree()
        void createMLRKSP(PetscMat, PetscMat, PetscKSP)
        void createMLRKSPMatrixFree(PetscMat, PetscKSP)
        void updateKSPPCMat(PetscMat, PetscKSP)
        int solveLinearEqn(PetscKSP, PetscVec, PetscVec)
        void calcdRdWOldTPsiAD(int, double *, double *)
        void convertMPIVec2SeqVec(PetscVec, PetscVec)
        void syncDAOptionToActuatorDVs()
        void updateOFField(PetscVec)
        void updateOFFieldArray(double *)
        void getOFField(double *)
        void updateOFMesh(PetscVec)
        void updateOFMeshArray(double *)
        void setdXvdFFDMat(PetscMat)
        void setFFD2XvSeedVec(PetscVec)
        int getGlobalXvIndex(int, int)
        void ofField2StateVec(PetscVec)
        void stateVec2OFField(PetscVec)
        int getNLocalAdjointStates()
        int getNLocalAdjointBoundaryStates()
        int getNLocalCells()
        int getNLocalPoints()
        int getNCouplingFaces()
        int getNCouplingPoints()
        int checkMesh()
        double getdFScaling(char *, int)
        double getTimeOpFuncVal(char *)
        double getElapsedClockTime()
        double getElapsedCpuTime()
        void calcCouplingFaceCoords(double *, double *)
        void calcCouplingFaceCoordsAD(double *, double *, double *)
        void getForces(PetscVec, PetscVec, PetscVec)
        void getThermal(double *, double *, double *)
        void getThermalAD(char *, double *, double *, double *, double *)
        void setThermal(double *)
        int getNRegressionParameters(char *)
        void setRegressionParameter(char *, int, double)
        void regressionModelCompute()
        void getOFField(char *, char *, PetscVec)
        void getAcousticData(PetscVec, PetscVec, PetscVec, PetscVec, PetscVec, PetscVec, PetscVec, PetscVec, PetscVec, PetscVec, char*)
        void printAllOptions()
        void updateDAOption(object)
        double getPrevPrimalSolTime()
        # functions for unit tests
        void pointVec2OFMesh(PetscVec)
        void ofMesh2PointVec(PetscVec)
        void resVec2OFResField(PetscVec)
        void ofResField2ResVec(PetscVec)
        void writeMatrixBinary(PetscMat, char *)
        void writeMatrixASCII(PetscMat, char *)
        void readMatrixBinary(PetscMat, char *)
        void writeVectorASCII(PetscVec, char *)
        void readVectorBinary(PetscVec, char *)
        void writeVectorBinary(PetscVec, char *)
        void setTimeInstanceField(int)
        void setTimeInstanceVar(char *, PetscMat, PetscMat, PetscVec, PetscVec)
        double getTimeInstanceFunction(int, char *)
        void setFieldValue4GlobalCellI(char *, double, int, int)
        void setFieldValue4LocalCellI(char *, double, int, int)
        void updateBoundaryConditions(char *, char *)
        void updateStateBoundaryConditions()
        void calcPrimalResidualStatistics(char *)
        double getForwardADDerivVal(char *)
        void calcResidualVec(PetscVec)
        void setPrimalBoundaryConditions(int)
        void calcFvSource(char *, PetscVec, PetscVec, PetscVec, PetscVec, PetscVec, PetscVec, PetscVec)
        void calcdFvSourcedInputsTPsiAD(char *, char *, PetscVec, PetscVec, PetscVec, PetscVec, PetscVec, PetscVec, PetscVec, PetscVec)
        void calcForceProfile(char *, PetscVec, PetscVec, PetscVec, PetscVec)
        void calcdForceProfiledXvWAD(char *, char *, char *, PetscVec, PetscVec, PetscVec, PetscVec)
        void calcdForcedStateTPsiAD(char *, PetscVec, PetscVec, PetscVec, PetscVec)
        int runFPAdj(PetscVec, PetscVec, PetscVec, PetscVec)
        void initTensorFlowFuncs(pyComputeInterface, void *, pyJacVecProdInterface, void *, pySetCharInterface, void *)
        void readStateVars(double, int)
        void calcPCMatWithFvMatrix(PetscMat)
        double getEndTime()
        double getDeltaT()
        void setTime(double, int)
        int getDdtSchemeOrder()
        int getUnsteadyFunctionStartTimeIndex()
        int getUnsteadyFunctionEndTimeIndex()
        void writeSensMapSurface(char *, double *, double *, int, double)
        void writeSensMapField(char *, double *, char *, double)
        double getLatestTime()
        void writeAdjointFields(char *, double, double *)
    
# create python wrappers that call cpp functions
cdef class pyDASolvers:

    # define a class pointer for cpp functions
    cdef:
        DASolvers * _thisptr

    # initialize this class pointer with NULL
    def __cinit__(self):
        self._thisptr = NULL

    # deallocate the class pointer, and
    # make sure we don't have memory leak
    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr

    # point the class pointer to the cpp class constructor
    def __init__(self, argsAll, pyOptions):
        """
        Parameters
        ----------

        argsAll: char
            Chars that contains all the arguments
            for running OpenFOAM solvers, including
            the name of the solver.

        pyOptions: dict
            Dictionary that defines all the options
            in DAFoam

        Examples
        --------
        solver = pyDASolvers("DASolvers -parallel -python", aeroOptions)
        """
        self._thisptr = new DASolvers(argsAll, pyOptions)

    # wrap all the other member functions in the cpp class
    def initSolver(self):
        self._thisptr.initSolver()

    def solvePrimal(self):
        return self._thisptr.solvePrimal()
    
    def runColoring(self):
        self._thisptr.runColoring()
    
    def setSolverInput(self,
            inputName,
            inputType,
            inputSize,
            np.ndarray[double, ndim=1, mode="c"] inputs,
            np.ndarray[double, ndim=1, mode="c"] seeds):
        
        assert len(inputs) == inputSize, "invalid input array size!"
        assert len(seeds) == inputSize, "invalid seed array size!"

        cdef double *inputs_data = <double*>inputs.data
        cdef double *seeds_data = <double*>seeds.data

        self._thisptr.setSolverInput(
            inputName.encode(),
            inputType.encode(),
            inputSize,
            inputs_data,
            seeds_data)
    
    def getInputSize(self, inputName, inputType):
        return self._thisptr.getInputSize(inputName.encode(), inputType.encode())
    
    def getOutputSize(self, outputName, outputType):
        return self._thisptr.getOutputSize(outputName.encode(), outputType.encode())
    
    def getInputDistributed(self, inputName, inputType):
        return self._thisptr.getInputDistributed(inputName.encode(), inputType.encode())
    
    def getOutputDistributed(self, outputName, outputType):
        return self._thisptr.getOutputDistributed(outputName.encode(), outputType.encode())

    def calcOutput(self, outputName, outputType, np.ndarray[double, ndim=1, mode="c"] output):
        cdef double *output_data = <double*>output.data
        self._thisptr.calcOutput(outputName.encode(), outputType.encode(), output_data)
    
    def calcJacTVecProduct(self,
            inputName,
            inputType,
            np.ndarray[double, ndim=1, mode="c"] inputs,
            outputName,
            outputType,
            np.ndarray[double, ndim=1, mode="c"] seeds,
            np.ndarray[double, ndim=1, mode="c"] product):

        inputSize = self.getInputSize(inputName, inputType)
        outputSize = self.getOutputSize(outputName, outputType)

        assert len(inputs) == inputSize, "invalid input array size!"
        assert len(seeds) == outputSize, "invalid seed array size!"
        assert len(product) == inputSize, "invalid product array size!"

        cdef double *inputs_data = <double*>inputs.data
        cdef double *seeds_data = <double*>seeds.data
        cdef double *product_data = <double*>product.data

        self._thisptr.calcJacTVecProduct(
            inputName.encode(),
            inputType.encode(),
            inputs_data,
            outputName.encode(),
            outputType.encode(),
            seeds_data, 
            product_data)
    
    def calcdRdWT(self, isPC, Mat dRdWT):
        self._thisptr.calcdRdWT(isPC, dRdWT.mat)
    
    def calcdRdWTPsiAD(self, Vec xvVec, Vec wVec, Vec psi, Vec dRdWTPsi):
        self._thisptr.calcdRdWTPsiAD(xvVec.vec, wVec.vec, psi.vec, dRdWTPsi.vec)
    
    def calcdRdWOldTPsiAD(self, 
        oldTimeLevel, 
        np.ndarray[double, ndim=1, mode="c"] psi, 
        np.ndarray[double, ndim=1, mode="c"] dRdWOldTPsi):

        assert len(psi) == self.getNLocalAdjointStates(), "invalid input array size!"
        assert len(dRdWOldTPsi) == self.getNLocalAdjointStates(), "invalid seed array size!"

        cdef double *psi_data = <double*>psi.data
        cdef double *dRdWOldTPsi_data = <double*>dRdWOldTPsi.data

        self._thisptr.calcdRdWOldTPsiAD(oldTimeLevel, psi_data, dRdWOldTPsi_data)
    
    def initializedRdWTMatrixFree(self):
        self._thisptr.initializedRdWTMatrixFree()
    
    def destroydRdWTMatrixFree(self):
        self._thisptr.destroydRdWTMatrixFree()
    
    def createMLRKSP(self, Mat jacMat, Mat jacPCMat, KSP myKSP):
        self._thisptr.createMLRKSP(jacMat.mat, jacPCMat.mat, myKSP.ksp)
    
    def createMLRKSPMatrixFree(self, Mat jacPCMat, KSP myKSP):
        self._thisptr.createMLRKSPMatrixFree(jacPCMat.mat, myKSP.ksp)
    
    def updateKSPPCMat(self, Mat PCMat, KSP myKSP):
        self._thisptr.updateKSPPCMat(PCMat.mat, myKSP.ksp)
    
    def solveLinearEqn(self, KSP myKSP, Vec rhsVec, Vec solVec):
        return self._thisptr.solveLinearEqn(myKSP.ksp, rhsVec.vec, solVec.vec)

    def convertMPIVec2SeqVec(self, Vec mpiVec, Vec seqVec):
        self._thisptr.convertMPIVec2SeqVec(mpiVec.vec, seqVec.vec)
    
    def syncDAOptionToActuatorDVs(self):
        self._thisptr.syncDAOptionToActuatorDVs()

    def updateOFField(self, Vec wVec):
        self._thisptr.updateOFField(wVec.vec)
    
    def updateOFFieldArray(self, np.ndarray[double, ndim=1, mode="c"] states):
        assert len(states) == self.getNLocalAdjointStates(), "invalid array size!"
        cdef double *states_data = <double*>states.data
        self._thisptr.updateOFFieldArray(states_data)
    
    def getOFField(self, np.ndarray[double, ndim=1, mode="c"] states):
        assert len(states) == self.getNLocalAdjointStates(), "invalid array size!"
        cdef double *states_data = <double*>states.data
        self._thisptr.getOFField(states_data)
    
    def updateOFMesh(self, Vec xvVec):
        self._thisptr.updateOFMesh(xvVec.vec)
    
    def updateOFMeshArray(self, np.ndarray[double, ndim=1, mode="c"] vol_coords):
        assert len(vol_coords) == self.getNLocalPoints() * 3, "invalid array size!"
        cdef double *vol_coords_data = <double*>vol_coords.data
        self._thisptr.updateOFMeshArray(vol_coords_data)

    def setdXvdFFDMat(self, Mat dXvdFFDMat):
        self._thisptr.setdXvdFFDMat(dXvdFFDMat.mat)
    
    def setFFD2XvSeedVec(self, Vec FFD2XvSeedVec):
        self._thisptr.setFFD2XvSeedVec(FFD2XvSeedVec.vec)
    
    def getGlobalXvIndex(self, pointI, coordI):
        return self._thisptr.getGlobalXvIndex(pointI, coordI)
    
    def ofField2StateVec(self, Vec stateVec):
        self._thisptr.ofField2StateVec(stateVec.vec)
    
    def stateVec2OFField(self, Vec stateVec):
        self._thisptr.stateVec2OFField(stateVec.vec)
    
    def pointVec2OFMesh(self, Vec xvVec):
        self._thisptr.pointVec2OFMesh(xvVec.vec)
    
    def ofMesh2PointVec(self, Vec xvVec):
        self._thisptr.ofMesh2PointVec(xvVec.vec)
    
    def resVec2OFResField(self, Vec rVec):
        self._thisptr.resVec2OFResField(rVec.vec)
    
    def ofResField2ResVec(self, Vec rVec):
        self._thisptr.ofResField2ResVec(rVec.vec)
    
    def getNLocalAdjointStates(self):
        return self._thisptr.getNLocalAdjointStates()
    
    def getNCouplingFaces(self):
        return self._thisptr.getNCouplingFaces()
    
    def getNCouplingPoints(self):
        return self._thisptr.getNCouplingPoints()
    
    def getNLocalAdjointBoundaryStates(self):
        return self._thisptr.getNLocalAdjointBoundaryStates()
    
    def getNLocalCells(self):
        return self._thisptr.getNLocalCells()
    
    def getNLocalPoints(self):
        return self._thisptr.getNLocalPoints()
    
    def checkMesh(self):
        return self._thisptr.checkMesh()
    
    def getTimeOpFuncVal(self, functionName):
        return self._thisptr.getTimeOpFuncVal(functionName.encode())
    
    def getdFScaling(self, functionName, timeIdx=-1):
        return self._thisptr.getdFScaling(functionName.encode(), timeIdx)
    
    def getElapsedClockTime(self):
        return self._thisptr.getElapsedClockTime()
    
    def getElapsedCpuTime(self):
        return self._thisptr.getElapsedCpuTime()
        
    def calcCouplingFaceCoords(self, 
            np.ndarray[double, ndim=1, mode="c"] volCoords,
            np.ndarray[double, ndim=1, mode="c"] surfCoords):

        assert len(volCoords) == self.getNLocalPoints() * 3, "invalid array size!"

        cdef double *volCoords_data = <double*>volCoords.data
        cdef double *surfCoords_data = <double*>surfCoords.data

        self._thisptr.calcCouplingFaceCoords(volCoords_data, surfCoords_data)
    
    def calcCouplingFaceCoordsAD(self, 
            np.ndarray[double, ndim=1, mode="c"] volCoords,
            np.ndarray[double, ndim=1, mode="c"] seeds,
            np.ndarray[double, ndim=1, mode="c"] product):

        assert len(volCoords) == self.getNLocalPoints() * 3, "invalid array size!"
        assert len(seeds) == self.getNCouplingFaces() * 6, "invalid array size!"
        assert len(product) == self.getNLocalPoints() * 3, "invalid array size!"

        cdef double *volCoords_data = <double*>volCoords.data
        cdef double *seeds_data = <double*>seeds.data
        cdef double *product_data = <double*>product.data

        self._thisptr.calcCouplingFaceCoordsAD(volCoords_data, seeds_data, product_data)

    def getForces(self, Vec fX, Vec fY, Vec fZ):
        self._thisptr.getForces(fX.vec, fY.vec, fZ.vec)
    
    def getOFField(self, fieldName, fieldType, Vec fieldVec):
        self._thisptr.getOFField(fieldName, fieldType, fieldVec.vec)
    
    def getThermal(self, 
            np.ndarray[double, ndim=1, mode="c"] volCoords,
            np.ndarray[double, ndim=1, mode="c"] states,
            np.ndarray[double, ndim=1, mode="c"] thermal):
        
        assert len(volCoords) == self.getNLocalPoints() * 3, "invalid array size!"
        assert len(states) == self.getNLocalAdjointStates(), "invalid array size!"
        assert len(thermal) == self.getNCouplingFaces() * 2, "invalid array size!"

        cdef double *volCoords_data = <double*>volCoords.data
        cdef double *states_data = <double*>states.data
        cdef double *thermal_data = <double*>thermal.data

        self._thisptr.getThermal(volCoords_data, states_data, thermal_data)
    
    def getThermalAD(self,
            inputName,
            np.ndarray[double, ndim=1, mode="c"] volCoords,
            np.ndarray[double, ndim=1, mode="c"] states,
            np.ndarray[double, ndim=1, mode="c"] seeds,
            np.ndarray[double, ndim=1, mode="c"] product):
        
        assert len(volCoords) == self.getNLocalPoints() * 3, "invalid array size!"
        assert len(states) == self.getNLocalAdjointStates(), "invalid array size!"
        assert len(seeds) == self.getNCouplingFaces() * 2, "invalid array size!"
        if inputName == "volCoords":
            assert len(product) == self.getNLocalPoints() * 3, "invalid array size!"
        elif inputName == "states":
            assert len(product) == self.getNLocalAdjointStates(), "invalid array size!"
        else:
            print("invalid inputName!")
            exit(1)

        cdef double *volCoords_data = <double*>volCoords.data
        cdef double *states_data = <double*>states.data
        cdef double *seeds_data = <double*>seeds.data
        cdef double *product_data = <double*>product.data

        self._thisptr.getThermalAD(
            inputName.encode(), 
            volCoords_data, 
            states_data, 
            seeds_data, 
            product_data)
    
    def setThermal(self, 
            np.ndarray[double, ndim=1, mode="c"] thermal):

        assert len(thermal) == self.getNCouplingFaces() * 2, "invalid array size!"
        cdef double *thermal_data = <double*>thermal.data
        self._thisptr.setThermal(thermal_data)
    
    def getNRegressionParameters(self, modelName):
        return self._thisptr.getNRegressionParameters(modelName)
    
    def setRegressionParameter(self, modelName, idx, val):
        self._thisptr.setRegressionParameter(modelName, idx, val)
    
    def regressionModelCompute(self):
        self._thisptr.regressionModelCompute()

    def getAcousticData(self, Vec x, Vec y, Vec z, Vec nX, Vec nY, Vec nZ, Vec a, Vec fX, Vec fY, Vec fZ, groupName):
        self._thisptr.getAcousticData(x.vec, y.vec, z.vec, nX.vec, nY.vec, nZ.vec, a.vec, fX.vec, fY.vec, fZ.vec, groupName)

    def getAcousticData(self, Vec x, Vec y, Vec z, Vec nX, Vec nY, Vec nZ, Vec a, Vec fX, Vec fY, Vec fZ, groupName):
        self._thisptr.getAcousticData(x.vec, y.vec, z.vec, nX.vec, nY.vec, nZ.vec, a.vec, fX.vec, fY.vec, fZ.vec, groupName)

    def printAllOptions(self):
        self._thisptr.printAllOptions()

    def updateDAOption(self, pyOptions):
        self._thisptr.updateDAOption(pyOptions)
    
    def getPrevPrimalSolTime(self):
        return self._thisptr.getPrevPrimalSolTime()
    
    def writeMatrixBinary(self, Mat magIn, prefix):
        self._thisptr.writeMatrixBinary(magIn.mat, prefix)
    
    def writeMatrixASCII(self, Mat magIn, prefix):
        self._thisptr.writeMatrixASCII(magIn.mat, prefix)
    
    def readMatrixBinary(self, Mat magIn, prefix):
        self._thisptr.readMatrixBinary(magIn.mat, prefix)
    
    def writeVectorASCII(self, Vec vecIn, prefix):
        self._thisptr.writeVectorASCII(vecIn.vec, prefix)
    
    def readVectorBinary(self, Vec vecIn, prefix):
        self._thisptr.readVectorBinary(vecIn.vec, prefix)
    
    def writeVectorBinary(self, Vec vecIn, prefix):
        self._thisptr.writeVectorBinary(vecIn.vec, prefix)
    
    def setTimeInstanceField(self, instanceI):
        self._thisptr.setTimeInstanceField(instanceI)
    
    def setTimeInstanceVar(self, mode, Mat stateMat, Mat stateBCMat, Vec timeVec, Vec timeIdxVec):
        self._thisptr.setTimeInstanceVar(mode, stateMat.mat, stateBCMat.mat, timeVec.vec, timeIdxVec.vec)
    
    def getTimeInstanceFunction(self, instanceI, functionName):
        return self._thisptr.getTimeInstanceFunction(instanceI, functionName)

    def setFieldValue4GlobalCellI(self, fieldName, val, globalCellI, compI = 0):
        self._thisptr.setFieldValue4GlobalCellI(fieldName, val, globalCellI, compI)
    
    def setFieldValue4LocalCellI(self, fieldName, val, localCellI, compI = 0):
        self._thisptr.setFieldValue4LocalCellI(fieldName, val, localCellI, compI)
    
    def updateBoundaryConditions(self, fieldName, fieldType):
        self._thisptr.updateBoundaryConditions(fieldName, fieldType)
    
    def updateStateBoundaryConditions(self):
        self._thisptr.updateStateBoundaryConditions()
    
    def calcPrimalResidualStatistics(self, mode):
        self._thisptr.calcPrimalResidualStatistics(mode.encode())
    
    def getForwardADDerivVal(self, functionName):
        return self._thisptr.getForwardADDerivVal(functionName)
    
    def calcResidualVec(self, Vec resVec):
        self._thisptr.calcResidualVec(resVec.vec)
    
    def setPrimalBoundaryConditions(self, printInfo):
        self._thisptr.setPrimalBoundaryConditions(printInfo)
    
    def readStateVars(self, timeVal, timeLevel):
        self._thisptr.readStateVars(timeVal, timeLevel)
    
    def calcPCMatWithFvMatrix(self, Mat PCMat):
        self._thisptr.calcPCMatWithFvMatrix(PCMat.mat)
    
    def setTime(self, time, timeIndex):
        self._thisptr.setTime(time, timeIndex)

    def getDdtSchemeOrder(self):
        return self._thisptr.getDdtSchemeOrder()
    
    def getEndTime(self):
        return self._thisptr.getEndTime()
    
    def getDeltaT(self):
        return self._thisptr.getDeltaT()
    
    def getUnsteadyFunctionStartTimeIndex(self):
        return self._thisptr.getUnsteadyFunctionStartTimeIndex()
    
    def getUnsteadyFunctionEndTimeIndex(self):
        return self._thisptr.getUnsteadyFunctionEndTimeIndex()
    
    def calcFvSource(self, propName, Vec aForce, Vec tForce, Vec rDist, Vec targetForce, Vec center, Vec xvVec, Vec fvSource):
        self._thisptr.calcFvSource(propName, aForce.vec, tForce.vec, rDist.vec, targetForce.vec, center.vec, xvVec.vec, fvSource.vec)
    
    def calcdFvSourcedInputsTPsiAD(self, propName, mode, Vec aForce, Vec tForce, Vec rDist, Vec targetForce, Vec center, Vec xvVec, Vec psi, Vec dFvSource):
        self._thisptr.calcdFvSourcedInputsTPsiAD(propName, mode, aForce.vec, tForce.vec, rDist.vec, targetForce.vec, center.vec, xvVec.vec, psi.vec, dFvSource.vec)
    
    def calcForceProfile(self, propName, Vec aForce, Vec tForce, Vec rDist, Vec integralForce):
        self._thisptr.calcForceProfile(propName, aForce.vec, tForce.vec, rDist.vec, integralForce.vec)

    def calcdForceProfiledXvWAD(self, propName, inputMode, outputMode, Vec xvVec, Vec wVec, Vec psi, Vec dForcedXvW):
        self._thisptr.calcdForceProfiledXvWAD(propName, inputMode, outputMode, xvVec.vec, wVec.vec, psi.vec, dForcedXvW.vec)
    
    def calcdForcedStateTPsiAD(self, mode, Vec xv, Vec state, Vec psi, Vec prod):
        self._thisptr.calcdForcedStateTPsiAD(mode, xv.vec, state.vec, psi.vec, prod.vec)
    
    def runFPAdj(self, Vec xvVec, Vec wVec, Vec dFdW, Vec psi):
        return self._thisptr.runFPAdj(xvVec.vec, wVec.vec, dFdW.vec, psi.vec)
    
    def initTensorFlowFuncs(self, compute, jacVecProd, setModelName):
        self._thisptr.initTensorFlowFuncs(pyCalcBetaCallBack, <void*>compute, pyCalcBetaJacVecProdCallBack, <void*>jacVecProd, pySetModelNameCallBack, <void*>setModelName)
    
    def writeSensMapSurface(self, 
            name,
            np.ndarray[double, ndim=1, mode="c"] dFdXs,
            np.ndarray[double, ndim=1, mode="c"] Xs,
            size,
            timeName):
        
        assert len(dFdXs) == size, "invalid array size!"
        assert len(Xs) == size, "invalid array size!"

        cdef double *dFdXs_data = <double*>dFdXs.data
        cdef double *Xs_data = <double*>Xs.data

        self._thisptr.writeSensMapSurface(
            name.encode(), 
            dFdXs_data, 
            Xs_data, 
            size,
            timeName)
    
    def writeSensMapField(self, 
            name,
            np.ndarray[double, ndim=1, mode="c"] dFdField,
            fieldType,
            timeName):
        
        nCells = self.getNLocalCells()
        if fieldType == "scalar":
            assert len(dFdField) == nCells, "invalid array size!"
        elif fieldType == "vector":
            assert len(dFdField) == 3 * nCells, "invalid array size!"
        else:
            print("fieldType can be either scalar or vector")
            exit(1)

        cdef double *dFdField_data = <double*>dFdField.data

        self._thisptr.writeSensMapField(
            name.encode(), 
            dFdField_data, 
            fieldType.encode(), 
            timeName)
    
    def getLatestTime(self):
        return self._thisptr.getLatestTime()
    
    def writeAdjointFields(self, function, writeTime, np.ndarray[double, ndim=1, mode="c"] psi):
        nAdjStates = self.getNLocalAdjointStates()

        assert len(psi) == nAdjStates, "invalid array size!"

        cdef double *psi_data = <double*>psi.data

        return self._thisptr.writeAdjointFields(function.encode(), writeTime, psi_data)
