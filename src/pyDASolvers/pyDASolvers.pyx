
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
        void initializedRdWTMatrixFree()
        void destroydRdWTMatrixFree()
        void createMLRKSPMatrixFree(PetscMat, PetscKSP)
        void updateKSPPCMat(PetscMat, PetscKSP)
        int solveLinearEqn(PetscKSP, PetscVec, PetscVec)
        void calcdRdWOldTPsiAD(int, double *, double *)
        void updateOFFields(double *)
        void getOFFields(double *)
        void getOFField(char *, char *, double *)
        void getOFMeshPoints(double *)
        void updateOFMesh(double *)
        int getGlobalXvIndex(int, int)
        int getNLocalAdjointStates()
        int getNLocalAdjointBoundaryStates()
        int getNLocalCells()
        int getNLocalPoints()
        int checkMesh()
        double getdFScaling(char *, int)
        double getTimeOpFuncVal(char *)
        double getElapsedClockTime()
        double getElapsedCpuTime()
        void calcCouplingFaceCoords(double *, double *)
        int getNRegressionParameters(char *)
        void printAllOptions()
        void updateDAOption(object)
        double getPrevPrimalSolTime()
        void writeFailedMesh()
        void updateBoundaryConditions(char *, char *)
        void updateStateBoundaryConditions()
        void calcPrimalResidualStatistics(char *)
        void setPrimalBoundaryConditions(int)
        int runFPAdj(PetscVec, PetscVec)
        int solveAdjointFP(PetscVec, PetscVec)
        void initTensorFlowFuncs(pyComputeInterface, void *, pyJacVecProdInterface, void *, pySetCharInterface, void *)
        void readStateVars(double, int)
        void readMeshPoints(double)
        void writeMeshPoints(double *, double)
        void calcPCMatWithFvMatrix(PetscMat, int)
        double getEndTime()
        double getDeltaT()
        void setTime(double, int)
        int getDdtSchemeOrder()
        void writeSensMapSurface(char *, double *, double *, int, double)
        void writeSensMapField(char *, double *, char *, double)
        double getLatestTime()
        void writeAdjointFields(char *, double, double *)
        int hasVolCoordInput()
        void meanStatesToStates()
        void updateInputFieldUnsteady()
    
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
    
    def createMLRKSPMatrixFree(self, Mat jacPCMat, KSP myKSP):
        self._thisptr.createMLRKSPMatrixFree(jacPCMat.mat, myKSP.ksp)
    
    def updateKSPPCMat(self, Mat PCMat, KSP myKSP):
        self._thisptr.updateKSPPCMat(PCMat.mat, myKSP.ksp)
    
    def solveLinearEqn(self, KSP myKSP, Vec rhsVec, Vec solVec):
        return self._thisptr.solveLinearEqn(myKSP.ksp, rhsVec.vec, solVec.vec)
    
    def updateOFFields(self, np.ndarray[double, ndim=1, mode="c"] states):
        assert len(states) == self.getNLocalAdjointStates(), "invalid array size!"
        cdef double *states_data = <double*>states.data
        self._thisptr.updateOFFields(states_data)
    
    def getOFFields(self, np.ndarray[double, ndim=1, mode="c"] states):
        assert len(states) == self.getNLocalAdjointStates(), "invalid array size!"
        cdef double *states_data = <double*>states.data
        self._thisptr.getOFFields(states_data)
    
    def getOFMeshPoints(self, np.ndarray[double, ndim=1, mode="c"] points):
        assert len(points) == self.getNLocalPoints() * 3, "invalid array size!"
        cdef double *points_data = <double*>points.data
        self._thisptr.getOFMeshPoints(points_data)
    
    def getOFField(self, fieldName, fieldType, np.ndarray[double, ndim=1, mode="c"] field):
        if fieldType == "scalar":
            assert len(field) == self.getNLocalCells(), "invalid array size!"
        elif fieldType == "vector":
            assert len(field) == self.getNLocalCells() * 3, "invalid array size!"
        cdef double *field_data = <double*>field.data
        self._thisptr.getOFField(fieldName.encode(), fieldType.encode(), field_data)
    
    def updateOFMesh(self, np.ndarray[double, ndim=1, mode="c"] vol_coords):
        assert len(vol_coords) == self.getNLocalPoints() * 3, "invalid array size!"
        cdef double *vol_coords_data = <double*>vol_coords.data
        self._thisptr.updateOFMesh(vol_coords_data)
    
    def getGlobalXvIndex(self, pointI, coordI):
        return self._thisptr.getGlobalXvIndex(pointI, coordI)

    def getNLocalAdjointStates(self):
        return self._thisptr.getNLocalAdjointStates()
    
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

    def getNRegressionParameters(self, modelName):
        return self._thisptr.getNRegressionParameters(modelName)

    def printAllOptions(self):
        self._thisptr.printAllOptions()

    def updateDAOption(self, pyOptions):
        self._thisptr.updateDAOption(pyOptions)
    
    def getPrevPrimalSolTime(self):
        return self._thisptr.getPrevPrimalSolTime()
    
    def writeFailedMesh(self):
        self._thisptr.writeFailedMesh()
    
    def updateBoundaryConditions(self, fieldName, fieldType):
        self._thisptr.updateBoundaryConditions(fieldName.encode(), fieldType.encode())
    
    def updateStateBoundaryConditions(self):
        self._thisptr.updateStateBoundaryConditions()
    
    def calcPrimalResidualStatistics(self, mode):
        self._thisptr.calcPrimalResidualStatistics(mode.encode())
    
    def setPrimalBoundaryConditions(self, printInfo):
        self._thisptr.setPrimalBoundaryConditions(printInfo)
    
    def readStateVars(self, timeVal, timeLevel):
        self._thisptr.readStateVars(timeVal, timeLevel)
    
    def readMeshPoints(self, timeVal):
        self._thisptr.readMeshPoints(timeVal)

    def writeMeshPoints(self, np.ndarray[double, ndim=1, mode="c"] points, timeVal):
        assert len(points) == self.getNLocalPoints() * 3, "invalid array size!"
        cdef double *points_data = <double*>points.data

        self._thisptr.writeMeshPoints(points_data, timeVal)
    
    def calcPCMatWithFvMatrix(self, Mat PCMat, turbOnly=0):
        self._thisptr.calcPCMatWithFvMatrix(PCMat.mat, turbOnly)
    
    def setTime(self, time, timeIndex):
        self._thisptr.setTime(time, timeIndex)

    def getDdtSchemeOrder(self):
        return self._thisptr.getDdtSchemeOrder()
    
    def getEndTime(self):
        return self._thisptr.getEndTime()
    
    def getDeltaT(self):
        return self._thisptr.getDeltaT()

    def runFPAdj(self, Vec dFdW, Vec psi):
        return self._thisptr.runFPAdj(dFdW.vec, psi.vec)
    
    def solveAdjointFP(self, Vec dFdW, Vec psi):
        return self._thisptr.solveAdjointFP(dFdW.vec, psi.vec)
    
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
    
    def hasVolCoordInput(self):
        return self._thisptr.hasVolCoordInput()
    
    def writeAdjointFields(self, function, writeTime, np.ndarray[double, ndim=1, mode="c"] psi):
        nAdjStates = self.getNLocalAdjointStates()

        assert len(psi) == nAdjStates, "invalid array size!"

        cdef double *psi_data = <double*>psi.data

        return self._thisptr.writeAdjointFields(function.encode(), writeTime, psi_data)
    
    def meanStatesToStates(self):
        self._thisptr.meanStatesToStates()
    
    def updateInputFieldUnsteady(self):
        self._thisptr.updateInputFieldUnsteady()
