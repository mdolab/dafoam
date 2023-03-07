
# distutils: language = c++
# distutils: sources = DASolvers.C

"""
    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Cython wrapper functions that call OpenFOAM libraries defined
        in the *.C and *.H files. The python naming convention is to
        add "py" before the C++ class name
"""

# for using Petsc
from petsc4py.PETSc cimport Vec, PetscVec, Mat, PetscMat, KSP, PetscKSP
cimport numpy as np

# declare cpp functions
cdef extern from "DASolvers.H" namespace "Foam":
    cppclass DASolvers:
        DASolvers(char *, object) except +
        void initSolver()
        int solvePrimal(PetscVec, PetscVec)
        void calcdRdWT(PetscVec, PetscVec, int, PetscMat)
        void calcdRdWTPsiAD(PetscVec, PetscVec, PetscVec, PetscVec)
        void initializedRdWTMatrixFree(PetscVec, PetscVec)
        void destroydRdWTMatrixFree()
        void calcdFdW(PetscVec, PetscVec, char *, PetscVec)
        void calcdFdWAD(PetscVec, PetscVec, char *, PetscVec)
        void createMLRKSP(PetscMat, PetscMat, PetscKSP)
        void createMLRKSPMatrixFree(PetscMat, PetscKSP)
        void solveLinearEqn(PetscKSP, PetscVec, PetscVec)
        void calcdRdBC(PetscVec, PetscVec, char *, PetscMat)
        void calcdFdBC(PetscVec, PetscVec, char *, char *, PetscVec)
        void calcdFdBCAD(PetscVec, PetscVec, char *, char *, PetscVec)
        void calcdRdAOA(PetscVec, PetscVec, char *, PetscMat)
        void calcdFdAOA(PetscVec, PetscVec, char *, char *, PetscVec)
        void calcdRdFFD(PetscVec, PetscVec, char *, PetscMat)
        void calcdRdXvTPsiAD(PetscVec, PetscVec, PetscVec, PetscVec)
        void calcdForcedXvAD(PetscVec, PetscVec, PetscVec, PetscVec)
        void calcdAcousticsdXvAD(PetscVec, PetscVec, PetscVec, PetscVec, char*, char*)
        void calcdRdActTPsiAD(PetscVec, PetscVec, PetscVec, char*, PetscVec)
        void calcdThermaldWTPsiAD(char * , PetscVec, PetscVec, PetscVec, PetscVec)
        void calcdThermaldXvTPsiAD(char * , PetscVec, PetscVec, PetscVec, PetscVec)
        void calcdForcedWAD(PetscVec, PetscVec, PetscVec, PetscVec)
        void calcdAcousticsdWAD(PetscVec, PetscVec, PetscVec, PetscVec, char*, char*)
        void calcdFdACT(PetscVec, PetscVec, char *, char*, char*, PetscVec)
        void calcdFdACTAD(PetscVec, PetscVec, char *, char*, PetscVec)
        void calcdRdAOATPsiAD(PetscVec, PetscVec, PetscVec, char*, PetscVec)
        void calcdRdBCTPsiAD(PetscVec, PetscVec, PetscVec, char*, PetscVec)
        void calcdFdFFD(PetscVec, PetscVec, char *, char *, PetscVec)
        void calcdFdXvAD(PetscVec, PetscVec, char *, char*, PetscVec)
        void calcdRdACT(PetscVec, PetscVec, char *, char *, PetscMat)
        void calcdRdFieldTPsiAD(PetscVec, PetscVec, PetscVec, char *, PetscVec)
        void calcdFdFieldAD(PetscVec, PetscVec, char *, char *, PetscVec)
        void calcdRdThermalTPsiAD(char *, PetscVec, PetscVec, PetscVec, PetscVec, PetscVec)
        void calcdRdWOldTPsiAD(int, PetscVec, PetscVec)
        void convertMPIVec2SeqVec(PetscVec, PetscVec)
        void syncDAOptionToActuatorDVs()
        void updateOFField(PetscVec)
        void updateOFMesh(PetscVec)
        void setdXvdFFDMat(PetscMat)
        void setFFD2XvSeedVec(PetscVec)
        int getGlobalXvIndex(int, int)
        void ofField2StateVec(PetscVec)
        void stateVec2OFField(PetscVec)
        int getNLocalAdjointStates()
        int getNLocalAdjointBoundaryStates()
        int getNLocalCells()
        int checkMesh()
        double getObjFuncValue(char *)
        void getFaceCoords(PetscVec, PetscVec)
        void getForces(PetscVec, PetscVec, PetscVec)
        void getThermal(char *, PetscVec)
        void setThermal(char *, double *)
        void calcdXvdXsTPsiAD(PetscVec, PetscVec, PetscVec)
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
        double getTimeInstanceObjFunc(int, char *)
        void setFieldValue4GlobalCellI(char *, double, int, int)
        void setFieldValue4LocalCellI(char *, double, int, int)
        void updateBoundaryConditions(char *, char *)
        void calcPrimalResidualStatistics(char *)
        double getForwardADDerivVal(char *)
        void calcResidualVec(PetscVec)
        void setPrimalBoundaryConditions(int)
        void calcFvSource(PetscVec, PetscVec, PetscVec, PetscVec, PetscVec, PetscVec)
        void calcdFvSourcedInputsTPsiAD(char *, PetscVec, PetscVec, PetscVec, PetscVec, PetscVec)
        void calcForceProfile(PetscVec, PetscVec, PetscVec, PetscVec)
        void calcdForcedStateTPsiAD(char *, PetscVec, PetscVec, PetscVec, PetscVec)
        int runFPAdj(PetscVec, PetscVec, PetscVec, PetscVec)
    
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

    def solvePrimal(self, Vec xvVec, Vec wVec):
        return self._thisptr.solvePrimal(xvVec.vec, wVec.vec)
    
    def calcdRdWT(self, Vec xvVec, Vec wVec, isPC, Mat dRdWT):
        self._thisptr.calcdRdWT(xvVec.vec, wVec.vec, isPC, dRdWT.mat)
    
    def calcdRdWTPsiAD(self, Vec xvVec, Vec wVec, Vec psi, Vec dRdWTPsi):
        self._thisptr.calcdRdWTPsiAD(xvVec.vec, wVec.vec, psi.vec, dRdWTPsi.vec)
    
    def initializedRdWTMatrixFree(self, Vec xvVec, Vec wVec):
        self._thisptr.initializedRdWTMatrixFree(xvVec.vec, wVec.vec)
    
    def destroydRdWTMatrixFree(self):
        self._thisptr.destroydRdWTMatrixFree()
    
    def calcdFdW(self, Vec xvVec, Vec wVec, objFuncName, Vec dFdW):
        self._thisptr.calcdFdW(xvVec.vec, wVec.vec, objFuncName, dFdW.vec)
    
    def calcdFdWAD(self, Vec xvVec, Vec wVec, objFuncName, Vec dFdW):
        self._thisptr.calcdFdWAD(xvVec.vec, wVec.vec, objFuncName, dFdW.vec)
    
    def createMLRKSP(self, Mat jacMat, Mat jacPCMat, KSP myKSP):
        self._thisptr.createMLRKSP(jacMat.mat, jacPCMat.mat, myKSP.ksp)
    
    def createMLRKSPMatrixFree(self, Mat jacPCMat, KSP myKSP):
        self._thisptr.createMLRKSPMatrixFree(jacPCMat.mat, myKSP.ksp)
    
    def solveLinearEqn(self, KSP myKSP, Vec rhsVec, Vec solVec):
        self._thisptr.solveLinearEqn(myKSP.ksp, rhsVec.vec, solVec.vec)

    def calcdRdBC(self, Vec xvVec, Vec wVec, designVarName, Mat dRdBC):
        self._thisptr.calcdRdBC(xvVec.vec, wVec.vec, designVarName, dRdBC.mat)
    
    def calcdFdBC(self, Vec xvVec, Vec wVec, objFuncName, designVarName, Vec dFdBC):
        self._thisptr.calcdFdBC(xvVec.vec, wVec.vec, objFuncName, designVarName, dFdBC.vec)
    
    def calcdFdBCAD(self, Vec xvVec, Vec wVec, objFuncName, designVarName, Vec dFdBC):
        self._thisptr.calcdFdBCAD(xvVec.vec, wVec.vec, objFuncName, designVarName, dFdBC.vec)

    def calcdRdAOA(self, Vec xvVec, Vec wVec, designVarName, Mat dRdAOA):
        self._thisptr.calcdRdAOA(xvVec.vec, wVec.vec, designVarName, dRdAOA.mat)

    def calcdFdAOA(self, Vec xvVec, Vec wVec, objFuncName, designVarName, Vec dFdAOA):
        self._thisptr.calcdFdAOA(xvVec.vec, wVec.vec, objFuncName, designVarName, dFdAOA.vec)

    def calcdRdFFD(self, Vec xvVec, Vec wVec, designVarName, Mat dRdFFD):
        self._thisptr.calcdRdFFD(xvVec.vec, wVec.vec, designVarName, dRdFFD.mat)
    
    def calcdRdXvTPsiAD(self, Vec xvVec, Vec wVec, Vec psi, Vec dRdXvTPsi):
        self._thisptr.calcdRdXvTPsiAD(xvVec.vec, wVec.vec, psi.vec, dRdXvTPsi.vec)

    def calcdForcedXvAD(self, Vec xvVec, Vec wVec, Vec fBarVec, Vec dForcedXv):
        self._thisptr.calcdForcedXvAD(xvVec.vec, wVec.vec, fBarVec.vec, dForcedXv.vec)

    def calcdAcousticsdXvAD(self, Vec xvVec, Vec wVec, Vec fBarVec, Vec dAcoudXv, varName, groupName):
        self._thisptr.calcdAcousticsdXvAD(xvVec.vec, wVec.vec, fBarVec.vec, dAcoudXv.vec, varName, groupName)

    def calcdRdActTPsiAD(self, Vec xvVec, Vec wVec, Vec psi, designVarName, Vec dRdActTPsi):
        self._thisptr.calcdRdActTPsiAD(xvVec.vec, wVec.vec, psi.vec, designVarName, dRdActTPsi.vec)
    
    def calcdThermaldWTPsiAD(self, mode, Vec xvVec, Vec wVec, Vec psiVec, Vec prodVec):
        self._thisptr.calcdThermaldWTPsiAD(mode, xvVec.vec, wVec.vec, psiVec.vec, prodVec.vec)
    
    def calcdThermaldXvTPsiAD(self, mode, Vec xvVec, Vec wVec, Vec psiVec, Vec prodVec):
        self._thisptr.calcdThermaldXvTPsiAD(mode, xvVec.vec, wVec.vec, psiVec.vec, prodVec.vec)

    def calcdForcedWAD(self, Vec xvVec, Vec wVec, Vec fBarVec, Vec dForcedW):
        self._thisptr.calcdForcedWAD(xvVec.vec, wVec.vec, fBarVec.vec, dForcedW.vec)

    def calcdAcousticsdWAD(self, Vec xvVec, Vec wVec, Vec fBarVec, Vec dAcoudW, varName, groupName):
        self._thisptr.calcdAcousticsdWAD(xvVec.vec, wVec.vec, fBarVec.vec, dAcoudW.vec, varName, groupName) 

    def calcdFdACTAD(self, Vec xvVec, Vec wVec, objFuncName, designVarName, Vec dFdACT):
        self._thisptr.calcdFdACTAD(xvVec.vec, wVec.vec, objFuncName, designVarName, dFdACT.vec)
    
    def calcdFdACT(self, Vec xvVec, Vec wVec, objFuncName, designVarName, designVarType, Vec dFdACT):
        self._thisptr.calcdFdACT(xvVec.vec, wVec.vec, objFuncName, designVarName, designVarType, dFdACT.vec)
    
    def calcdRdAOATPsiAD(self, Vec xvVec, Vec wVec, Vec psi, designVarName, Vec dRdAOATPsi):
        self._thisptr.calcdRdAOATPsiAD(xvVec.vec, wVec.vec, psi.vec, designVarName, dRdAOATPsi.vec)
    
    def calcdRdBCTPsiAD(self, Vec xvVec, Vec wVec, Vec psi, designVarName, Vec dRdBCTPsi):
        self._thisptr.calcdRdBCTPsiAD(xvVec.vec, wVec.vec, psi.vec, designVarName, dRdBCTPsi.vec)

    def calcdFdFFD(self, Vec xvVec, Vec wVec, objFuncName, designVarName, Vec dFdFFD):
        self._thisptr.calcdFdFFD(xvVec.vec, wVec.vec, objFuncName, designVarName, dFdFFD.vec)

    def calcdFdXvAD(self, Vec xvVec, Vec wVec, objFuncName, designVarName, Vec dFdXv):
        self._thisptr.calcdFdXvAD(xvVec.vec, wVec.vec, objFuncName, designVarName, dFdXv.vec)

    def calcdRdACT(self, Vec xvVec, Vec wVec, designVarName, designVarType, Mat dRdACT):
        self._thisptr.calcdRdACT(xvVec.vec, wVec.vec, designVarName, designVarType, dRdACT.mat)

    def calcdRdFieldTPsiAD(self, Vec xvVec, Vec wVec, Vec psiVec, designVarName, Vec dRdFieldTPsi):
        self._thisptr.calcdRdFieldTPsiAD(xvVec.vec, wVec.vec, psiVec.vec, designVarName, dRdFieldTPsi.vec)

    def calcdFdFieldAD(self, Vec xvVec, Vec wVec, objFuncName, designVarName, Vec dFdField):
        self._thisptr.calcdFdFieldAD(xvVec.vec, wVec.vec, objFuncName, designVarName, dFdField.vec)

    def calcdRdThermalTPsiAD(self, varName, Vec xvVec, Vec wVec, Vec psiVec, Vec thermalVec, Vec prodVec):
        self._thisptr.calcdRdThermalTPsiAD(varName, xvVec.vec, wVec.vec, psiVec.vec, thermalVec.vec, prodVec.vec)
    
    def calcdXvdXsTPsiAD(self, Vec xvVec, Vec psi, Vec prodVec):
        self._thisptr.calcdXvdXsTPsiAD(xvVec.vec, psi.vec, prodVec.vec)
    
    def calcdRdWOldTPsiAD(self, oldTimeLevel, Vec psi, Vec dRdWOldTPsi):
        self._thisptr.calcdRdWOldTPsiAD(oldTimeLevel, psi.vec, dRdWOldTPsi.vec)

    def convertMPIVec2SeqVec(self, Vec mpiVec, Vec seqVec):
        self._thisptr.convertMPIVec2SeqVec(mpiVec.vec, seqVec.vec)
    
    def syncDAOptionToActuatorDVs(self):
        self._thisptr.syncDAOptionToActuatorDVs()

    def updateOFField(self, Vec wVec):
        self._thisptr.updateOFField(wVec.vec)
    
    def updateOFMesh(self, Vec xvVec):
        self._thisptr.updateOFMesh(xvVec.vec)

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
    
    def getNLocalAdjointBoundaryStates(self):
        return self._thisptr.getNLocalAdjointBoundaryStates()
    
    def getNLocalCells(self):
        return self._thisptr.getNLocalCells()
    
    def checkMesh(self):
        return self._thisptr.checkMesh()
    
    def getObjFuncValue(self, objFuncName):
        return self._thisptr.getObjFuncValue(objFuncName)

    def getFaceCoords(self, Vec xvVec, Vec xsVec):
        self._thisptr.getFaceCoords(xvVec.vec, xsVec.vec)

    def getForces(self, Vec fX, Vec fY, Vec fZ):
        self._thisptr.getForces(fX.vec, fY.vec, fZ.vec)
    
    def getThermal(self, varName, Vec thermalVec):
        self._thisptr.getThermal(varName, thermalVec.vec)
    
    def setThermal(self, varName, np.ndarray[double, ndim=1, mode="c"] thermal):
        cdef double *thermal_data = <double*>thermal.data
        self._thisptr.setThermal(varName, thermal_data)

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
    
    def getTimeInstanceObjFunc(self, instanceI, objFuncName):
        return self._thisptr.getTimeInstanceObjFunc(instanceI, objFuncName)

    def setFieldValue4GlobalCellI(self, fieldName, val, globalCellI, compI = 0):
        self._thisptr.setFieldValue4GlobalCellI(fieldName, val, globalCellI, compI)
    
    def setFieldValue4LocalCellI(self, fieldName, val, localCellI, compI = 0):
        self._thisptr.setFieldValue4LocalCellI(fieldName, val, localCellI, compI)
    
    def updateBoundaryConditions(self, fieldName, fieldType):
        self._thisptr.updateBoundaryConditions(fieldName, fieldType)
    
    def calcPrimalResidualStatistics(self, mode):
        self._thisptr.calcPrimalResidualStatistics(mode)
    
    def getForwardADDerivVal(self, objFuncName):
        return self._thisptr.getForwardADDerivVal(objFuncName)
    
    def calcResidualVec(self, Vec resVec):
        self._thisptr.calcResidualVec(resVec.vec)
    
    def setPrimalBoundaryConditions(self, printInfo):
        self._thisptr.setPrimalBoundaryConditions(printInfo)
    
    def calcFvSource(self, Vec aForce, Vec tForce, Vec rDistExt, Vec targetForce, Vec center, Vec fvSource):
        self._thisptr.calcFvSource(aForce.vec, tForce.vec, rDistExt.vec, targetForce.vec, center.vec, fvSource.vec)
    
    def calcdFvSourcedInputsTPsiAD(self, mode, Vec c, Vec r, Vec f, Vec psi, Vec prod):
        self._thisptr.calcdFvSourcedInputsTPsiAD(mode, c.vec, r.vec, f.vec, psi.vec, prod.vec)
    
    def calcForceProfile(self, Vec center, Vec aForceL, Vec tForceL, Vec rDistL):
        self._thisptr.calcForceProfile(center.vec, aForceL.vec, tForceL.vec, rDistL.vec)
    
    def calcdForcedStateTPsiAD(self, mode, Vec xv, Vec state, Vec psi, Vec prod):
        self._thisptr.calcdForcedStateTPsiAD(mode, xv.vec, state.vec, psi.vec, prod.vec)
    
    def runFPAdj(self, Vec xvVec, Vec wVec, Vec dFdW, Vec psi):
        return self._thisptr.runFPAdj(xvVec.vec, wVec.vec, dFdW.vec, psi.vec)
