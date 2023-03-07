/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Python wrapper library for DASolver

\*---------------------------------------------------------------------------*/

#ifndef DASolvers_H
#define DASolvers_H

#include <petscksp.h>
#include "Python.h"
#include "DASolver.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DASolvers Declaration
\*---------------------------------------------------------------------------*/

class DASolvers
{

private:
    /// Disallow default bitwise copy construct
    DASolvers(const DASolvers&);

    /// Disallow default bitwise assignment
    void operator=(const DASolvers&);

    /// all the arguments
    char* argsAll_;

    /// all options in DAFoam
    PyObject* pyOptions_;

    /// DASolver
    autoPtr<DASolver> DASolverPtr_;

public:
    // Constructors

    /// Construct from components
    DASolvers(
        char* argsAll,
        PyObject* pyOptions);

    /// Destructor
    virtual ~DASolvers();

    /// initialize fields and variables
    void initSolver()
    {
        DASolverPtr_->initSolver();
    }

    /// solve the primal equations
    label solvePrimal(
        const Vec xvVec,
        Vec wVec)
    {
        return DASolverPtr_->solvePrimal(xvVec, wVec);
    }

    /// compute dRdWT
    void calcdRdWT(
        const Vec xvVec,
        const Vec wVec,
        const label isPC,
        Mat dRdWT)
    {
        DASolverPtr_->calcdRdWT(xvVec, wVec, isPC, dRdWT);
    }

    /// compute dFdW
    void calcdFdW(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        Vec dFdW)
    {
        DASolverPtr_->calcdFdW(xvVec, wVec, objFuncName, dFdW);
    }

    /// compute dFdW using reverse-mode AD
    void calcdFdWAD(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        Vec dFdW)
    {
        DASolverPtr_->calcdFdWAD(xvVec, wVec, objFuncName, dFdW);
    }

    /// compute dFdXv using reverse-mode AD
    void calcdFdXvAD(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        const word designVarName,
        Vec dFdXv)
    {
        DASolverPtr_->calcdFdXvAD(xvVec, wVec, objFuncName, designVarName, dFdXv);
    }

    /// compute dRdXv^T*Psi
    void calcdRdXvTPsiAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec psi,
        Vec dRdXvTPsi)
    {
        DASolverPtr_->calcdRdXvTPsiAD(xvVec, wVec, psi, dRdXvTPsi);
    }

    /// compute dForcedXv
    void calcdForcedXvAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec fBarVec,
        Vec dForcedW)
    {
        DASolverPtr_->calcdForcedXvAD(xvVec, wVec, fBarVec, dForcedW);
    }

    /// compute dAcoudXv
    void calcdAcousticsdXvAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec fBarVec,
        Vec dAcoudW,
        word varName,
        word groupName)
    {
        DASolverPtr_->calcdAcousticsdXvAD(xvVec, wVec, fBarVec, dAcoudW, varName, groupName);
    }

    /// compute dRdAct^T*Psi
    void calcdRdActTPsiAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec psi,
        const word designVarName,
        Vec dRdActTPsi)
    {
        DASolverPtr_->calcdRdActTPsiAD(xvVec, wVec, psi, designVarName, dRdActTPsi);
    }

    void calcdThermaldWTPsiAD(
        const word mode,
        const Vec xvVec,
        const Vec wVec,
        const Vec psiVec,
        Vec prodVec)
    {
        DASolverPtr_->calcdThermaldWTPsiAD(mode, xvVec, wVec, psiVec, prodVec);
    }

    void calcdThermaldXvTPsiAD(
        const word mode,
        const Vec xvVec,
        const Vec wVec,
        const Vec psiVec,
        Vec prodVec)
    {
        DASolverPtr_->calcdThermaldXvTPsiAD(mode, xvVec, wVec, psiVec, prodVec);
    }

    /// compute dForcedW
    void calcdForcedWAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec fBarVec,
        Vec dForcedW)
    {
        DASolverPtr_->calcdForcedWAD(xvVec, wVec, fBarVec, dForcedW);
    }

    /// compute dAcoudW
    void calcdAcousticsdWAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec fBarVec,
        Vec dAcoudW,
        word varName,
        word groupName)
    {
        DASolverPtr_->calcdAcousticsdWAD(xvVec, wVec, fBarVec, dAcoudW, varName, groupName);
    }

    /// compute dRdAOA^T*Psi
    void calcdRdAOATPsiAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec psi,
        const word designVarName,
        Vec dRdAOATPsi)
    {
        DASolverPtr_->calcdRdAOATPsiAD(xvVec, wVec, psi, designVarName, dRdAOATPsi);
    }

    /// create the multi-level Richardson KSP for solving linear equation
    void createMLRKSP(
        const Mat jacMat,
        const Mat jacPCMat,
        KSP ksp)
    {
        DASolverPtr_->createMLRKSP(jacMat, jacPCMat, ksp);
    }

    /// create a multi-level, Richardson KSP object with matrix-free Jacobians
    void createMLRKSPMatrixFree(
        const Mat jacPCMat,
        KSP ksp)
    {
        DASolverPtr_->createMLRKSPMatrixFree(jacPCMat, ksp);
    }

    /// initialize matrix free dRdWT
    void initializedRdWTMatrixFree(
        const Vec xvVec,
        const Vec wVec)
    {
        DASolverPtr_->initializedRdWTMatrixFree(xvVec, wVec);
    }

    /// destroy matrix free dRdWT
    void destroydRdWTMatrixFree()
    {
        DASolverPtr_->destroydRdWTMatrixFree();
    }

    /// solve the linear equation
    void solveLinearEqn(
        const KSP ksp,
        const Vec rhsVec,
        Vec solVec)
    {
        DASolverPtr_->solveLinearEqn(ksp, rhsVec, solVec);
    }

    /// convert the mpi vec to a seq vec
    void convertMPIVec2SeqVec(
        const Vec mpiVec,
        Vec seqVec)
    {
        DASolverPtr_->convertMPIVec2SeqVec(mpiVec, seqVec);
    }

    /// compute dRdBC
    void calcdRdBC(
        const Vec xvVec,
        const Vec wVec,
        const word designVarName,
        Mat dRdBC)
    {
        DASolverPtr_->calcdRdBC(xvVec, wVec, designVarName, dRdBC);
    }

    /// compute dFdBC
    void calcdFdBC(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        const word designVarName,
        Vec dFdBC)
    {
        DASolverPtr_->calcdFdBC(xvVec, wVec, objFuncName, designVarName, dFdBC);
    }

    /// compute dFdBCAD
    void calcdFdBCAD(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        const word designVarName,
        Vec dFdBC)
    {
        DASolverPtr_->calcdFdBCAD(xvVec, wVec, objFuncName, designVarName, dFdBC);
    }

    /// compute dRdBC^T*Psi
    void calcdRdBCTPsiAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec psi,
        const word designVarName,
        Vec dRdBCTPsi)
    {
        DASolverPtr_->calcdRdBCTPsiAD(xvVec, wVec, psi, designVarName, dRdBCTPsi);
    }

    /// compute dRdAOA
    void calcdRdAOA(
        const Vec xvVec,
        const Vec wVec,
        const word designVarName,
        Mat dRdAOA)
    {
        DASolverPtr_->calcdRdAOA(xvVec, wVec, designVarName, dRdAOA);
    }

    /// compute dFdAOA
    void calcdFdAOA(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        const word designVarName,
        Vec dFdAOA)
    {
        DASolverPtr_->calcdFdAOA(xvVec, wVec, objFuncName, designVarName, dFdAOA);
    }

    /// compute dRdFFD
    void calcdRdFFD(
        const Vec xvVec,
        const Vec wVec,
        const word designVarName,
        Mat dRdFFD)
    {
        DASolverPtr_->calcdRdFFD(xvVec, wVec, designVarName, dRdFFD);
    }

    /// compute dFdFFD
    void calcdFdFFD(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        const word designVarName,
        Vec dFdFFD)
    {
        DASolverPtr_->calcdFdFFD(xvVec, wVec, objFuncName, designVarName, dFdFFD);
    }

    /// compute dRdACT
    void calcdRdACT(
        const Vec xvVec,
        const Vec wVec,
        const word designVarName,
        const word designVarType,
        Mat dRdACT)
    {
        DASolverPtr_->calcdRdACT(xvVec, wVec, designVarName, designVarType, dRdACT);
    }

    /// compute dFdACT
    void calcdFdACTAD(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        const word designVarName,
        Vec dFdACT)
    {
        DASolverPtr_->calcdFdACTAD(xvVec, wVec, objFuncName, designVarName, dFdACT);
    }

    /// compute dFdACT
    void calcdFdACT(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        const word designVarName,
        const word designVarType,
        Vec dFdACT)
    {
        DASolverPtr_->calcdFdACT(xvVec, wVec, objFuncName, designVarName, designVarType, dFdACT);
    }

    /// compute dRdField
    void calcdRdFieldTPsiAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec psi,
        const word designVarName,
        Vec dRdFieldTPsi)
    {
        DASolverPtr_->calcdRdFieldTPsiAD(xvVec, wVec, psi, designVarName, dRdFieldTPsi);
    }

    /// compute dFdField
    void calcdFdFieldAD(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        const word designVarName,
        Vec dFdField)
    {
        DASolverPtr_->calcdFdFieldAD(xvVec, wVec, objFuncName, designVarName, dFdField);
    }

    /// compute dRdWOld^T*Psi
    void calcdRdWOldTPsiAD(
        const label oldTimeLevel,
        const Vec psi,
        Vec dRdWOldTPsi)
    {
        DASolverPtr_->calcdRdWOldTPsiAD(oldTimeLevel, psi, dRdWOldTPsi);
    }

    /// compute [dRdW]^T*Psi
    void calcdRdWTPsiAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec psi,
        Vec dRdWTPsi)
    {
        DASolverPtr_->calcdRdWTPsiAD(xvVec, wVec, psi, dRdWTPsi);
    }

    /// Update the OpenFOAM field values (including both internal and boundary fields) based on the state vector wVec
    void updateOFField(const Vec wVec)
    {
        DASolverPtr_->updateOFField(wVec);
    }

    /// Update the OpenFoam mesh point coordinates based on the point vector xvVec
    void updateOFMesh(const Vec xvVec)
    {
        DASolverPtr_->updateOFMesh(xvVec);
    }

    /// basically, we call DAIndex::getGlobalXvIndex
    label getGlobalXvIndex(
        const label idxPoint,
        const label idxCoord) const
    {
        return DASolverPtr_->getGlobalXvIndex(idxPoint, idxCoord);
    }

    /// basically, we call DASolver::ofField2StateVec
    void ofField2StateVec(Vec stateVec) const
    {
        DASolverPtr_->ofField2StateVec(stateVec);
    }

    /// basically, we call DASolver::stateVec2OFField
    void stateVec2OFField(const Vec stateVec) const
    {
        DASolverPtr_->stateVec2OFField(stateVec);
    }

    /// basically, we call DASolver::checkMesh
    label checkMesh() const
    {
        return DASolverPtr_->checkMesh();
    }

    /// return the number of local adjoint states
    label getNLocalAdjointStates() const
    {
        return DASolverPtr_->getNLocalAdjointStates();
    }

    /// return the number of local adjoint boundary states
    label getNLocalAdjointBoundaryStates() const
    {
        return DASolverPtr_->getNLocalAdjointBoundaryStates();
    }

    /// return the number of local cells
    label getNLocalCells() const
    {
        return DASolverPtr_->getNLocalCells();
    }

    /// synchronize the values in DAOption and actuatorDiskDVs_
    void syncDAOptionToActuatorDVs()
    {
        DASolverPtr_->syncDAOptionToActuatorDVs();
    }

    /// return the value of objective
    double getObjFuncValue(const word objFuncName)
    {
        double returnVal = 0.0;
        assignValueCheckAD(returnVal, DASolverPtr_->getObjFuncValue(objFuncName));
        return returnVal;
    }

    void getFaceCoords(
        Vec xvVec,
        Vec xsVec)
    {
        DASolverPtr_->getFaceCoords(xvVec, xsVec);
    }

    /// return the force on the FSI patches
    void getForces(Vec fX, Vec fY, Vec fZ)
    {
        DASolverPtr_->getForces(fX, fY, fZ);
    }

    void calcdXvdXsTPsiAD(
        Vec xvVec,
        Vec psi,
        Vec prod)
    {
        DASolverPtr_->calcdXvdXsTPsiAD(xvVec, psi, prod);
    }

    /// return the thermal (temperature or heat flux) on the conjugate heat transfer patches
    void getThermal(word varName, Vec thermalVec)
    {
        DASolverPtr_->getThermal(varName, thermalVec);
    }

    /// set the thermal (temperature or heat flux) BC to the conjugate heat transfer patches
    void setThermal(word varName, double* thermal)
    {
#if !defined(CODI_AD_REVERSE) && !defined(CODI_AD_FORWARD)
        DASolverPtr_->setThermal(varName, thermal);
        return;
#endif
        FatalErrorIn("getForces") << "should not call this func in AD mode!"
                                  << abort(FatalError);
    }

    void calcdRdThermalTPsiAD(
        const word varName,
        const Vec xvVec,
        const Vec wVec,
        const Vec psiVec,
        const Vec thermalVec,
        Vec prodVec)
    {
        DASolverPtr_->calcdRdThermalTPsiAD(varName, xvVec, wVec, psiVec, thermalVec, prodVec);
    }

    /// return the acoustic data values
    void getAcousticData(Vec x, Vec y, Vec z, Vec nX, Vec nY, Vec nZ, Vec a, Vec fX, Vec fY, Vec fZ, word groupName)
    {
        DASolverPtr_->getAcousticData(x, y, z, nX, nY, nZ, a, fX, fY, fZ, groupName);
    }

    /// call DASolver::printAllOptions
    void printAllOptions()
    {
        DASolverPtr_->printAllOptions();
    }

    /// set values for dXvdFFDMat
    void setdXvdFFDMat(const Mat dXvdFFDMat)
    {
        DASolverPtr_->setdXvdFFDMat(dXvdFFDMat);
    }

    /// set the value for DASolver::FFD2XvSeedVec_
    void setFFD2XvSeedVec(Vec vecIn)
    {
        DASolverPtr_->setFFD2XvSeedVec(vecIn);
    }

    /// update the allOptions_ dict in DAOption based on the pyOptions from pyDAFoam
    void updateDAOption(PyObject* pyOptions)
    {
        DASolverPtr_->updateDAOption(pyOptions);
    }

    /// get the solution time folder for previous primal solution
    double getPrevPrimalSolTime()
    {
        double returnVal = 0.0;
        assignValueCheckAD(returnVal, DASolverPtr_->getPrevPrimalSolTime());
        return returnVal;
    }

    /// assign the points in fvMesh of OpenFOAM based on the point vector
    void pointVec2OFMesh(const Vec xvVec) const
    {
        DASolverPtr_->pointVec2OFMesh(xvVec);
    }

    /// assign the point vector based on the points in fvMesh of OpenFOAM
    void ofMesh2PointVec(Vec xvVec) const
    {
        DASolverPtr_->ofMesh2PointVec(xvVec);
    }

    /// assign the OpenFOAM residual fields based on the resVec
    void resVec2OFResField(const Vec resVec) const
    {
        DASolverPtr_->resVec2OFResField(resVec);
    }

    /// assign the resVec based on OpenFOAM residual fields
    void ofResField2ResVec(Vec resVec) const
    {
        DASolverPtr_->resVec2OFResField(resVec);
    }

    /// write the matrix in binary format
    void writeMatrixBinary(
        const Mat matIn,
        const word prefix)
    {
        DASolverPtr_->writeMatrixBinary(matIn, prefix);
    }

    /// write the matrix in ASCII format
    void writeMatrixASCII(
        const Mat matIn,
        const word prefix)
    {
        DASolverPtr_->writeMatrixASCII(matIn, prefix);
    }

    /// read petsc matrix in binary format
    void readMatrixBinary(
        Mat matIn,
        const word prefix)
    {
        DASolverPtr_->readMatrixBinary(matIn, prefix);
    }

    /// write petsc vector in ascii format
    void writeVectorASCII(
        const Vec vecIn,
        const word prefix)
    {
        DASolverPtr_->writeVectorASCII(vecIn, prefix);
    }

    /// read petsc vector in binary format
    void readVectorBinary(
        Vec vecIn,
        const word prefix)
    {
        DASolverPtr_->readVectorBinary(vecIn, prefix);
    }

    /// write petsc vector in binary format
    void writeVectorBinary(
        const Vec vecIn,
        const word prefix)
    {
        DASolverPtr_->writeVectorBinary(vecIn, prefix);
    }

    /// assign primal variables based on the current time instance
    void setTimeInstanceField(const label instanceI)
    {
        DASolverPtr_->setTimeInstanceField(instanceI);
    }

    /// assign the time instance mats to/from the lists in the OpenFOAM layer depending on the mode
    void setTimeInstanceVar(
        const word mode,
        Mat stateMat,
        Mat stateBCMat,
        Vec timeVec,
        Vec timeIdxVec)
    {
        DASolverPtr_->setTimeInstanceVar(mode, stateMat, stateBCMat, timeVec, timeIdxVec);
    }

    /// return the value of objective function at the given time instance and name
    double getTimeInstanceObjFunc(
        const label instanceI,
        const word objFuncName)
    {
        double returnVal = 0.0;
        assignValueCheckAD(returnVal, DASolverPtr_->getTimeInstanceObjFunc(instanceI, objFuncName));
        return returnVal;
    }

    /// set the field value
    void setFieldValue4GlobalCellI(
        const word fieldName,
        const scalar val,
        const label globalCellI,
        const label compI = 0)
    {
        DASolverPtr_->setFieldValue4GlobalCellI(fieldName, val, globalCellI, compI);
    }

    /// set the field value
    void setFieldValue4LocalCellI(
        const word fieldName,
        const scalar val,
        const label localCellI,
        const label compI = 0)
    {
        DASolverPtr_->setFieldValue4LocalCellI(fieldName, val, localCellI, compI);
    }

    /// update the boundary condition for a field
    void updateBoundaryConditions(
        const word fieldName,
        const word fieldType)
    {
        DASolverPtr_->updateBoundaryConditions(fieldName, fieldType);
    }

    /// Calculate the mean, max, and norm2 for all residuals and print it to screen
    void calcPrimalResidualStatistics(const word mode)
    {
        DASolverPtr_->calcPrimalResidualStatistics(mode, 0);
    }

    /// get forwardADDerivVal_
    PetscScalar getForwardADDerivVal(const word objFuncName)
    {
        return DASolverPtr_->getForwardADDerivVal(objFuncName);
    }

    /// calculate the residual and assign it to the resVec vector
    void calcResidualVec(Vec resVec)
    {
        DASolverPtr_->calcResidualVec(resVec);
    }

    void setPrimalBoundaryConditions(const label printInfo = 1)
    {
        DASolverPtr_->setPrimalBoundaryConditions(printInfo);
    }

    /// calculate the fvSource based on the force profile
    void calcFvSource(
        Vec aForce,
        Vec tForce,
        Vec rDistExt,
        Vec targetForce,
        Vec center,
        Vec fvSource)
    {
        DASolverPtr_->calcFvSource(aForce, tForce, rDistExt, targetForce, center, fvSource);
    }

    void calcdFvSourcedInputsTPsiAD(
        const word mode,
        Vec center,
        Vec radius,
        Vec force,
        Vec psi,
        Vec prod)
    {
        DASolverPtr_->calcdFvSourcedInputsTPsiAD(mode, center, radius, force, psi, prod);
    }

    void calcForceProfile(
        Vec center,
        Vec aForceL,
        Vec tForceL,
        Vec rDistL)
    {
        DASolverPtr_->calcForceProfile(center, aForceL, tForceL, rDistL);
    }

    void calcdForcedStateTPsiAD(
        const word mode,
        Vec xvVec,
        Vec stateVec,
        Vec psiVec,
        Vec prodVec)
    {
        DASolverPtr_->calcdForcedStateTPsiAD(mode, xvVec, stateVec, psiVec, prodVec);
    }

    label runFPAdj(
        const Vec xvVec,
        const Vec wVec,
        Vec dFdW,
        Vec psi)
    {
        return DASolverPtr_->runFPAdj(xvVec, wVec, dFdW, psi);
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
