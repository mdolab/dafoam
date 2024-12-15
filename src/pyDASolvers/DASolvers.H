/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

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
    label solvePrimal()
    {
        return DASolverPtr_->solvePrimal();
    }

    /// calculate the Jacobian-matrix-transposed and vector product for product = [dOutput/dInput]^T * seed
    void calcJacTVecProduct(
        const word inputName,
        const int inputSize,
        const int distributedInput,
        const double* input,
        const word outputName,
        const int outputSize,
        const int distributedOutput,
        const double* seed,
        double* product)
    {
        DASolverPtr_->calcJacTVecProduct(
            inputName,
            inputSize,
            distributedInput,
            input,
            outputName,
            outputSize,
            distributedOutput,
            seed,
            product);
    }

    /// compute dRdWT
    void calcdRdWT(
        const label isPC,
        Mat dRdWT)
    {
        DASolverPtr_->calcdRdWT(isPC, dRdWT);
    }

    /// compute dFdW using reverse-mode AD
    void calcdFdWAD(
        const Vec xvVec,
        const Vec wVec,
        const word functionName,
        Vec dFdW)
    {
        DASolverPtr_->calcdFdWAD(xvVec, wVec, functionName, dFdW);
    }

    /// compute dFdXv using reverse-mode AD
    void calcdFdXvAD(
        const Vec xvVec,
        const Vec wVec,
        const word functionName,
        const word designVarName,
        Vec dFdXv)
    {
        DASolverPtr_->calcdFdXvAD(xvVec, wVec, functionName, designVarName, dFdXv);
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

    /// compute dRdHSC^T*Psi
    void calcdRdHSCTPsiAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec psi,
        const word designVarName,
        Vec dRdHSCTPsi)
    {
        DASolverPtr_->calcdRdHSCTPsiAD(xvVec, wVec, psi, designVarName, dRdHSCTPsi);
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

    /// Update the preconditioner matrix for the ksp object
    void updateKSPPCMat(
        Mat PCMat,
        KSP ksp)
    {
        DASolverPtr_->updateKSPPCMat(PCMat, ksp);
    }

    /// create a multi-level, Richardson KSP object with matrix-free Jacobians
    void createMLRKSPMatrixFree(
        const Mat jacPCMat,
        KSP ksp)
    {
        DASolverPtr_->createMLRKSPMatrixFree(jacPCMat, ksp);
    }

    /// initialize matrix free dRdWT
    void initializedRdWTMatrixFree()
    {
        DASolverPtr_->initializedRdWTMatrixFree();
    }

    /// destroy matrix free dRdWT
    void destroydRdWTMatrixFree()
    {
        DASolverPtr_->destroydRdWTMatrixFree();
    }

    /// solve the linear equation
    label solveLinearEqn(
        const KSP ksp,
        const Vec rhsVec,
        Vec solVec)
    {
        return DASolverPtr_->solveLinearEqn(ksp, rhsVec, solVec);
    }

    /// convert the mpi vec to a seq vec
    void convertMPIVec2SeqVec(
        const Vec mpiVec,
        Vec seqVec)
    {
        DASolverPtr_->convertMPIVec2SeqVec(mpiVec, seqVec);
    }

    /// compute dFdBCAD
    void calcdFdBCAD(
        const Vec xvVec,
        const Vec wVec,
        const word functionName,
        const word designVarName,
        Vec dFdBC)
    {
        DASolverPtr_->calcdFdBCAD(xvVec, wVec, functionName, designVarName, dFdBC);
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

    /// compute dFdACT
    void calcdFdACTAD(
        const Vec xvVec,
        const Vec wVec,
        const word functionName,
        const word designVarName,
        Vec dFdACT)
    {
        DASolverPtr_->calcdFdACTAD(xvVec, wVec, functionName, designVarName, dFdACT);
    }

    /// compute dFdHSC
    void calcdFdHSCAD(
        const Vec xvVec,
        const Vec wVec,
        const word functionName,
        const word designVarName,
        Vec dFdHSC)
    {
        DASolverPtr_->calcdFdHSCAD(xvVec, wVec, functionName, designVarName, dFdHSC);
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
        const word functionName,
        const word designVarName,
        Vec dFdField)
    {
        DASolverPtr_->calcdFdFieldAD(xvVec, wVec, functionName, designVarName, dFdField);
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

    /// Update the OpenFOAM field values (including both internal and boundary fields) based on the states array
    void updateOFFieldArray(const double* states)
    {
#if !defined(CODI_ADR) && !defined(CODI_ADF)
        DASolverPtr_->updateOFField(states);
#else
        label localSize = this->getNLocalAdjointStates();
        scalar* statesArray = new scalar[localSize];
        for (label i = 0; i < localSize; i++)
        {
            statesArray[i] = states[i];
        }
        DASolverPtr_->updateOFField(statesArray);
        delete[] statesArray;
#endif
    }

    /// Assign the OpenFOAM field values to the states array
    void getOFField(double* states)
    {
#if !defined(CODI_ADR) && !defined(CODI_ADF)
        DASolverPtr_->getOFField(states);
#else
        label localSize = this->getNLocalAdjointStates();
        scalar* statesArray = new scalar[localSize];
        DASolverPtr_->getOFField(statesArray);
        for (label i = 0; i < localSize; i++)
        {
            states[i] = statesArray[i].value();
        }
        delete[] statesArray;
#endif
    }

    /// Update the OpenFoam mesh point coordinates based on the point vector xvVec
    void updateOFMesh(const Vec xvVec)
    {
        DASolverPtr_->updateOFMesh(xvVec);
    }

    /// Update the OpenFoam mesh point coordinates based on the point vector xvVec
    void updateOFMeshArray(const double* volCoords)
    {
#if !defined(CODI_ADR) && !defined(CODI_ADF)
        DASolverPtr_->updateOFMesh(volCoords);
#else
        label localSize = this->getNLocalPoints() * 3;
        scalar* volCoordsArray = new scalar[localSize];
        for (label i = 0; i < localSize; i++)
        {
            volCoordsArray[i] = volCoords[i];
        }
        DASolverPtr_->updateOFMesh(volCoordsArray);
        delete[] volCoordsArray;
#endif
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

    /// Get the number of faces for the MDO coupling patches
    label getNCouplingFaces()
    {
        return DASolverPtr_->getNCouplingFaces();
    }

    /// Get the number of points for the MDO coupling patches
    label getNCouplingPoints()
    {
        return DASolverPtr_->getNCouplingPoints();
    }

    // return the number of local mesh points
    label getNLocalPoints() const
    {
        return DASolverPtr_->getNLocalPoints();
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
    double getFunctionValue(const word functionName)
    {
        double returnVal = 0.0;
        assignValueCheckAD(returnVal, DASolverPtr_->getFunctionValue(functionName));
        return returnVal;
    }

    double getFunctionValueUnsteady(const word functionName)
    {
        double returnVal = 0.0;
        assignValueCheckAD(returnVal, DASolverPtr_->getFunctionValueUnsteady(functionName));
        return returnVal;
    }

    double getFunctionUnsteadyScaling()
    {
        double returnVal = 0.0;
        assignValueCheckAD(returnVal, DASolverPtr_->getFunctionUnsteadyScaling());
        return returnVal;
    }

    /// return the elapsed clock time for testing speed
    double getElapsedClockTime()
    {
        double returnVal = 0.0;
        assignValueCheckAD(returnVal, DASolverPtr_->getElapsedClockTime());
        return returnVal;
    }

    /// return the elapsed CPU time for testing speed
    double getElapsedCpuTime()
    {
        double returnVal = 0.0;
        assignValueCheckAD(returnVal, DASolverPtr_->getElapsedCpuTime());
        return returnVal;
    }

    void calcCouplingFaceCoords(
        const double* volCoords,
        double* surfCoords)
    {
#if !defined(CODI_ADR) && !defined(CODI_ADF)
        DASolverPtr_->calcCouplingFaceCoords(volCoords, surfCoords);
        return;
#endif
        FatalErrorIn("calcCouplingFaceCoords") << "should not call this func in AD mode!"
                                               << abort(FatalError);
    }

    /// calc matrix-vector products for calcCouplingFaceCoords
    void calcCouplingFaceCoordsAD(
        const double* volCoords,
        const double* seeds,
        double* product)
    {
        DASolverPtr_->calcCouplingFaceCoordsAD(volCoords, seeds, product);
    }

    /// return the force on the FSI patches
    void getForces(Vec fX, Vec fY, Vec fZ)
    {
        DASolverPtr_->getForces(fX, fY, fZ);
    }

    /// return the thermal (temperature or heat flux) on the conjugate heat transfer patches
    void getThermal(
        const double* volCoords,
        const double* states,
        double* thermal)
    {
#if !defined(CODI_ADR) && !defined(CODI_ADF)
        DASolverPtr_->getThermal(volCoords, states, thermal);
        return;
#endif
        FatalErrorIn("getThermal") << "should not call this func in AD mode!"
                                   << abort(FatalError);
    }

    /// compute the temperature on the conjugate heat transfer patches AD
    void getThermalAD(
        const word inputName,
        const double* volCoords,
        const double* states,
        const double* seeds,
        double* product)
    {
        DASolverPtr_->getThermalAD(inputName, volCoords, states, seeds, product);
    }

    /// set the thermal (temperature or heat flux) BC to the conjugate heat transfer patches
    void setThermal(double* thermal)
    {
        label nCouplingFaces = this->getNCouplingFaces();
        scalar* thermalArray = new scalar[nCouplingFaces * 2];
        for (label i = 0; i < nCouplingFaces * 2; i++)
        {
            thermalArray[i] = thermal[i];
        }

        DASolverPtr_->setThermal(thermalArray);
        delete[] thermalArray;
    }

    /// get the number of regression model parameters
    label getNRegressionParameters(word modelName)
    {
        return DASolverPtr_->getNRegressionParameters(modelName);
    }

    /// call the compute method of the regression model
    void regressionModelCompute()
    {
        DASolverPtr_->regressionModelCompute();
    }

    /// set the regression parameter
    void setRegressionParameter(word modelName, label idxI, double val)
    {
        scalar val1 = val;
        DASolverPtr_->setRegressionParameter(modelName, idxI, val1);
    }

    /// calculte [dR/dRegParameters]^T * psi
    void calcdRdRegParTPsiAD(
        const double* volCoords,
        const double* states,
        const double* parameters,
        const double* seeds,
        const word modelName,
        double* product)
    {
        DASolverPtr_->calcdRdRegParTPsiAD(volCoords, states, parameters, seeds, modelName, product);
    }

    /// compute dFdRegPar AD
    void calcdFdRegParAD(
        const double* volCoords,
        const double* states,
        const double* parameters,
        const word functionName,
        const word designVarName,
        const word modelName,
        double* dFdRegPar)
    {
        DASolverPtr_->calcdFdRegParAD(volCoords, states, parameters, functionName, designVarName, modelName, dFdRegPar);
    }

    void calcdRdThermalTPsiAD(
        const double* volCoords,
        const double* states,
        const double* thermal,
        const double* seeds,
        double* product)
    {
        DASolverPtr_->calcdRdThermalTPsiAD(volCoords, states, thermal, seeds, product);
    }

    /// get a variable from OF layer
    void getOFField(
        const word fieldName,
        const word fieldType,
        Vec field)
    {
        DASolverPtr_->getOFField(fieldName, fieldType, field);
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

    /// read the state variables from the disk and assign the value to the prescribe time level
    void readStateVars(
        scalar timeVal,
        label timeLevel = 0)
    {
        DASolverPtr_->readStateVars(timeVal, timeLevel);
    }

    /// calculate the PC mat using fvMatrix
    void calcPCMatWithFvMatrix(Mat PCMat)
    {
        DASolverPtr_->calcPCMatWithFvMatrix(PCMat);
    }

    /// setTime for OF fields
    void setTime(scalar time, label timeIndex)
    {
        DASolverPtr_->setTime(time, timeIndex);
    }

    /// get the ddtScheme order
    label getDdtSchemeOrder()
    {
        return DASolverPtr_->getDdtSchemeOrder();
    }

    /// return the endTime
    double getEndTime()
    {
#if !defined(CODI_ADR) && !defined(CODI_ADF)
        return DASolverPtr_->getRunTime().endTime().value();
#endif
        FatalErrorIn("getEndTime") << "should not call this func in AD mode!"
                                   << abort(FatalError);
    }

    /// return the deltaT
    double getDeltaT()
    {
#if !defined(CODI_ADR) && !defined(CODI_ADF)
        return DASolverPtr_->getRunTime().deltaT().value();
#endif
        FatalErrorIn("getDeltaT") << "should not call this func in AD mode!"
                                  << abort(FatalError);
    }

    /// get the unsteady adjoint start time
    int getUnsteadyFunctionStartTimeIndex()
    {
#if !defined(CODI_ADR) && !defined(CODI_ADF)
        return DASolverPtr_->getUnsteadyFunctionStartTimeIndex();
#endif
        FatalErrorIn("getUnsteadyFunctionStartTimeIndex") << "should not call this func in AD mode!"
                                                          << abort(FatalError);
    }

    /// get the unsteady adjoint end time
    int getUnsteadyFunctionEndTimeIndex()
    {
#if !defined(CODI_ADR) && !defined(CODI_ADF)
        return DASolverPtr_->getUnsteadyFunctionEndTimeIndex();
#endif
        FatalErrorIn("getUnsteadyFunctionEndTimeIndex") << "should not call this func in AD mode!"
                                                        << abort(FatalError);
    }

    /// return the value of objective function at the given time instance and name
    double getTimeInstanceFunction(
        const label instanceI,
        const word functionName)
    {
        double returnVal = 0.0;
        assignValueCheckAD(returnVal, DASolverPtr_->getTimeInstanceFunction(instanceI, functionName));
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

    /// update the boundary conditions for all states and intermediate variables
    void updateStateBoundaryConditions()
    {
        DASolverPtr_->updateStateBoundaryConditions();
    }

    /// Calculate the mean, max, and norm2 for all residuals and print it to screen
    void calcPrimalResidualStatistics(const word mode)
    {
        DASolverPtr_->calcPrimalResidualStatistics(mode, 0);
    }

    /// get forwardADDerivVal_
    PetscScalar getForwardADDerivVal(const word functionName)
    {
        return DASolverPtr_->getForwardADDerivVal(functionName);
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
        const word propName,
        Vec aForce,
        Vec tForce,
        Vec rDist,
        Vec targetForce,
        Vec center,
        Vec xvVec,
        Vec fvSource)
    {
        DASolverPtr_->calcFvSource(propName, aForce, tForce, rDist, targetForce, center, xvVec, fvSource);
    }

    void calcdFvSourcedInputsTPsiAD(
        const word propName,
        const word mode,
        Vec aForce,
        Vec tForce,
        Vec rDist,
        Vec targetForce,
        Vec center,
        Vec xvVec,
        Vec psi,
        Vec dFvSource)
    {
        DASolverPtr_->calcdFvSourcedInputsTPsiAD(propName, mode, aForce, tForce, rDist, targetForce, center, xvVec, psi, dFvSource);
    }

    void calcForceProfile(
        const word propName,
        Vec aForce,
        Vec tForce,
        Vec rDist,
        Vec integralForce)
    {
        DASolverPtr_->calcForceProfile(propName, aForce, tForce, rDist, integralForce);
    }

    void calcdForceProfiledXvWAD(
        const word propName,
        const word inputMode,
        const word outputMode,
        const Vec xvVec,
        const Vec wVec,
        Vec psi,
        Vec dForcedXvW)
    {
        DASolverPtr_->calcdForceProfiledXvWAD(propName, inputMode, outputMode, xvVec, wVec, psi, dForcedXvW);
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

    /// initialize the call back function pointer
    void initTensorFlowFuncs(
        pyComputeInterface computeInterface,
        void* compute,
        pyJacVecProdInterface jacVecProdInterface,
        void* jacVecProd,
        pySetCharInterface setModelNameInterface,
        void* setModelName)
    {
        DASolverPtr_->initTensorFlowFuncs(
            computeInterface, compute, jacVecProdInterface, jacVecProd, setModelNameInterface, setModelName);
    }

    /// write the adjoint variables for all states
    void writeAdjointFields(
        const word function,
        const double writeTime,
        const double* psi)
    {
        DASolverPtr_->writeAdjointFields(function, writeTime, psi);
    }

    /// write the sensitivity map for all wall surfaces
    void writeSensMapSurface(
        const word name,
        const double* dFdXs,
        const double* Xs,
        const label size,
        const double timeName)
    {
        DASolverPtr_->writeSensMapSurface(name, dFdXs, Xs, size, timeName);
    }

    /// write the sensitivity map for the entire field
    void writeSensMapField(
        const word name,
        const double* dFdField,
        const word fieldType,
        const double timeName)
    {
        DASolverPtr_->writeSensMapField(name, dFdField, fieldType, timeName);
    }

    /// get the latest solution time
    double getLatestTime()
    {
        double latestTime = 0.0;
#if !defined(CODI_ADR) && !defined(CODI_ADF)
        latestTime = DASolverPtr_->getLatestTime();
#else
        scalar tmp = DASolverPtr_->getLatestTime();
        latestTime = tmp.getValue();
#endif
        return latestTime;
    };
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
