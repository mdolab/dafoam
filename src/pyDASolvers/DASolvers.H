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

    label getInputSize(
        const word inputName,
        const word inputType)
    {
        return DASolverPtr_->getInputSize(inputName, inputType);
    }

    label getOutputSize(
        const word outputName,
        const word outputType)
    {
        return DASolverPtr_->getOutputSize(outputName, outputType);
    }

    void calcOutput(
        const word outputName,
        const word outputType,
        double* output)
    {
        DASolverPtr_->calcOutput(outputName, outputType, output);
    }

    label getInputDistributed(
        const word inputName,
        const word inputType)
    {
        return DASolverPtr_->getInputDistributed(inputName, inputType);
    }

    label getOutputDistributed(
        const word outputName,
        const word outputType)
    {
        return DASolverPtr_->getOutputDistributed(outputName, outputType);
    }

    /// calculate the Jacobian-matrix-transposed and vector product for product = [dOutput/dInput]^T * seed
    void calcJacTVecProduct(
        const word inputName,
        const word inputType,
        const double* input,
        const word outputName,
        const word outputType,
        const double* seed,
        double* product)
    {
        DASolverPtr_->calcJacTVecProduct(
            inputName,
            inputType,
            input,
            outputName,
            outputType,
            seed,
            product);
    }

    void setSolverInput(
        const word inputName,
        const word inputType,
        const int inputSize,
        const double* input,
        const double* seed)
    {
        DASolverPtr_->setSolverInput(
            inputName,
            inputType,
            inputSize,
            input,
            seed);
    }

    /// run the coloring solver
    void runColoring()
    {
        DASolverPtr_->runColoring();
    }

    /// compute dRdWT
    void calcdRdWT(
        const label isPC,
        Mat dRdWT)
    {
        DASolverPtr_->calcdRdWT(isPC, dRdWT);
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

    /// compute dRdWOld^T*Psi
    void calcdRdWOldTPsiAD(
        const label oldTimeLevel,
        const double* psi,
        double* dRdWOldTPsi)
    {
        DASolverPtr_->calcdRdWOldTPsiAD(oldTimeLevel, psi, dRdWOldTPsi);
    }

    /// Update the OpenFOAM field values (including both internal and boundary fields) based on the states array
    void updateOFFields(const double* states)
    {
#if !defined(CODI_ADR) && !defined(CODI_ADF)
        DASolverPtr_->updateOFFields(states);
#else
        label localSize = this->getNLocalAdjointStates();
        scalar* statesArray = new scalar[localSize];
        for (label i = 0; i < localSize; i++)
        {
            statesArray[i] = states[i];
        }
        DASolverPtr_->updateOFFields(statesArray);
        delete[] statesArray;
#endif
    }

    /// get the flatten mesh points coordinates
    void getOFMeshPoints(double* points)
    {
        DASolverPtr_->getOFMeshPoints(points);
    }

    /// Assign the OpenFOAM field values to the states array
    void getOFFields(double* states)
    {
#if !defined(CODI_ADR) && !defined(CODI_ADF)
        DASolverPtr_->getOFFields(states);
#else
        label localSize = this->getNLocalAdjointStates();
        scalar* statesArray = new scalar[localSize];
        DASolverPtr_->getOFFields(statesArray);
        for (label i = 0; i < localSize; i++)
        {
            states[i] = statesArray[i].value();
        }
        delete[] statesArray;
#endif
    }

    /// get a field variable from OF layer
    void getOFField(
        const word fieldName,
        const word fieldType,
        double* field)
    {
        DASolverPtr_->getOFField(fieldName, fieldType, field);
    }

    /// Update the OpenFoam mesh point coordinates based on the point vector xvVec
    void updateOFMesh(const double* volCoords)
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

    /// basically, we call DASolver::checkMesh
    label checkMesh() const
    {
        return DASolverPtr_->checkMesh();
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

    /// return the value of objective based on timeOp
    double getTimeOpFuncVal(const word functionName)
    {
        return DASolverPtr_->getTimeOpFuncVal(functionName);
    }

    /// get the scaling factor for dF/d? computation
    double getdFScaling(const word functionName, const label timeIdx)
    {
        double returnVal = 0.0;
        assignValueCheckAD(returnVal, DASolverPtr_->getdFScaling(functionName, timeIdx));
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

    /// get the number of regression model parameters
    label getNRegressionParameters(word modelName)
    {
        return DASolverPtr_->getNRegressionParameters(modelName);
    }

    /// call DASolver::printAllOptions
    void printAllOptions()
    {
        DASolverPtr_->printAllOptions();
    }

    /// whether the volCoord input is defined
    label hasVolCoordInput()
    {
        return DASolverPtr_->hasVolCoordInput();
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

    void writeFailedMesh()
    {
        DASolverPtr_->writeFailedMesh();
    }

    /// read the state variables from the disk and assign the value to the prescribe time level
    void readStateVars(
        scalar timeVal,
        label timeLevel = 0)
    {
        DASolverPtr_->readStateVars(timeVal, timeLevel);
    }

    /// read the mesh points from the disk and run movePoints to deform the mesh
    void readMeshPoints(const scalar timeVal)
    {
        DASolverPtr_->readMeshPoints(timeVal);
    }

    /// write the mesh points to the disk for the given timeVal
    void writeMeshPoints(const double* points, const scalar timeVal)
    {
        DASolverPtr_->writeMeshPoints(points, timeVal);
    }

    /// calculate the PC mat using fvMatrix
    void calcPCMatWithFvMatrix(Mat PCMat, const label turbOnly)
    {
        DASolverPtr_->calcPCMatWithFvMatrix(PCMat, turbOnly);
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

    void setPrimalBoundaryConditions(const label printInfo = 1)
    {
        DASolverPtr_->setPrimalBoundaryConditions(printInfo);
    }

    label runFPAdj(
        Vec dFdW,
        Vec psi)
    {
        return DASolverPtr_->runFPAdj(dFdW, psi);
    }

    label solveAdjointFP(
        Vec dFdW,
        Vec psi)
    {
        return DASolverPtr_->solveAdjointFP(dFdW, psi);
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

    /// assign the mean states values to states
    void meanStatesToStates()
    {
        DASolverPtr_->meanStatesToStates();
    }

    /// assign the inputFieldUnsteady values to the OF field vars
    void updateInputFieldUnsteady()
    {
        DASolverPtr_->updateInputFieldUnsteady();
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
