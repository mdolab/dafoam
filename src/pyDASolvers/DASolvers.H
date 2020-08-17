/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

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

    /// solve the adjoint linear equations
    label solveAdjoint(
        const Vec xvVec,
        const Vec wVec)
    {
        return DASolverPtr_->solveAdjoint(xvVec, wVec);
    }

    /// compute the total derivatives
    label calcTotalDeriv(
        const Vec xvVec,
        const Vec wVec,
        const word designVarName)
    {
        return DASolverPtr_->calcTotalDeriv(xvVec, wVec, designVarName);
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

    /// return the value of objective
    scalar getObjFuncValue(const word objFuncName)
    {
        return DASolverPtr_->getObjFuncValue(objFuncName);
    }

    /// basically, we call DASolver::getTotalDerivVal
    scalar getTotalDerivVal(
        const word objFuncName,
        const word designVarName,
        const label idxI) const
    {
        return DASolverPtr_->getTotalDerivVal(objFuncName, designVarName, idxI);
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

    /// update the allOptions_ dict in DAOption based on the pyOptions from pyDAFoam
    void updateDAOption(PyObject* pyOptions)
    {
        DASolverPtr_->updateDAOption(pyOptions);
    }

    /// get the solution time folder for previous primal solution
    scalar getPrevPrimalSolTime()
    {
        return DASolverPtr_->getPrevPrimalSolTime();
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
    void getTimeInstance(const label instanceI)
    {
        DASolverPtr_->getTimeInstance(instanceI);
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
