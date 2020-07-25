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
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
