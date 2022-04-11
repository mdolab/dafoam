/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Solve the linear euqation using Petsc

\*---------------------------------------------------------------------------*/

#ifndef DALinearEqn_H
#define DALinearEqn_H

#include "fvOptions.H"
#include "surfaceFields.H"
#include "DAOption.H"
#include "DAUtility.H"
#include "DAStateInfo.H"
#include "DAModel.H"
#include "DAIndex.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DALinearEqn Declaration
\*---------------------------------------------------------------------------*/

class DALinearEqn
{

private:
    /// Disallow default bitwise copy construct
    DALinearEqn(const DALinearEqn&);

    /// Disallow default bitwise assignment
    void operator=(const DALinearEqn&);

protected:
    /// Foam::fvMesh object
    const fvMesh& mesh_;

    /// Foam::DAOption object
    const DAOption& daOption_;

public:
    /// Constructors
    DALinearEqn(
        const fvMesh& mesh,
        const DAOption& daOption);

    /// Destructor
    virtual ~DALinearEqn()
    {
    }

    // Members

    /// create a multi-level, Richardson KSP object
    void createMLRKSP(
        const Mat jacMat,
        const Mat jacPCMat,
        KSP ksp);

    /// solve the linear equation given a ksp and right-hand-side vector
    label solveLinearEqn(
        const KSP ksp,
        const Vec rhsVec,
        Vec solVec);

    /// ksp monitor function
    static PetscErrorCode myKSPMonitor(
        KSP,
        PetscInt,
        PetscReal,
        void*);

    /// return the runtime for the adjoint solver
    label getRunTime();

    /// return printInterval from DAOption
    label getPrintInterval();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
