/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for DAPimpleDyMFoam

\*---------------------------------------------------------------------------*/

#ifndef DAPimpleDyMFoam_H
#define DAPimpleDyMFoam_H

#include "DASolver.H"
#include "addToRunTimeSelectionTable.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "pimpleControlDF.H"
#include "DARegDbSinglePhaseTransportModel.H"
#include "DARegDbTurbulenceModelIncompressible.H"
#include "pimpleControl.H"
#include "dynamicFvMesh.H"
#include "CorrectPhi.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAPimpleDyMFoam Declaration
\*---------------------------------------------------------------------------*/

class DAPimpleDyMFoam
    : public DASolver
{

protected:

public:
    TypeName("DAPimpleDyMFoam");
    // Constructors

    //- Construct from components
    DAPimpleDyMFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DAPimpleDyMFoam()
    {
    }

    /// initialize fields and variables
    virtual void initSolver();

    /// solve the primal equations
    virtual label solvePrimal(
        const Vec xvVec,
        Vec wVec);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
