/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for DAPimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef DAPimpleFoam_H
#define DAPimpleFoam_H

#include "DASolver.H"
#include "addToRunTimeSelectionTable.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "pimpleControlDF.H"
#include "DARegDbSinglePhaseTransportModel.H"
#include "DARegDbTurbulenceModelIncompressible.H"
#include "DAFvSource.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAPimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class DAPimpleFoam
    : public DASolver
{

protected:
    /// pimple pointer
    autoPtr<pimpleControlDF> pimplePtr_;

    /// pressure field pointer
    autoPtr<volScalarField> pPtr_;

    /// velocity field pointer
    autoPtr<volVectorField> UPtr_;

    /// surface flux field pointer
    autoPtr<surfaceScalarField> phiPtr_;

    /// laminar transport properties pointer
    autoPtr<singlePhaseTransportModel> laminarTransportPtr_;

    /// turbulence pointer
    autoPtr<incompressible::turbulenceModel> turbulencePtr_;

    /// DATurbulenceModel pointer
    autoPtr<DATurbulenceModel> daTurbulenceModelPtr_;

    /// DASource pointer
    autoPtr<DAFvSource> daFvSourcePtr_;

    /// fvSource term
    autoPtr<volVectorField> fvSourcePtr_;

    /// whether to have fvSource term
    label hasFvSource_ = 0;

    /// continuity error
    scalar cumulativeContErr_ = 0.0;

    /// pressure referefence cell id
    label pRefCell_ = 0;

    /// pressure reference value
    scalar pRefValue_ = 0.0;

    /// whether the hybrid adjoint or time accurate adjoint is active
    word mode_ = "None";

public:
    TypeName("DAPimpleFoam");
    // Constructors

    //- Construct from components
    DAPimpleFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DAPimpleFoam()
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
