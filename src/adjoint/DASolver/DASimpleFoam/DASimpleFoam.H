/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for DASimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef DASimpleFoam_H
#define DASimpleFoam_H

#include "DASolver.H"
#include "addToRunTimeSelectionTable.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "DARegDbSinglePhaseTransportModel.H"
#include "DARegDbTurbulenceModelIncompressible.H"
#include "DAFvSource.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DASimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class DASimpleFoam
    : public DASolver
{

protected:

    /// simple pointer
    autoPtr<simpleControl> simplePtr_;

    /// pressure field pointer
    autoPtr<volScalarField> pPtr_;

    /// velocity field pointer
    autoPtr<volVectorField> UPtr_;

    /// surface flux field pointer
    autoPtr<surfaceScalarField> phiPtr_;

    /// alpha porosity
    autoPtr<volScalarField> alphaPorosityPtr_;

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

    /// MRF pointer
    autoPtr<IOMRFZoneList> MRFPtr_;

    /// whether to have fvSource term
    label hasFvSource_ = 0;

    /// continuity error
    scalar cumulativeContErr_ = 0.0;

    /// pressure referefence cell id
    label pRefCell_ = 0;

    /// pressure reference value
    scalar pRefValue_ = 0.0;

    
public:
    TypeName("DASimpleFoam");
    // Constructors

    //- Construct from components
    DASimpleFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DASimpleFoam()
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
