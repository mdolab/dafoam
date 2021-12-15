/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for DARhoSimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef DARhoSimpleFoam_H
#define DARhoSimpleFoam_H

#include "DASolver.H"
#include "addToRunTimeSelectionTable.H"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "simpleControl.H"
#include "pressureControl.H"
#include "DARegDbFluidThermo.H"
#include "DARegDbTurbulenceModelCompressible.H"
#include "DAFvSource.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DARhoSimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class DARhoSimpleFoam
    : public DASolver
{

protected:
    /// simple pointer
    autoPtr<simpleControl> simplePtr_;

    /// thermal model
    autoPtr<fluidThermo> pThermoPtr_;

    /// pressure field pointer
    autoPtr<volScalarField> pPtr_;

    /// density field pointer
    autoPtr<volScalarField> rhoPtr_;

    /// velocity field pointer
    autoPtr<volVectorField> UPtr_;

    /// surface flux field pointer
    autoPtr<surfaceScalarField> phiPtr_;

    /// pressure control pointer
    autoPtr<pressureControl> pressureControlPtr_;

    /// turbulence pointer
    autoPtr<compressible::turbulenceModel> turbulencePtr_;

    /// DATurbulenceModel pointer
    autoPtr<DATurbulenceModel> daTurbulenceModelPtr_;

    /// DASource pointer
    autoPtr<DAFvSource> daFvSourcePtr_;

    /// fvSource term
    autoPtr<volVectorField> fvSourcePtr_;

    /// fvSource term for the energy equation
    autoPtr<volScalarField> fvSourceEnergyPtr_;

    /// whether to have fvSource term
    label hasFvSource_ = 0;

    /// initial mass
    dimensionedScalar initialMass_;

    /// MRF pointer
    autoPtr<IOMRFZoneList> MRFPtr_;

    /// continuity error
    scalar cumulativeContErr_ = 0.0;

public:
    TypeName("DARhoSimpleFoam");
    // Constructors

    //- Construct from components
    DARhoSimpleFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DARhoSimpleFoam()
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
