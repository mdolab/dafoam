/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for DARhoSimpleCFoam

\*---------------------------------------------------------------------------*/

#ifndef DARhoSimpleCFoam_H
#define DARhoSimpleCFoam_H

#include "DASolver.H"
#include "addToRunTimeSelectionTable.H"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "simpleControl.H"
#include "pressureControl.H"
#include "DARegDbFluidThermo.H"
#include "DARegDbTurbulenceModelCompressible.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DARhoSimpleCFoam Declaration
\*---------------------------------------------------------------------------*/

class DARhoSimpleCFoam
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

    /// initial mass
    dimensionedScalar initialMass_;

    /// continuity error
    scalar cumulativeContErr_ = 0.0;

public:
    TypeName("DARhoSimpleCFoam");
    // Constructors

    //- Construct from components
    DARhoSimpleCFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DARhoSimpleCFoam()
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
