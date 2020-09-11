/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "dummyTurbulenceModel.H"
#include "fvOptions.H"
#include "bound.H"
#include "wallDist.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
void dummyTurbulenceModel<BasicTurbulenceModel>::correctNut()
{
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
dummyTurbulenceModel<BasicTurbulenceModel>::dummyTurbulenceModel(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type)
    : eddyViscosity<RASModel<BasicTurbulenceModel>>(
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName)
{
    // set turbulence variable to zero
    forAll(this->nut_, idxI) this->nut_[idxI] = 0.0;
    forAll(this->nut_.boundaryField(), patchI)
    {
        forAll(this->nut_.boundaryField()[patchI], faceI)
        {
            this->nut_.boundaryFieldRef()[patchI][faceI] = 0.0;
        }
    }
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool dummyTurbulenceModel<BasicTurbulenceModel>::read()
{

    return true;
}

template<class BasicTurbulenceModel>
tmp<volScalarField> dummyTurbulenceModel<BasicTurbulenceModel>::k() const
{
    return this->nut_;
}

template<class BasicTurbulenceModel>
tmp<volScalarField> dummyTurbulenceModel<BasicTurbulenceModel>::epsilon() const
{
    return this->nut_;
}

template<class BasicTurbulenceModel>
void dummyTurbulenceModel<BasicTurbulenceModel>::correct()
{
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
