/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "kOmegaSSTFIML.H"
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
void kOmegaSSTFIML<BasicTurbulenceModel>::correctNut()
{
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
kOmegaSSTFIML<BasicTurbulenceModel>::kOmegaSSTFIML(
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
        propertiesName),
      omega_(
          IOobject(
              "omega",
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::MUST_READ,
              IOobject::AUTO_WRITE),
          this->mesh_),
      k_(
          IOobject(
              "k",
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::MUST_READ,
              IOobject::AUTO_WRITE),
          this->mesh_),
      y_(wallDist::New(this->mesh_).y())

{
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool kOmegaSSTFIML<BasicTurbulenceModel>::read()
{

    return true;
}

template<class BasicTurbulenceModel>
tmp<volScalarField> kOmegaSSTFIML<BasicTurbulenceModel>::k() const
{
    return this->nut_;
}

template<class BasicTurbulenceModel>
tmp<volScalarField> kOmegaSSTFIML<BasicTurbulenceModel>::epsilon() const
{
    return this->nut_;
}

template<class BasicTurbulenceModel>
void kOmegaSSTFIML<BasicTurbulenceModel>::correct()
{
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
