/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "SpalartAllmarasFv3Beta.H"
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
void SpalartAllmarasFv3Beta<BasicTurbulenceModel>::correctNut()
{
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
SpalartAllmarasFv3Beta<BasicTurbulenceModel>::SpalartAllmarasFv3Beta(
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
      nuTilda_(
          IOobject(
              "nuTilda",
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::MUST_READ,
              IOobject::AUTO_WRITE),
          this->mesh_),
      betaSA_(
          IOobject(
              "betaSA",
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::MUST_READ,
              IOobject::AUTO_WRITE),
          this->mesh_),
      betaSATrue_(
          IOobject(
              "betaSATrue",
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::MUST_READ,
              IOobject::AUTO_WRITE),
          this->mesh_),
      UTrue_(
          IOobject(
              "UTrue",
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
bool SpalartAllmarasFv3Beta<BasicTurbulenceModel>::read()
{
    return true;
}

template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasFv3Beta<BasicTurbulenceModel>::k() const
{
    return this->nut_;
}

template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasFv3Beta<BasicTurbulenceModel>::epsilon() const
{

    return this->nut_;
}

template<class BasicTurbulenceModel>
void SpalartAllmarasFv3Beta<BasicTurbulenceModel>::correct()
{
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
