/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "SpalartAllmarasFieldInversion.H"
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
void SpalartAllmarasFieldInversion<BasicTurbulenceModel>::correctNut()
{
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
SpalartAllmarasFieldInversion<BasicTurbulenceModel>::SpalartAllmarasFieldInversion(
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
      betaFieldInversion_(
          IOobject(
              "betaFieldInversion",
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::MUST_READ,
              IOobject::AUTO_WRITE),
          this->mesh_),
      UData_(
          IOobject(
              "UData",
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::READ_IF_PRESENT,
              IOobject::AUTO_WRITE),
          this->mesh_,
          dimensionedVector("UData", dimensionSet(0, 1, -1, 0, 0, 0, 0), vector(0, 0, 0)),
          "zeroGradient"),
      surfaceFriction_(
          IOobject(
              "surfaceFriction",
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::READ_IF_PRESENT,
              IOobject::AUTO_WRITE),
          this->mesh_,
          dimensionedScalar("surfaceFriction", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchField<scalar>::typeName),
      surfaceFrictionData_(
          IOobject(
              "surfaceFrictionData",
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::READ_IF_PRESENT,
              IOobject::AUTO_WRITE),
          this->mesh_,
          dimensionedScalar("surfaceFrictionData", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchField<scalar>::typeName),
      pData_(
          IOobject(
              "pData",
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::READ_IF_PRESENT,
              IOobject::AUTO_WRITE),
          this->mesh_,
          dimensionedScalar("pData", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchField<scalar>::typeName),
      USingleComponentData_(
          IOobject(
              "USingleComponentData",
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::READ_IF_PRESENT,
              IOobject::AUTO_WRITE),
          this->mesh_,
          dimensionedScalar("USingleComponentData", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchField<scalar>::typeName),
      y_(wallDist::New(this->mesh_).y())
{
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool SpalartAllmarasFieldInversion<BasicTurbulenceModel>::read()
{
    return true;
}

template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasFieldInversion<BasicTurbulenceModel>::k() const
{
    return this->nut_;
}

template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasFieldInversion<BasicTurbulenceModel>::epsilon() const
{

    return this->nut_;
}

template<class BasicTurbulenceModel>
void SpalartAllmarasFieldInversion<BasicTurbulenceModel>::correct()
{
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
