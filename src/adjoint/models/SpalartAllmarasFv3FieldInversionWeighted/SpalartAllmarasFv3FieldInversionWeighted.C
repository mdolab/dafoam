/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "SpalartAllmarasFv3FieldInversionWeighted.H"
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
void SpalartAllmarasFv3FieldInversionWeighted<BasicTurbulenceModel>::correctNut()
{
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
SpalartAllmarasFv3FieldInversionWeighted<BasicTurbulenceModel>::SpalartAllmarasFv3FieldInversionWeighted(
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
      betaRefFieldInversion_(
          IOobject(
              "betaRefFieldInversion",
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::READ_IF_PRESENT,
              IOobject::NO_WRITE),
          this->mesh_,
          dimensionedScalar("betaRefFieldInversion", dimensionSet(0, 0, 0, 0, 0, 0, 0), 1.0),
          zeroGradientFvPatchField<scalar>::typeName),
      varRefFieldInversion_(
          IOobject(
              "varRefFieldInversion",
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::MUST_READ,
              IOobject::AUTO_WRITE),
          this->mesh_),
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
      surfaceFrictionRef_(
          IOobject(
              "surfaceFrictionRef",
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::READ_IF_PRESENT,
              IOobject::AUTO_WRITE),
          this->mesh_,
          dimensionedScalar("surfaceFrictionRef", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchField<scalar>::typeName),
      surfacePressureRef_(
          IOobject(
              "surfacePressureRef",
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::READ_IF_PRESENT,
              IOobject::AUTO_WRITE),
          this->mesh_,
          dimensionedScalar("surfacePressureRef", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchField<scalar>::typeName),
      profileRefFieldInversion_(
          IOobject(
              "profileRefFieldInversion",
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::READ_IF_PRESENT,
              IOobject::AUTO_WRITE),
          this->mesh_,
          dimensionedScalar("profileRefFieldInversion", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchField<scalar>::typeName),
      weightsObjFunc_(
          IOobject(
              "weightsObjFunc",
              this->runTime_.timeName(),
              this->mesh_,
              IOobject::READ_IF_PRESENT,
              IOobject::AUTO_WRITE),
          this->mesh_,
          dimensionedScalar("weightsObjFunc", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
          zeroGradientFvPatchField<scalar>::typeName),
      y_(wallDist::New(this->mesh_).y())
{
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool SpalartAllmarasFv3FieldInversionWeighted<BasicTurbulenceModel>::read()
{
    return true;
}

template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasFv3FieldInversionWeighted<BasicTurbulenceModel>::k() const
{
    return this->nut_;
}

template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasFv3FieldInversionWeighted<BasicTurbulenceModel>::epsilon() const
{

    return this->nut_;
}

template<class BasicTurbulenceModel>
void SpalartAllmarasFv3FieldInversionWeighted<BasicTurbulenceModel>::correct()
{
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
