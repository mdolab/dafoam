// add dummy U, phi, and laminarTransport

Time& runTime = runTimePtr_();
fvMesh& mesh = meshPtr_();

volVectorField U(
    IOobject(
        "U",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedVector("U", dimensionSet(0, 1, -1, 0, 0, 0, 0), vector::zero),
    zeroGradientFvPatchVectorField::typeName);

// need the p field for DATurbulenceModel
volScalarField p(
    IOobject(
        "p",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("p", dimensionSet(0, 2, -2, 0, 0, 0, 0), 0.0),
    zeroGradientFvPatchScalarField::typeName);

volScalarField alphat(
    IOobject(
        "alphat",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("alphat", dimensionSet(1, -1, -1, 0, 0, 0, 0), 0.0),
    zeroGradientFvPatchScalarField::typeName);

volScalarField T(
    IOobject(
        "T",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("T", dimensionSet(0, 0, 0, 1, 0, 0, 0), 0.0),
    zeroGradientFvPatchScalarField::typeName);

#include "createPhi.H"

// fvSource term
volVectorField fvSource(
    IOobject(
        "fvSource",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedVector("fvSource", dimensionSet(0, 1, -2, 0, 0, 0, 0), vector::zero),
    zeroGradientFvPatchVectorField::typeName);

singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::turbulenceModel> turbulence(
    incompressible::turbulenceModel::New(U, phi, laminarTransport));
