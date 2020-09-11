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
    zeroGradientFvPatchField<vector>::typeName);

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
    zeroGradientFvPatchField<scalar>::typeName);

#include "createPhi.H"

singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::turbulenceModel> turbulence(
    incompressible::turbulenceModel::New(U, phi, laminarTransport));
