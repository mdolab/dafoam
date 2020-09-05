// add dummy U, phi, and laminarTransport

fvMesh& mesh = meshPtr_();

volVectorField D(
    IOobject(
        "D",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedVector("D", dimensionSet(0, 1, 0, 0, 0, 0, 0), vector::zero),
    zeroGradientFvPatchField<vector>::typeName);

volScalarField mu(
    IOobject(
        "solid:mu",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("solid:mu", dimensionSet(0, 1, 0, 0, 0, 0, 0), 0.0),
    zeroGradientFvPatchField<scalar>::typeName);

volScalarField rho(
    IOobject(
        "solid:rho",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("solid:rho", dimensionSet(0, 1, 0, 0, 0, 0, 0), 0.0),
    zeroGradientFvPatchField<scalar>::typeName);

volScalarField nu(
    IOobject(
        "solid:nu",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("solid:nu", dimensionSet(0, 1, 0, 0, 0, 0, 0), 0.0),
    zeroGradientFvPatchField<scalar>::typeName);

volScalarField lambda(
    IOobject(
        "solid:lambda",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("solid:lambda", dimensionSet(0, 1, 0, 0, 0, 0, 0), 0.0),
    zeroGradientFvPatchField<scalar>::typeName);

volTensorField gradD(
    IOobject(
        "gradD",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    fvc::grad(D));

volSymmTensorField sigmaD(
    IOobject(
        "sigmaD",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mu* twoSymm(fvc::grad(D)) + lambda * (I * tr(fvc::grad(D))));

volVectorField divSigmaExp(
    IOobject(
        "divSigmaExp",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedVector("divSigmaExp", dimensionSet(0, 1, 0, 0, 0, 0, 0), vector::zero),
    zeroGradientFvPatchField<vector>::typeName);
