// add dummy U, phi, and laminarTransport

Time& runTime = runTimePtr_();
fvMesh& mesh = meshPtr_();

autoPtr<fluidThermo> pThermo(fluidThermo::New(mesh));
fluidThermo& thermo = pThermo();

volScalarField rho(
    IOobject(
        "rho",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("rho", dimensionSet(1, -3, 0, 0, 0, 0, 0), 1.0),
    zeroGradientFvPatchField<scalar>::typeName);

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
    dimensionedScalar("p", dimensionSet(1, -1, -2, 0, 0, 0, 0), 101325.0),
    zeroGradientFvPatchField<scalar>::typeName);

surfaceScalarField phi(
    IOobject(
        "phi",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE),
    linearInterpolate(rho* U) & mesh.Sf());
    
autoPtr<compressible::turbulenceModel> turbulence(
    compressible::turbulenceModel::New(
        rho,
        U,
        phi,
        thermo));
