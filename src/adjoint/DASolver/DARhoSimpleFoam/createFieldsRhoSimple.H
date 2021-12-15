Info << "Reading thermophysical properties\n"
     << endl;

pThermoPtr_.reset(fluidThermo::New(mesh));
fluidThermo& thermo = pThermoPtr_();
thermo.validate(args.executable(), "h", "e");

volScalarField& p = thermo.p();

rhoPtr_.reset(
    new volScalarField(
        IOobject(
            "rho",
            runTime.timeName(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE),
        thermo.rho()));
volScalarField& rho = rhoPtr_();

Info << "Reading field U\n"
     << endl;

UPtr_.reset(
    new volVectorField(
        IOobject(
            "U",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE),
        mesh));
volVectorField& U = UPtr_();

#include "compressibleCreatePhiPython.H"

// actuator related stuff
Info << "Creating source term. " << endl;
fvSourcePtr_.reset(
    new volVectorField(
        IOobject(
            "fvSource",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE),
        mesh,
        dimensionedVector("fvSource", dimensionSet(1, -2, -2, 0, 0, 0, 0), vector::zero),
        zeroGradientFvPatchField<vector>::typeName));

fvSourceEnergyPtr_.reset(
    new volScalarField(
        IOobject(
            "fvSourceEnergy",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar("fvSourceEnergy", dimensionSet(1, -1, -3, 0, 0, 0, 0), 0.0),
        zeroGradientFvPatchField<scalar>::typeName));

pressureControlPtr_.reset(new pressureControl(p, rho, simple.dict()));

mesh.setFluxRequired(p.name());

Info << "Creating turbulence model\n"
     << endl;

turbulencePtr_.reset(
    compressible::turbulenceModel::New(
        rho,
        U,
        phi,
        thermo));

initialMass_ = fvc::domainIntegrate(rho);

#include "createMRFPython.H"
