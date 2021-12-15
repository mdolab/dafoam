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
