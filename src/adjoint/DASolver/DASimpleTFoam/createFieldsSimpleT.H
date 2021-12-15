Info << "Reading field p\n"
     << endl;
pPtr_.reset(
    new volScalarField(
        IOobject(
            "p",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE),
        mesh));
volScalarField& p = pPtr_();

Info << "Reading field T\n"
     << endl;
TPtr_.reset(
    new volScalarField(
        IOobject(
            "T",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE),
        mesh));

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

#include "createPhiPython.H"

// create alpha porosity term
alphaPorosityPtr_.reset(
    new volScalarField(
        IOobject(
            "alphaPorosity",
            runTime.timeName(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar("alphaPorosity", dimensionSet(0, 0, -1, 0, 0, 0, 0), pTraits<scalar>::zero),
        zeroGradientFvPatchField<scalar>::typeName));

setRefCell(p, simple.dict(), pRefCell_, pRefValue_);
mesh.setFluxRequired(p.name());

laminarTransportPtr_.reset(
    new singlePhaseTransportModel(U, phi));
singlePhaseTransportModel& laminarTransport = laminarTransportPtr_();

// Laminar Prandtl number
PrPtr_.reset(
    new dimensionedScalar("Pr", dimless, laminarTransport));

// Turbulent Prandtl number
PrtPtr_.reset(
    new dimensionedScalar("Prt", dimless, laminarTransport));

// kinematic turbulent thermal thermal conductivity m2/s
Info << "Reading field alphat\n"
     << endl;
alphatPtr_.reset(
    new volScalarField(
        IOobject(
            "alphat",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE),
        mesh));

turbulencePtr_.reset(
    incompressible::turbulenceModel::New(U, phi, laminarTransport));
