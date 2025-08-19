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
        dimensionedVector("fvSource", dimensionSet(0, 1, -2, 0, 0, 0, 0), vector::zero),
        zeroGradientFvPatchField<vector>::typeName));

setRefCell(p, simple.dict(), pRefCell_, pRefValue_);
mesh.setFluxRequired(p.name());

laminarTransportPtr_.reset(
    new singlePhaseTransportModel(U, phi));
singlePhaseTransportModel& laminarTransport = laminarTransportPtr_();

turbulencePtr_.reset(
    incompressible::turbulenceModel::New(U, phi, laminarTransport));

// create MRF
MRFPtr_.reset(new IOMRFZoneListDF(mesh));

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

// Laminar Prandtl number
PrPtr_.reset(
    new dimensionedScalar("Pr", dimless, laminarTransport));

// Turbulent Prandtl number
PrtPtr_.reset(
    new dimensionedScalar("Prt", dimless, laminarTransport));

IOdictionary transportProperties(
    IOobject(
        "transportProperties",
        mesh.time().constant(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE,
        false));
alphaPorosityFluid_ = readScalar(transportProperties.lookup("alphaPorosityFluid"));
alphaPorositySolid_ = readScalar(transportProperties.lookup("alphaPorositySolid"));
diffusivitySolid_ = readScalar(transportProperties.lookup("diffusivitySolid"));
pCoeff_ = readScalar(transportProperties.lookup("pCoeff"));

// total diffusivity field
Info << "Reading field diffusivity\n"
     << endl;
diffusivityPtr_.reset(
    new volScalarField(
        IOobject(
            "diffusivity",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar("diffusivity", dimensionSet(0, 2, -1, 0, 0, 0, 0), 0.0),
        zeroGradientFvPatchField<scalar>::typeName));

diffusivityFluidPtr_.reset(
    new volScalarField(
        IOobject(
            "alphat",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE),
        mesh));

// topology optimization's design variable
etaPtr_.reset(
    new volScalarField(
        IOobject(
            "eta",
            runTime.timeName(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar("eta", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
        zeroGradientFvPatchField<scalar>::typeName));
