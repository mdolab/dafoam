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
phi.write();

Info << "Constructing face velocity Uf\n"
     << endl;

UfPtr_.reset(
    new surfaceVectorField(
        IOobject(
            "Uf",
            runTime.timeName(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE),
        fvc::interpolate(U)));

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
volVectorField& fvSource = fvSourcePtr_();

setRefCell(p, pimple.dict(), pRefCell_, pRefValue_);
mesh.setFluxRequired(p.name());

laminarTransportPtr_.reset(
    new singlePhaseTransportModel(U, phi));
singlePhaseTransportModel& laminarTransport = laminarTransportPtr_();

turbulencePtr_.reset(
    incompressible::turbulenceModel::New(U, phi, laminarTransport));

if (hasTField_)
{
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
}
