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
/*fvSourcePtr_.reset(
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
*/



fvSourcePtr_.reset(
    new volVectorField(
        IOobject(
            "fvSource",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE),
        mesh));

volVectorField& fvSource = fvSourcePtr_();

/*********************************************

	User defined fields 

**********************************************/

// User defined function for destruction term
Info << "Reading field destruct\n"
     << endl;
destructionPtr_.reset(
    new volScalarField(
        IOobject(
            "destruction",
            runTime.timeName(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar("destruction", dimensionSet(0, 2, -3, 0, 0, 0, 0), 0.0),
        zeroGradientFvPatchField<scalar>::typeName));
volScalarField& destruction = destructionPtr_();

// User defined function for production term
Info << "Reading field production\n"
     << endl;
productionPtr_.reset(
    new volScalarField(
        IOobject(
            "production",
            runTime.timeName(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar("production", dimensionSet(0, 2, -3, 0, 0, 0, 0), 0.0),
        zeroGradientFvPatchField<scalar>::typeName));
volScalarField& production = productionPtr_();

/*// User defined function for magnitude of strain rate 
Info << "Reading field strain rate\n"
     << endl;
strainRatePtr_.reset(
    new volSymmTensorField(
        IOobject(
            "strainRate",
            runTime.timeName(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar("strainRate", dimensionSet(0, 0, -1, 0, 0, 0, 0),  symm(fvc::grad(U))),
        zeroGradientFvPatchField<scalar>::typeName));
volSymmTensorField& strainRate = strainRatePtr_();

// User defined function for strain rate term
Info << "Reading field rotation rate\n"
     << endl;
rotationRatePtr_.reset(
    new volTensorField(
        IOobject(
            "rotationRate",
            runTime.timeName(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar("rotationRate", dimensionSet(0, 0, -1, 0, 0, 0, 0),  skew(fvc::grad(U))),
        zeroGradientFvPatchField<scalar>::typeName));
volTensorField& rotationRate = rotationRatePtr_();
*/

/*********************************************

	User defined fields 

**********************************************/



setRefCell(p, simple.dict(), pRefCell_, pRefValue_);
mesh.setFluxRequired(p.name());

laminarTransportPtr_.reset(
    new singlePhaseTransportModel(U, phi));
singlePhaseTransportModel& laminarTransport = laminarTransportPtr_();

turbulencePtr_.reset(
    incompressible::turbulenceModel::New(U, phi, laminarTransport));

// create MRF
MRFPtr_.reset(new IOMRFZoneListDF(mesh));

