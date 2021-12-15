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

Info << "Reading transportProperties\n"
     << endl;

IOdictionary transportProperties(
    IOobject(
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE));

Info << "Reading diffusivity DT\n"
     << endl;

DTPtr_.reset(
    new dimensionedScalar(
        "DT",
        dimViscosity,
        transportProperties));

Info << "Reading/calculating face flux field phi\n"
     << endl;

phiPtr_.reset(
    new surfaceScalarField(
        IOobject(
            "phi",
            runTime.timeName(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE),
        fvc::flux(U)));