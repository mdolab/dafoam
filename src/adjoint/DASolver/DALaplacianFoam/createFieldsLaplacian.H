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