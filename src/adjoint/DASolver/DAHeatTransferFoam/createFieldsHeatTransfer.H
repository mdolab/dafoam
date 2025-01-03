Info << "Reading field T" << endl;
TPtr_.reset(
    new volScalarField(
        IOobject(
            "T",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE),
        mesh));

Info << "Reading solidProperties\n"
     << endl;

IOdictionary solidProperties(
    IOobject(
        "solidProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE));

// thermal conductivity unit: W/(m*K)
Info << "Reading thermal conductivity" << endl;
kPtr_.reset(
    new dimensionedScalar(
        "k",
        dimPower / dimLength / dimTemperature,
        solidProperties));

// heat source unit: W/(m^3)
Info << "Creating source term. " << endl;
fvSourcePtr_.reset(
    new volScalarField(
        IOobject(
            "fvSource",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar("fvSource", dimPower / dimLength / dimLength / dimLength, 0.0),
        "zeroGradient"));
