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
if (solidProperties.found("k"))
{
    kCoeffs = List<scalar>(1, solidProperties.getScalar("k"));
}
else if (solidProperties.found("kCoeffs"))
{
    kCoeffs = solidProperties.lookup("kCoeffs");
}
else
{
    FatalErrorInFunction
        << "Neither 'k' nor 'kCoeffs' found in dictionary: "
        << solidProperties.name() << exit(FatalError);
}

kPtr_.reset(
    new volScalarField(
        IOobject(
            "k",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        mesh,
        dimensionedScalar(dimPower / dimLength / dimTemperature, Zero)));

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
