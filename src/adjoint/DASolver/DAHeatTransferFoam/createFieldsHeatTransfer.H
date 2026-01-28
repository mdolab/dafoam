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
if (solidProperties.found("kappa"))
{
    kappaCoeffs = List<scalar>(1, solidProperties.getScalar("kappa"));
}
else if (solidProperties.found("kappaCoeffs"))
{
    kappaCoeffs = solidProperties.lookup("kappaCoeffs");
}
else
{
    FatalErrorInFunction
        << "Neither 'kappa' nor 'kappaCoeffs' found in dictionary: "
        << solidProperties.name() << exit(FatalError);
}

kappaPtr_.reset(
    new volScalarField(
        IOobject(
            "kappa",
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
