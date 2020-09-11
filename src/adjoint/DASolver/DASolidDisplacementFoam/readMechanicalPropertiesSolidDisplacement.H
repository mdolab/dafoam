Info << "Reading mechanical properties\n"
     << endl;

mechanicalPropertiesPtr_.reset(
    new IOdictionary(
        IOobject(
            "mechanicalProperties",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE)));
IOdictionary& mechanicalProperties = mechanicalPropertiesPtr_();

const dictionary& rhoDict(mechanicalProperties.subDict("rho"));
word rhoType(rhoDict.lookup("type"));

IOobject rhoIO(
    "solid:rho",
    runTime.timeName(0),
    mesh,
    IOobject::NO_READ,
    IOobject::NO_WRITE);

if (rhoType == "uniform")
{
    scalar rhoValue(readScalar(rhoDict.lookup("value")));

    rhoPtr_.reset(
        new volScalarField(
            rhoIO,
            mesh,
            dimensionedScalar(
                "solid:rho",
                dimMass / dimVolume,
                rhoValue)));
}
else if (rhoType == "field")
{
    rhoIO.readOpt() = IOobject::MUST_READ;

    rhoPtr_.reset(
        new volScalarField(
            rhoIO,
            mesh));
}
else
{
    FatalErrorInFunction
        << "Valid type entries are uniform or field for rho"
        << abort(FatalError);
}

volScalarField& rho = rhoPtr_();

const dictionary& EDict(mechanicalProperties.subDict("E"));
word EType(EDict.lookup("type"));

IOobject EHeader(
    "E",
    runTime.timeName(0),
    mesh,
    IOobject::NO_READ,
    IOobject::NO_WRITE);

if (EType == "uniform")
{
    scalar rhoEValue(readScalar(EDict.lookup("value")));

    EPtr_.reset(
        new volScalarField(
            EHeader,
            mesh,
            dimensionedScalar(
                "Erho",
                dimMass / dimLength / sqr(dimTime),
                rhoEValue)));
}
else if (EType == "field")
{
    EHeader.readOpt() = IOobject::MUST_READ;

    EPtr_.reset(
        new volScalarField(
            EHeader,
            mesh));
}
else
{
    FatalErrorInFunction
        << "Valid type entries are uniform or field for E"
        << abort(FatalError);
}

volScalarField& rhoE = EPtr_();

IOobject nuIO(
    "solid:nu",
    runTime.timeName(0),
    mesh,
    IOobject::NO_READ,
    IOobject::NO_WRITE);

const dictionary& nuDict(mechanicalProperties.subDict("nu"));
word nuType(nuDict.lookup("type"));

if (nuType == "uniform")
{
    scalar nuValue(readScalar(nuDict.lookup("value")));
    nuPtr_.reset(
        new volScalarField(
            nuIO,
            mesh,
            dimensionedScalar(
                "solid:nu",
                dimless,
                nuValue)));
}
else if (nuType == "field")
{
    nuIO.readOpt() = IOobject::MUST_READ;
    nuPtr_.reset(
        new volScalarField(
            nuIO,
            mesh));
}
else
{
    FatalErrorInFunction
        << "Valid type entries are uniform or field for nu"
        << abort(FatalError);
}

volScalarField& nu = nuPtr_();

Info << "Normalising E : E/rho\n"
     << endl;
volScalarField E(rhoE / rho);

Info << "Calculating Lame's coefficients\n"
     << endl;

muPtr_.reset(
    new volScalarField(
        "solid:mu",
        E / (2.0 * (1.0 + nu))));
lambdaPtr_.reset(new volScalarField(
    "solid:lambda",
    nu* E / ((1.0 + nu) * (1.0 - 2.0 * nu))));

Switch planeStress(mechanicalProperties.lookup("planeStress"));

if (planeStress)
{
    Info << "Plane Stress\n"
         << endl;

    lambdaPtr_() = nu * E / ((1.0 + nu) * (1.0 - nu));
}
else
{
    Info << "Plane Strain\n"
         << endl;
}
