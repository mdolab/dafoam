#include "readMechanicalPropertiesSolidDisplacement.H"

Info << "Reading field D\n"
     << endl;
DPtr_.reset(
    new volVectorField(
        IOobject(
            "D",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE),
        mesh));
volVectorField& D = DPtr_();

Info << "Calculating stress field sigmaD\n"
     << endl;

sigmaDPtr_.reset(
    new volSymmTensorField(
        IOobject(
            "sigmaD",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        muPtr_() * twoSymm(fvc::grad(D)) + lambdaPtr_() * (I * tr(fvc::grad(D)))));
volSymmTensorField& sigmaD = sigmaDPtr_();

// gradD is used in the tractionDisplacement BC
gradDPtr_.reset(
    new volTensorField(
        IOobject(
            "gradD",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        fvc::grad(D)));

Info << "Calculating explicit part of div(sigma) divSigmaExp\n"
     << endl;
divSigmaExpPtr_.reset(
    new volVectorField(
    IOobject(
        "divSigmaExp",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    fvc::div(sigmaD)));
volVectorField& divSigmaExp = divSigmaExpPtr_();

if (compactNormalStress_)
{
    divSigmaExp -= fvc::laplacian(2 * muPtr_() + lambdaPtr_(), D, "laplacian(DD,D)");
}
else
{
    divSigmaExp -= fvc::div((2 * muPtr_() + lambdaPtr_()) * fvc::grad(D), "div(sigmaD)");
}

mesh.setFluxRequired(D.name());
