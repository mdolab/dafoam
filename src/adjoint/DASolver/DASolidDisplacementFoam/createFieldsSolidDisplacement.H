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
