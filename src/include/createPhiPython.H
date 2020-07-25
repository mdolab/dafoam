/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

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

surfaceScalarField& phi = phiPtr_();
