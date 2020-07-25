
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
        linearInterpolate(rho* U) & mesh.Sf()));

surfaceScalarField& phi = phiPtr_();
// ************************************************************************* //
