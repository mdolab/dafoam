{
    volSymmTensorField sigmaD = mu * twoSymm(gradD) + (lambda * I) * tr(gradD);

    volSymmTensorField sigma(
        IOobject(
            "sigma",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE),
        rho * sigmaD);

    volScalarField sigmaEq(
        IOobject(
            "sigmaEq",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE),
        sqrt((3.0 / 2.0) * magSqr(dev(sigma))));

    Info << "Max sigmaEq = " << gMax(sigmaEq) << endl;

    runTime.write();
}