
// Calculates and prints the continuity errors.

{
    volScalarField contErr(fvc::div(phi));

    scalar sumLocalContErr = runTime.deltaTValue() * mag(contErr)().weightedAverage(mesh.V()).value();

    scalar globalContErr = runTime.deltaTValue() * contErr.weightedAverage(mesh.V()).value();
    cumulativeContErr += globalContErr;

    Info << "Time step continuity errors : sum local = " << sumLocalContErr << endl
         << "                                 global = " << globalContErr << endl
         << "                             cumulative = " << cumulativeContErr << endl;
}
