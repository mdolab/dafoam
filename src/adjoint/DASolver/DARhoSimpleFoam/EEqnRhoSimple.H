{
    if (hasFvSource_)
    {
        fvSourceEnergy = fvSource & U;
    }

    volScalarField& he = thermo.he();

    fvScalarMatrix EEqn(
        fvm::div(phi, he)
        + (he.name() == "e"
               ? fvc::div(phi, volScalarField("Ekp", 0.5 * magSqr(U) + p / rho))
               : fvc::div(phi, volScalarField("K", 0.5 * magSqr(U))))
        - fvm::laplacian(turbulencePtr_->alphaEff(), he)
        - fvSourceEnergy);

    EEqn.relax();

    // get the solver performance info such as initial
    // and final residuals
    SolverPerformance<scalar> solverE = EEqn.solve();

    this->primalResidualControl<scalar>(solverE, printToScreen, printInterval, "he");

    // bound he
    DAUtility::boundVar(allOptions, he, printToScreen);

    thermo.correct();
}
