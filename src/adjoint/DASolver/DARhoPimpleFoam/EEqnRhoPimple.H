{

    fvSourceEnergy = fvSource & U;

    volScalarField& he = thermo.he();

    fvScalarMatrix EEqn(
        fvm::ddt(rho, he)
        + fvm::div(phi, he)
        + fvc::ddt(rho, K)
        + fvc::div(phi, K)
        + (he.name() == "e"
               ? fvc::div(
                   fvc::absolute(phi / fvc::interpolate(rho), U),
                   p,
                   "div(phiv,p)")
               : -dpdt)
        - fvm::laplacian(turbulencePtr_->alphaEff(), he)
        - fvSourceEnergy);

    EEqn.relax();

    // get the solver performance info such as initial
    // and final residuals
    SolverPerformance<scalar> solverE = EEqn.solve();

    DAUtility::primalResidualControl(solverE, pimplePrintToScreen, "he", daGlobalVarPtr_->primalMaxRes);

    // bound he
    DAUtility::boundVar(allOptions, he, pimplePrintToScreen);

    thermo.correct();
}
