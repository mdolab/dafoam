{
    volScalarField& he = thermo.he();
    volSymmTensorField Teff = -turbulencePtr_->devRhoReff();

    URel == U;
    MRF.makeRelative(URel);

    fvScalarMatrix EEqn(
        fvm::div(phi, he)
        + (he.name() == "e"
               ? fvc::div(phi, volScalarField("Ekp", 0.5 * magSqr(U) + p / rho))
               : fvc::div(phi, volScalarField("K", 0.5 * magSqr(U))) - fvc::div(Teff.T() & U) + fvc::div(p * (U - URel)))
        - fvm::Sp(fvc::div(phi), he)
        - fvm::laplacian(turbulencePtr_->alphaEff(), he));

    EEqn.relax();

    // get the solver performance info such as initial
    // and final residuals
    SolverPerformance<scalar> solverE = EEqn.solve();

    DAUtility::primalResidualControl(solverE, printToScreen_, "he", daGlobalVarPtr_->primalMaxRes);

    // bound he
    DAUtility::boundVar(allOptions, he, printToScreen_);

    thermo.correct();
}
