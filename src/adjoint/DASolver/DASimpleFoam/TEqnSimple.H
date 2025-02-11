{
    volScalarField& T = TPtr_();
    dimensionedScalar Pr = PrPtr_();
    dimensionedScalar Prt = PrtPtr_();
    volScalarField& alphat = alphatPtr_();

    alphat = turbulencePtr_->nut() / Prt;
    alphat.correctBoundaryConditions();

    volScalarField alphaEff("alphaEff", turbulencePtr_->nu() / Pr + alphat);

    fvScalarMatrix TEqn(
        fvm::div(phi, T)
        - fvm::laplacian(alphaEff, T));

    TEqn.relax();

    // get the solver performance info such as initial
    // and final residuals
    SolverPerformance<scalar> solverT = TEqn.solve();

    DAUtility::primalResidualControl(solverT, printToScreen_, "T", daGlobalVarPtr_->primalMaxRes);

    // bound T
    DAUtility::boundVar(allOptions, T, printToScreen_);
}
