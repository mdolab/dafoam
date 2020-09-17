{
    alphat = turbulencePtr_->nut() / Prt;
    alphat.correctBoundaryConditions();

    volScalarField alphaEff("alphaEff", turbulencePtr_->nu() / Pr + alphat);

    fvScalarMatrix TEqn(
        fvm::div(phi, T)
        - fvm::laplacian(alphaEff, T)
    );

    TEqn.relax();

    // get the solver performance info such as initial
    // and final residuals
    SolverPerformance<scalar> solverT = TEqn.solve();

    this->primalResidualControl<scalar>(solverT, printToScreen, printInterval, "T");
}
