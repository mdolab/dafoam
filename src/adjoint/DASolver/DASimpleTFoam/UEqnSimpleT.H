// Momentum predictor

tmp<fvVectorMatrix> tUEqn(
    fvm::div(phi, U)
    + fvm::Sp(alphaPorosity, U)
    + turbulencePtr_->divDevReff(U));
fvVectorMatrix& UEqn = tUEqn.ref();

UEqn.relax();

if (simple.momentumPredictor())
{
    // get the solver performance info such as initial
    // and final residuals
    SolverPerformance<vector> solverU = solve(UEqn == -fvc::grad(p));

    this->primalResidualControl<vector>(solverU, printToScreen, printInterval, "U");
}
