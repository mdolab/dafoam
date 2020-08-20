// Solve the Momentum equation

fvVectorMatrix UEqn(
    fvm::ddt(U)
    + fvm::div(phi, U)
    + turbulencePtr_->divDevReff(U));

UEqn.relax();

if (piso.momentumPredictor())
{
    // get the solver performance info such as initial
    // and final residuals
    SolverPerformance<vector> solverU = solve(UEqn == -fvc::grad(p));

    this->primalResidualControl<vector>(solverU, nSolverIters, printInterval, "U");
}