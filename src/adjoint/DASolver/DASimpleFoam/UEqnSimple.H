// Momentum predictor

MRF.correctBoundaryVelocity(U);

tmp<fvVectorMatrix> tUEqn(
    fvm::div(phi, U)
    + MRF.DDt(U)
    + turbulencePtr_->divDevReff(U)
    - fvSource);
fvVectorMatrix& UEqn = tUEqn.ref();

UEqn.relax();

if (simple.momentumPredictor())
{
    // get the solver performance info such as initial
    // and final residuals
    SolverPerformance<vector> solverU = solve(UEqn == -fvc::grad(p));

    this->primalResidualControl<vector>(solverU, nSolverIters, printInterval, "U");
}
