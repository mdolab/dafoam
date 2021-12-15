// Solve the Momentum equation

if (hasFvSource_)
{
    // update the actuator source term
    daFvSourcePtr_->calcFvSource(fvSource);
}

tmp<fvVectorMatrix> tUEqn(
    fvm::ddt(U)
    + fvm::div(phi, U)
    + turbulencePtr_->divDevReff(U)
    - fvSource);

fvVectorMatrix& UEqn = tUEqn.ref();

UEqn.relax();

if (pimple.momentumPredictor())
{
    // get the solver performance info such as initial
    // and final residuals
    SolverPerformance<vector> solverU = solve(UEqn == -fvc::grad(p));

    this->primalResidualControl<vector>(solverU, printToScreen, printInterval, "U");
}
