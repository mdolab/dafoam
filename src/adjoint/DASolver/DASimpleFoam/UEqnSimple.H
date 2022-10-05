// Momentum predictor

MRF.correctBoundaryVelocity(U);

if (hasFvSource_)
{
    daFvSourcePtr_->calcFvSource(fvSource);
}

tmp<fvVectorMatrix> tUEqn(
    fvm::div(phi, U, divUScheme)
    + fvm::Sp(alphaPorosity, U)
    + MRF.DDt(U)
    + turbulencePtr_->divDevReff(U)
    - fvSource
    - fvOptions(U));
fvVectorMatrix& UEqn = tUEqn.ref();

UEqn.relax();

fvOptions.constrain(UEqn);

if (simple.momentumPredictor())
{
    // get the solver performance info such as initial
    // and final residuals
    SolverPerformance<vector> solverU = solve(UEqn == -fvc::grad(p));

    this->primalResidualControl<vector>(solverU, printToScreen, printInterval, "U");

    fvOptions.correct(U);
}
