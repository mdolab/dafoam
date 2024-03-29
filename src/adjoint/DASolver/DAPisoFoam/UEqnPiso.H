// Solve the Momentum equation

if (hasFvSource_)
{
    // update the actuator source term
    daFvSourcePtr_->calcFvSource(fvSource);
}

fvVectorMatrix UEqn(
    fvm::ddt(U)
    + fvm::div(phi, U)
    + turbulencePtr_->divDevReff(U)
    - fvSource);

UEqn.relax();

if (piso.momentumPredictor())
{
    // get the solver performance info such as initial
    // and final residuals
    SolverPerformance<vector> solverU = solve(UEqn == -fvc::grad(p));

    DAUtility::primalResidualControl(solverU, printToScreen, "U");
}

// bound U
DAUtility::boundVar(allOptions, U, printToScreen);
