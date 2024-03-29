// Solve the Momentum equation

tmp<fvVectorMatrix> tUEqn(
    fvm::div(phi, U, divUScheme)
    + turbulencePtr_->divDevRhoReff(U));
fvVectorMatrix& UEqn = tUEqn.ref();

UEqn.relax();

// get the solver performance info such as initial
// and final residuals
SolverPerformance<vector> solverU = solve(UEqn == -fvc::grad(p));

DAUtility::primalResidualControl(solverU, printToScreen, "U");

// bound U
DAUtility::boundVar(allOptions, U, printToScreen);
