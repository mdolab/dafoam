// Solve the Momentum equation

if (hasFvSource_)
{
    daFvSourcePtr_->calcFvSource(fvSource);
}

tmp<fvVectorMatrix> tUEqn(
    fvm::ddt(rho, U)
    + fvm::div(phi, U)
    + turbulencePtr_->divDevRhoReff(U)
    - fvSource);
fvVectorMatrix& UEqn = tUEqn.ref();

UEqn.relax();

// get the solver performance info such as initial
// and final residuals
SolverPerformance<vector> solverU = solve(UEqn == -fvc::grad(p));

DAUtility::primalResidualControl(solverU, pimplePrintToScreen, "U", daGlobalVarPtr_->primalMaxRes);

// bound U
DAUtility::boundVar(allOptions, U, pimplePrintToScreen);

K = 0.5 * magSqr(U);
