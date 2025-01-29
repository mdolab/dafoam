// Solve the Momentum equation

MRF.correctBoundaryVelocity(U);

tmp<fvVectorMatrix> tUEqn(
    fvm::div(phi, U)
    + MRF.DDt(rho, U)
    + turbulencePtr_->divDevRhoReff(U));
fvVectorMatrix& UEqn = tUEqn.ref();

UEqn.relax();

// get the solver performance info such as initial
// and final residuals
SolverPerformance<vector> solverU = solve(UEqn == -fvc::grad(p));

DAUtility::primalResidualControl(solverU, printToScreen_, "U", daGlobalVarPtr_->primalMaxRes);

// bound U
DAUtility::boundVar(allOptions, U, printToScreen_);
U.correctBoundaryConditions();
