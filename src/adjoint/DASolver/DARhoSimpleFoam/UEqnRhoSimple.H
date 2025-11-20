// Solve the Momentum equation

MRF.correctBoundaryVelocity(U);

if (hasFvSource_)
{
    daFvSourcePtr_->calcFvSource(fvSource);
}

tmp<fvVectorMatrix> tUEqn(
    fvm::div(phi, U, divUScheme)
    + MRF.DDt(rho, U)
    + turbulencePtr_->divDevRhoReff(U)
    - fvSource
    - fvOptions(rho, U));
fvVectorMatrix& UEqn = tUEqn.ref();

UEqn.relax();

fvOptions.constrain(UEqn);

// get the solver performance info such as initial
// and final residuals
SolverPerformance<vector> solverU = solve(UEqn == -fvc::grad(p));

this->primalResidualControl<vector>(solverU, printToScreen, printInterval, "U");

fvOptions.correct(U);

// bound U
DAUtility::boundVar(allOptions, U, printToScreen);
