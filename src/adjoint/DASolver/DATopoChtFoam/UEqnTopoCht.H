// Momentum predictor

MRF.correctBoundaryVelocity(U);

if (hasFvSource_)
{
    daFvSourcePtr_->calcFvSource(fvSource);
}

forAll(alphaPorosity, cellI)
{
    alphaPorosity[cellI] = alphaPorositySolid_ + (alphaPorosityFluid_ - alphaPorositySolid_) * eta[cellI] * (1 + pCoeff_) / (eta[cellI] + pCoeff_);
}
alphaPorosity.correctBoundaryConditions();

tmp<fvVectorMatrix> tUEqn(
    fvm::div(phi, U)
    + fvm::Sp(alphaPorosity, U)
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

    DAUtility::primalResidualControl(solverU, printToScreen_, "U", daGlobalVarPtr_->primalMaxRes);
}

// bound U
DAUtility::boundVar(allOptions, U, printToScreen_);
