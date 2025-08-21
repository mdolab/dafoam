
diffusivityFluid = turbulencePtr_->nut() / Prt;
diffusivityFluid.correctBoundaryConditions();

volScalarField diffusivityFluidEff("alphaEff", turbulencePtr_->nu() / Pr + diffusivityFluid);

forAll(diffusivity, cellI)
{
    diffusivity[cellI] = diffusivitySolid_ + (diffusivityFluidEff[cellI] - diffusivitySolid_) * eta[cellI] * (1 + pCoeff_) / (eta[cellI] + pCoeff_);
}
diffusivity.correctBoundaryConditions();

fvScalarMatrix TEqn(
    fvm::div(phi, T)
    - fvm::laplacian(diffusivity, T));

TEqn.relax();

// get the solver performance info such as initial
// and final residuals
SolverPerformance<scalar> solverT = TEqn.solve();

DAUtility::primalResidualControl(solverT, printToScreen_, "T", daGlobalVarPtr_->primalMaxRes);

// bound T
DAUtility::boundVar(allOptions, T, printToScreen_);
