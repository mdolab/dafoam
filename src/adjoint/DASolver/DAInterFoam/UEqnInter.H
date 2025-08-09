
fvVectorMatrix UEqn(
    fvm::ddt(rho, U)
    + fvm::div(rhoPhi, U)
    + turbulencePtr_->divDevRhoReff(rho, U));

UEqn.relax();

if (pimple.momentumPredictor())
{
    SolverPerformance<vector> solverU = solve(
        UEqn
        == fvc::reconstruct(
            (
                mixture.surfaceTensionForce()
                - ghf * fvc::snGrad(rho)
                - fvc::snGrad(p_rgh))
            * mesh.magSf()));

    DAUtility::primalResidualControl(
        solverU, pimplePrintToScreen, "U", daGlobalVarPtr_->primalMaxRes);
}

// bound U
DAUtility::boundVar(allOptions, U, pimplePrintToScreen);
