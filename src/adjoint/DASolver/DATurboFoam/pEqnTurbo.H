rho = thermo.rho();
// bound rho
DAUtility::boundVar(allOptions, rho, printToScreen_);

if (!simple.transonic())
{
    rho.relax();
}

volScalarField p0(p);

volScalarField AU(UEqn.A());
volScalarField AtU(AU - UEqn.H1());
volVectorField HbyA("HbyA", U);
HbyA = UEqn.H() / AU;

volScalarField rAU(1.0 / UEqn.A());
tUEqn.clear();

bool closedVolume = false;

if (simple.transonic())
{

    surfaceScalarField phid(
        "phid",
        fvc::interpolate(psi) * (fvc::interpolate(HbyA) & mesh.Sf()));

    MRF.makeRelative(fvc::interpolate(psi), phid);

    while (simple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn(
            fvm::div(phid, p)
            - fvm::laplacian(rho * rAU, p));

        // Relax the pressure equation to ensure diagonal-dominance
        pEqn.relax();

        pEqn.setReference(
            pressureControl.refCell(),
            pressureControl.refValue());

        // get the solver performance info such as initial
        // and final residuals
        SolverPerformance<scalar> solverP = pEqn.solve();

        DAUtility::primalResidualControl(solverP, printToScreen_, "p", daGlobalVarPtr_->primalMaxRes);

        if (simple.finalNonOrthogonalIter())
        {
            phi == pEqn.flux();
        }
    }
}
else
{
    surfaceScalarField phiHbyA(
        "phiHbyA",
        fvc::interpolate(rho * HbyA) & mesh.Sf());

    MRF.makeRelative(fvc::interpolate(rho), phiHbyA);

    closedVolume = adjustPhi(phiHbyA, U, p);
    phiHbyA += fvc::interpolate(rho / AtU - rho / AU) * fvc::snGrad(p) * mesh.magSf();

    while (simple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn(
            fvc::div(phiHbyA)
            - fvm::laplacian(rho / AtU, p));

        pEqn.setReference(
            pressureControl.refCell(),
            pressureControl.refValue());

        // get the solver performance info such as initial
        // and final residuals
        SolverPerformance<scalar> solverP = pEqn.solve();

        DAUtility::primalResidualControl(solverP, printToScreen_, "p", daGlobalVarPtr_->primalMaxRes);

        if (simple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
    }
}

if (printToScreen_)
{
#include "continuityErrsPython.H"
}

// Explicitly relax pressure for momentum corrector
p.relax();

// bound p
DAUtility::boundVar(allOptions, p, printToScreen_);

U = HbyA - (fvc::grad(p0) * (1.0 / AU - 1.0 / AtU) + fvc::grad(p) / AtU);

// bound U
DAUtility::boundVar(allOptions, U, printToScreen_);
U.correctBoundaryConditions();

// For closed-volume cases adjust the pressure and density levels
// to obey overall mass continuity
if (closedVolume)
{
    p += (initialMass_ - fvc::domainIntegrate(psi * p))
        / fvc::domainIntegrate(psi);
}

rho = thermo.rho();
DAUtility::boundVar(allOptions, rho, printToScreen_);

if (!simple.transonic())
{
    rho.relax();
}
