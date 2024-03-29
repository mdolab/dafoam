volScalarField rAU(1.0 / UEqn.A());
volVectorField HbyA(constrainHbyA(rAU* UEqn.H(), U, p));
surfaceScalarField phiHbyA(
    "phiHbyA",
    fvc::flux(HbyA)
        + fvc::interpolate(rAU) * fvc::ddtCorr(U, phi));

adjustPhi(phiHbyA, U, p);

// Non-orthogonal pressure corrector loop
while (piso.correctNonOrthogonal())
{
    // Pressure corrector

    fvScalarMatrix pEqn(
        fvm::laplacian(rAU, p) == fvc::div(phiHbyA));

    pEqn.setReference(pRefCell, pRefValue);

    // get the solver performance info such as initial
    // and final residuals
    SolverPerformance<scalar> solverP = pEqn.solve(mesh.solver(p.select(piso.finalInnerIter())));

    DAUtility::primalResidualControl(solverP, printToScreen, "p");

    if (piso.finalNonOrthogonalIter())
    {
        phi = phiHbyA - pEqn.flux();
    }
}

if (printToScreen)
{
#include "continuityErrsPython.H"
}

// bound p
DAUtility::boundVar(allOptions, p, printToScreen);

U = HbyA - rAU * fvc::grad(p);
// bound U
DAUtility::boundVar(allOptions, U, printToScreen);
U.correctBoundaryConditions();
