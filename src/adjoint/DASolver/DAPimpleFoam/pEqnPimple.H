volScalarField rAU(1.0 / UEqn.A());
volVectorField HbyA(constrainHbyA(rAU* UEqn.H(), U, p));
surfaceScalarField phiHbyA(
    "phiHbyA",
    fvc::flux(HbyA)
        + fvc::interpolate(rAU) * fvc::ddtCorr(U, phi));

adjustPhi(phiHbyA, U, p);

tmp<volScalarField> rAtU(rAU);

if (pimple.consistent())
{
    rAtU = 1.0 / max(1.0 / rAU - UEqn.H1(), 0.1 / rAU);
    phiHbyA +=
        fvc::interpolate(rAtU() - rAU) * fvc::snGrad(p) * mesh.magSf();
    HbyA -= (rAU - rAtU()) * fvc::grad(p);
}

if (pimple.nCorrPISO() <= 1)
{
    tUEqn.clear();
}

// Non-orthogonal pressure corrector loop
while (pimple.correctNonOrthogonal())
{
    // Pressure corrector
    fvScalarMatrix pEqn(
        fvm::laplacian(rAtU(), p) == fvc::div(phiHbyA));

    pEqn.setReference(pRefCell, pRefValue);

    // get the solver performance info such as initial
    // and final residuals
    SolverPerformance<scalar> solverP = pEqn.solve(mesh.solver(p.select(pimple.finalInnerIter())));

    this->primalResidualControl<scalar>(solverP, printToScreen, printInterval, "p");

    if (pimple.finalNonOrthogonalIter())
    {
        phi = phiHbyA - pEqn.flux();
    }
}

if (printToScreen)
{
#include "continuityErrsPython.H"
}

// Explicitly relax pressure for momentum corrector
p.relax();

U = HbyA - rAtU() * fvc::grad(p);
U.correctBoundaryConditions();
