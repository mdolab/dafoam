volScalarField rAU(1.0 / UEqn.A());
volVectorField HbyA(constrainHbyA(rAU* UEqn.H(), U, p));
surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA));

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

    this->primalResidualControl<scalar>(solverP, nSolverIters, printInterval, "p");

    if (piso.finalNonOrthogonalIter())
    {
        phi = phiHbyA - pEqn.flux();
    }
}

if (nSolverIters % printInterval == 0 || nSolverIters == 1)
{
#include "continuityErrsPython.H"
}

U = HbyA - rAU * fvc::grad(p);
U.correctBoundaryConditions();
