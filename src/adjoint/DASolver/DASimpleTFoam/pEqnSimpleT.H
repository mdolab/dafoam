{
    volScalarField rAU(1.0 / UEqn.A());
    //volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
    //***************** NOTE *******************
    // constrainHbyA has been used since OpenFOAM-v1606; however, We do NOT use the constrainHbyA
    // function in DAFoam because we found it significantly degrades the accuracy of shape derivatives. 
    // Basically, we should not constrain any variable because it will create discontinuity.
    // Instead, we use the old implementation used in OpenFOAM-3.0+ and before
    volVectorField HbyA("HbyA", U);
    HbyA = rAU * UEqn.H();
    surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA));

    MRF.makeRelative(phiHbyA);

    adjustPhi(phiHbyA, U, p);

    tmp<volScalarField> rAtU(rAU);

    if (simple.consistent())
    {
        rAtU = 1.0 / (1.0 / rAU - UEqn.H1());
        phiHbyA +=
            fvc::interpolate(rAtU() - rAU) * fvc::snGrad(p) * mesh.magSf();
        HbyA -= (rAU - rAtU()) * fvc::grad(p);
    }

    tUEqn.clear();

    // Update the pressure BCs to ensure flux consistency
    constrainPressure(p, U, phiHbyA, rAtU(), MRF);

    // Non-orthogonal pressure corrector loop
    while (simple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn(
            fvm::laplacian(rAtU(), p) == fvc::div(phiHbyA));

        pEqn.setReference(pRefCell, pRefValue);

        // get the solver performance info such as initial
        // and final residuals
        SolverPerformance<scalar> solverP = pEqn.solve();

        this->primalResidualControl<scalar>(solverP, printToScreen, printInterval, "p");

        if (simple.finalNonOrthogonalIter())
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

    // Momentum corrector
    U = HbyA - rAtU() * fvc::grad(p);
    U.correctBoundaryConditions();
}
