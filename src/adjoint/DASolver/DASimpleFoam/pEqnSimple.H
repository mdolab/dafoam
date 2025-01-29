{
    volScalarField rAU(1.0 / UEqn.A());
    //***************** NOTE *******************
    // constrainHbyA has been used since OpenFOAM-v1606; however, it may degrade the accuracy of derivatives
    // because constraining variables will create discontinuity. Here we have a option to use the old
    // implementation in OpenFOAM-3.0+ and before (no constraint for HbyA)
    autoPtr<volVectorField> HbyAPtr = nullptr;
    label useConstrainHbyA = daOptionPtr_->getOption<label>("useConstrainHbyA");
    if (useConstrainHbyA)
    {
        HbyAPtr.reset(new volVectorField(constrainHbyA(rAU * UEqn.H(), U, p)));
    }
    else
    {
        HbyAPtr.reset(new volVectorField("HbyA", U));
        HbyAPtr() = rAU * UEqn.H();
    }
    volVectorField& HbyA = HbyAPtr();
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

        DAUtility::primalResidualControl(solverP, printToScreen_, "p", daGlobalVarPtr_->primalMaxRes);

        if (simple.finalNonOrthogonalIter())
        {
            phi = phiHbyA - pEqn.flux();
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

    // Momentum corrector
    U = HbyA - rAtU() * fvc::grad(p);
    // bound U
    DAUtility::boundVar(allOptions, U, printToScreen_);
    U.correctBoundaryConditions();
    fvOptions.correct(U);
}
