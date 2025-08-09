{
    volScalarField rAU("rAU", 1.0 / UEqn.A());
    surfaceScalarField rAUf("rAUf", fvc::interpolate(rAU));

    //***************** NOTE *******************
    // constrainHbyA has been used since OpenFOAM-v1606; however, it may degrade the accuracy of derivatives
    // because constraining variables will create discontinuity. Here we have a option to use the old
    // implementation in OpenFOAM-3.0+ and before (no constraint for HbyA)
    autoPtr<volVectorField> HbyAPtr = nullptr;
    label useConstrainHbyA = daOptionPtr_->getOption<label>("useConstrainHbyA");
    if (useConstrainHbyA)
    {
        HbyAPtr.reset(new volVectorField(constrainHbyA(rAU * UEqn.H(), U, p_rgh)));
    }
    else
    {
        HbyAPtr.reset(new volVectorField("HbyA", U));
        HbyAPtr() = rAU * UEqn.H();
    }
    volVectorField& HbyA = HbyAPtr();

    surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA));

    adjustPhi(phiHbyA, U, p_rgh);

    surfaceScalarField phig(
        (
            mixture.surfaceTensionForce()
            - ghf * fvc::snGrad(rho))
        * rAUf * mesh.magSf());

    phiHbyA += phig;

    // Update the pressure BCs to ensure flux consistency
    constrainPressure(p_rgh, U, phiHbyA, rAUf);

    while (pimple.correctNonOrthogonal())
    {
        fvScalarMatrix p_rghEqn(
            fvm::laplacian(rAUf, p_rgh) == fvc::div(phiHbyA));

        p_rghEqn.setReference(pRefCell, getRefCellValue(p_rgh, pRefCell));

        SolverPerformance<scalar> solverP =
            p_rghEqn.solve(mesh.solver(p_rgh.select(pimple.finalInnerIter())));

        DAUtility::primalResidualControl(solverP, pimplePrintToScreen, "p_rgh", daGlobalVarPtr_->primalMaxRes);

        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA - p_rghEqn.flux();

            p_rgh.relax();

            // bound p
            DAUtility::boundVar(allOptions, p_rgh, pimplePrintToScreen);

            U = HbyA + rAU * fvc::reconstruct((phig - p_rghEqn.flux()) / rAUf);
            // bound U
            DAUtility::boundVar(allOptions, U, pimplePrintToScreen);
            U.correctBoundaryConditions();
        }
    }

    if (pimplePrintToScreen)
    {
#include "continuityErrsPython.H"
    }

    p == p_rgh + rho* gh;

    if (p_rgh.needReference())
    {
        p += dimensionedScalar(
            "p",
            p.dimensions(),
            pRefValue - getRefCellValue(p, pRefCell));
        p_rgh = p - rho * gh;
    }
}
