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
surfaceScalarField phiHbyA(
    "phiHbyA",
    fvc::flux(HbyA));

tmp<volScalarField> rAtU(rAU);

if (pimple.nCorrPISO() <= 1)
{
    tUEqn.clear();
}

// Non-orthogonal pressure corrector loop
while (pimple.correctNonOrthogonal())
{
    fvScalarMatrix pEqn(
        fvm::laplacian(rAtU(), p) == fvc::div(phiHbyA));

    pEqn.setReference(pRefCell, pRefValue);

    // get the solver performance info such as initial
    // and final residuals
    SolverPerformance<scalar> solverP = pEqn.solve(mesh.solver(p.select(pimple.finalInnerIter())));

    DAUtility::primalResidualControl(solverP, pimplePrintToScreen, "p", daGlobalVarPtr_->primalMaxRes);

    if (pimple.finalNonOrthogonalIter())
    {
        phi = phiHbyA - pEqn.flux();
    }
}

if (pimplePrintToScreen)
{
#include "continuityErrsPython.H"
}

// Explicitly relax pressure for momentum corrector
p.relax();

// bound p
DAUtility::boundVar(allOptions, p, pimplePrintToScreen);

U = HbyA - rAtU * fvc::grad(p);
// bound U
DAUtility::boundVar(allOptions, U, pimplePrintToScreen);
U.correctBoundaryConditions();

// Correct Uf if the mesh is moving
this->correctUfPimpleDyM(Uf, U, phi);

// Make the fluxes relative to the mesh motion
fvc::makeRelative(phi, U);
