rho = thermo.rho();
DAUtility::boundVar(allOptions, rho, pimplePrintToScreen);
rho.relax();

volScalarField rAU(1.0 / UEqn.A());
surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho* rAU));
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

if (pimple.nCorrPISO() <= 1)
{
    tUEqn.clear();
}

surfaceScalarField phiHbyA("phiHbyA", fvc::interpolate(rho) * fvc::flux(HbyA));

// NOTE: we don't support transonic = true

while (pimple.correctNonOrthogonal())
{
    fvScalarMatrix pEqn(
        fvm::ddt(psi, p)
        + fvc::div(phiHbyA)
        - fvm::laplacian(rhorAUf, p));

    // get the solver performance info such as initial
    // and final residuals
    SolverPerformance<scalar> solverP = pEqn.solve(mesh.solver(p.select(pimple.finalInnerIter())));

    DAUtility::primalResidualControl(solverP, pimplePrintToScreen, "p", daGlobalVarPtr_->primalMaxRes);

    if (pimple.finalNonOrthogonalIter())
    {
        phi = phiHbyA + pEqn.flux();
    }
}

#include "rhoEqnRhoPimple.H"

if (pimplePrintToScreen)
{
#include "continuityErrsPython.H"
}

// Explicitly relax pressure for momentum corrector
p.relax();

// bound p
DAUtility::boundVar(allOptions, p, pimplePrintToScreen);

rho = thermo.rho();
DAUtility::boundVar(allOptions, rho, pimplePrintToScreen);
rho.relax();

U = HbyA - rAU * fvc::grad(p);
// bound U
DAUtility::boundVar(allOptions, U, pimplePrintToScreen);
U.correctBoundaryConditions();

K = 0.5 * magSqr(U);

if (thermo.dpdt())
{
    dpdt = fvc::ddt(p);
}
