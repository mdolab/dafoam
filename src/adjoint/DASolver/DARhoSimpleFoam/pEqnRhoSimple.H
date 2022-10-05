volScalarField rAU(1.0 / UEqn.A());
surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho* rAU));
//volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
//***************** NOTE *******************
// constrainHbyA has been used since OpenFOAM-v1606; however, We do NOT use the constrainHbyA
// function in DAFoam because we found it significantly degrades the accuracy of shape derivatives.
// Basically, we should not constrain any variable because it will create discontinuity.
// Instead, we use the old implementation used in OpenFOAM-3.0+ and before
volVectorField HbyA("HbyA", U);
HbyA = rAU * UEqn.H();
tUEqn.clear();

bool closedVolume = false;

surfaceScalarField phiHbyA("phiHbyA", fvc::interpolate(rho) * fvc::flux(HbyA));

MRF.makeRelative(fvc::interpolate(rho), phiHbyA);

// Update the pressure BCs to ensure flux consistency
constrainPressure(p, rho, U, phiHbyA, rhorAUf, MRF);

// NOTE: we don't support transonic = true

closedVolume = adjustPhi(phiHbyA, U, p);

while (simple.correctNonOrthogonal())
{
    fvScalarMatrix pEqn(
        fvc::div(phiHbyA)
        - fvm::laplacian(rhorAUf, p)
        - fvOptions(psi, p, rho.name()));

    pEqn.setReference(
        pressureControl.refCell(),
        pressureControl.refValue());

    // get the solver performance info such as initial
    // and final residuals
    SolverPerformance<scalar> solverP = pEqn.solve();

    this->primalResidualControl<scalar>(solverP, printToScreen, printInterval, "p");

    if (simple.finalNonOrthogonalIter())
    {
        phi = phiHbyA + pEqn.flux();
    }
}

if (printToScreen)
{
#include "continuityErrsPython.H"
}

// Explicitly relax pressure for momentum corrector
p.relax();

// bound p
DAUtility::boundVar(allOptions, p, printToScreen);

U = HbyA - rAU * fvc::grad(p);
// bound U
DAUtility::boundVar(allOptions, U, printToScreen);
U.correctBoundaryConditions();
fvOptions.correct(U);

bool pLimited = pressureControl.limit(p);

// For closed-volume cases adjust the pressure and density levels
// to obey overall mass continuity
if (closedVolume)
{
    p += (initialMass_ - fvc::domainIntegrate(psi * p))
        / fvc::domainIntegrate(psi);
}

if (pLimited || closedVolume)
{
    p.correctBoundaryConditions();
}

rho = thermo.rho();

rho.relax();

// bound rho
DAUtility::boundVar(allOptions, rho, printToScreen);
