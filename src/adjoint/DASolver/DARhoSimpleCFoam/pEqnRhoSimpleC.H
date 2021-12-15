rho = thermo.rho();

volScalarField rAU(1.0 / UEqn.A());
volScalarField rAtU(1.0 / (1.0 / rAU - UEqn.H1()));
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

// NOTE: we don't support transonic = false

volScalarField rhorAtU("rhorAtU", rho* rAtU);

surfaceScalarField phid(
    "phid",
    (fvc::interpolate(psi) / fvc::interpolate(rho)) * phiHbyA);

phiHbyA +=
    fvc::interpolate(rho * (rAtU - rAU)) * fvc::snGrad(p) * mesh.magSf()
    - fvc::interpolate(psi * p) * phiHbyA / fvc::interpolate(rho);

HbyA -= (rAU - rAtU) * fvc::grad(p);

while (simple.correctNonOrthogonal())
{
    fvScalarMatrix pEqn(
        fvc::div(phiHbyA)
        + fvm::div(phid, p)
        - fvm::laplacian(rhorAtU, p));

    // Relax the pressure equation to maintain diagonal dominance
    pEqn.relax();

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

U = HbyA - rAtU * fvc::grad(p);
// bound U
DAUtility::boundVar(allOptions, U, printToScreen);
U.correctBoundaryConditions();

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

// bound rho
DAUtility::boundVar(allOptions, rho, printToScreen);
