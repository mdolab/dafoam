rho = thermo.rho();

volScalarField rAU(1.0 / UEqn.A());
volScalarField rAtU(1.0 / (1.0 / rAU - UEqn.H1()));
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

    DAUtility::primalResidualControl(solverP, printToScreen_, "p", daGlobalVarPtr_->primalMaxRes);

    if (simple.finalNonOrthogonalIter())
    {
        phi = phiHbyA + pEqn.flux();
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

U = HbyA - rAtU * fvc::grad(p);
// bound U
DAUtility::boundVar(allOptions, U, printToScreen_);
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
DAUtility::boundVar(allOptions, rho, printToScreen_);
