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

U = HbyA - rAU * fvc::grad(p);
// bound U
DAUtility::boundVar(allOptions, U, printToScreen_);
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
DAUtility::boundVar(allOptions, rho, printToScreen_);
