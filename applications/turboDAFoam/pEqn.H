rho = thermo.rho();
adjIO.boundVar(rho);

if (!simple.transonic())
{
    rho.relax();
}

volScalarField p0(p);

volScalarField AU(UEqn.A());
volScalarField AtU(AU - UEqn.H1());
volVectorField HbyA = UEqn.H()/AU;

volScalarField rAU(1.0/UEqn.A());
tUEqn.clear();

bool closedVolume = false;

if (simple.transonic())
{

    surfaceScalarField phid
    (
        "phid",
        fvc::interpolate(psi)*(fvc::interpolate(HbyA) & mesh.Sf())
    );

    MRF.makeRelative(fvc::interpolate(psi), phid);

    while (simple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn
        (
            fvm::div(phid, p)
          - fvm::laplacian(rho*rAU, p)
          ==
            fvOptions(psi, p, rho.name())
        );

        // Relax the pressure equation to ensure diagonal-dominance
        pEqn.relax();

        fvOptions.constrain(pEqn);

        pEqn.setReference
        (
            pressureControl.refCell(),
            pressureControl.refValue()
        );

        pEqn.solve();

        if (simple.finalNonOrthogonalIter())
        {
            phi == pEqn.flux();
        }
    }
}
else
{
    surfaceScalarField phiHbyA
    (
        "phiHbyA",
        fvc::interpolate(rho*HbyA) & mesh.Sf()
    );

    MRF.makeRelative(fvc::interpolate(rho), phiHbyA);

    closedVolume = adjustPhi(phiHbyA, U, p);
    phiHbyA += fvc::interpolate(rho/AtU - rho/AU)*fvc::snGrad(p)*mesh.magSf();

    while (simple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn
        (
            fvc::div(phiHbyA)
          - fvm::laplacian(rho/AtU, p)
          ==
            fvOptions(psi, p, rho.name())
        );

        pEqn.setReference
        (
            pressureControl.refCell(),
            pressureControl.refValue()
        );

        fvOptions.constrain(pEqn);

        pEqn.solve();

        if (simple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
    }
}

#include "incompressible/continuityErrs.H"

// Explicitly relax pressure for momentum corrector
p.relax();

// bound p
adjIO.boundVar(p);

U = HbyA - (fvc::grad(p0)*(1.0/AU - 1.0/AtU) + fvc::grad(p)/AtU);

fvOptions.correct(U);

// bound U
adjIO.boundVar(U);
U.correctBoundaryConditions();


// For closed-volume cases adjust the pressure and density levels
// to obey overall mass continuity
if (closedVolume)
{
    p += (initialMass - fvc::domainIntegrate(psi*p))
        /fvc::domainIntegrate(psi);
}

rho = thermo.rho();
adjIO.boundVar(rho);

if (!simple.transonic())
{
    rho.relax();
}
