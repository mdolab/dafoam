volScalarField rAU2(1.0 / U2Eqn.A());
volVectorField HbyA2(constrainHbyA(rAU2* U2Eqn.H(), U2, p2));
surfaceScalarField phiHbyA2(
    "phiHbyA2",
    fvc::flux(HbyA2));

tmp<volScalarField> rAtU2(rAU2);

// Non-orthogonal pressure corrector loop
fvScalarMatrix p2Eqn(
    fvm::laplacian(rAtU2(), p2) == fvc::div(phiHbyA2));

//p1Eqn.setReference(pRefCell, pRefValue);
p2Eqn.setReference(0, 0.0);

//p1Eqn.solve(mesh.solver(p1.select(pimple.finalInnerIter())));
p2Eqn.solve(solverDictP);

phi2 = phiHbyA2 - p2Eqn.flux();
phi2 = phi2.oldTime() + relaxPhi * relaxStage2 * (phi2 - phi2.oldTime());

// Explicitly relax pressure for momentum corrector
//p2.relax(relaxP);
p2 = p2.oldTime() + relaxP * relaxStage2 * (p2 - p2.oldTime());

U2 = HbyA2 - rAtU2 * fvc::grad(p2);
U2 = U2.oldTime() + relaxU * relaxStage2 * (U2 - U2.oldTime());
U2.correctBoundaryConditions();

phi2.oldTime() = phi2;
p2.oldTime() = p2;
U2.oldTime() = U2;
