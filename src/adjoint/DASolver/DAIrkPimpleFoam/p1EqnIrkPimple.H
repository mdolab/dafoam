volScalarField rAU1(1.0 / U1Eqn.A());
volVectorField HbyA1(constrainHbyA(rAU1* U1Eqn.H(), U1, p1));
surfaceScalarField phiHbyA1(
    "phiHbyA1",
    fvc::flux(HbyA1));

tmp<volScalarField> rAtU1(rAU1);

// Non-orthogonal pressure corrector loop
fvScalarMatrix p1Eqn(
    fvm::laplacian(rAtU1(), p1) == fvc::div(phiHbyA1));

//p1Eqn.setReference(pRefCell, pRefValue);
p1Eqn.setReference(0, 0.0);

//p1Eqn.solve(mesh.solver(p1.select(pimple.finalInnerIter())));
p1Eqn.solve(solverDictP);

phi1 = phiHbyA1 - p1Eqn.flux();
phi1 = phi1.oldTime() + relaxPhi * relaxStage1 * (phi1 - phi1.oldTime());

// Explicitly relax pressure for momentum corrector
//p1.relax(relaxP);
p1 = p1.oldTime() + relaxP * relaxStage1 * (p1 - p1.oldTime());

U1 = HbyA1 - rAtU1 * fvc::grad(p1);
U1 = U1.oldTime() + relaxU * relaxStage1 * (U1 - U1.oldTime());
U1.correctBoundaryConditions();

phi1.oldTime() = phi1;
p1.oldTime() = p1;
U1.oldTime() = U1;
