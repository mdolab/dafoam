// Get chi1 and fv11
volScalarField chi1("chi1", chi(nuTilda1, nu));
volScalarField fv11("fv11", fv1(chi1));

// Get Stilda1
volScalarField Stilda1(
    "Stilda1",
    fv3(chi1, fv11) * ::sqrt(2.0) * mag(skew(fvc::grad(U1))) + fv2(chi1, fv11) * nuTilda1 / sqr(kappa * y));

// Construct nuTilda1Eqn w/o ddt term
fvScalarMatrix nuTilda1Eqn(
    fvm::div(phi1, nuTilda1, divNuTildaScheme)
        - fvm::laplacian(DnuTildaEff(nuTilda1, nu), nuTilda1)
        - Cb2 / sigmaNut * magSqr(fvc::grad(nuTilda1))
    == Cb1 * Stilda1 * nuTilda1 // If field inversion, beta should be multiplied here
        - fvm::Sp(Cw1 * fw(Stilda1, nuTilda1, y) * nuTilda1 / sqr(y), nuTilda1));

// Update nuTilda1Eqn with pseudo-spectral terms
forAll(nuTilda1, cellI)
{
    scalar meshV = nuTilda1.mesh().V()[cellI];

    // Add D11 / halfDeltaT[i] * V() to diagonal
    nuTilda1Eqn.diag()[cellI] += D11 / deltaT * meshV;

    // Minus D10 / halfDeltaT[i] * T0 * V() to source term
    nuTilda1Eqn.source()[cellI] -= D10 / deltaT * nuTilda[cellI] * meshV;

    // Minus D12 / halfDeltaT[i] * T2 * V() to source term
    nuTilda1Eqn.source()[cellI] -= D12 / deltaT * nuTilda2[cellI] * meshV;
}

nuTilda1Eqn.relax(relaxNuTildaEqn);

nuTilda1Eqn.solve(solverDictNuTilda);

// Explicitly relax nuTilda1
nuTilda1 = nuTilda1.oldTime() + relaxNuTilda * relaxStage1 * (nuTilda1 - nuTilda1.oldTime());

// Correct nut after solving nuTilda1
correctNut(nut1, nuTilda1, nu);

nuTilda1.oldTime() = nuTilda1;
