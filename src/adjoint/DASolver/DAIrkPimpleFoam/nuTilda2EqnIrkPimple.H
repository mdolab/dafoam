// Get chi2 and fv12
volScalarField chi2("chi2", chi(nuTilda2, nu));
volScalarField fv12("fv12", fv1(chi2));

// Get Stilda2
volScalarField Stilda2(
    "Stilda2",
    fv3(chi2, fv12) * ::sqrt(2.0) * mag(skew(fvc::grad(U2))) + fv2(chi2, fv12) * nuTilda2 / sqr(kappa * y));

// Construct nuTilda2Eqn w/o ddt term
fvScalarMatrix nuTilda2Eqn(
    fvm::div(phi2, nuTilda2, divNuTildaScheme)
        - fvm::laplacian(DnuTildaEff(nuTilda2, nu), nuTilda2)
        - Cb2 / sigmaNut * magSqr(fvc::grad(nuTilda2))
    == Cb1 * Stilda2 * nuTilda2 // If field inversion, beta should be multiplied here
        - fvm::Sp(Cw1 * fw(Stilda2, nuTilda2, y) * nuTilda2 / sqr(y), nuTilda2));

// Update nuTilda2Eqn with pseudo-spectral terms
forAll(nuTilda2, cellI)
{
    scalar meshV = nuTilda2.mesh().V()[cellI];

    // Add D22 / halfDeltaT[i] * V() to diagonal
    nuTilda2Eqn.diag()[cellI] += D22 / deltaT * meshV;

    // Minus D20 / halfDeltaT[i] * T0 * V() to source term
    nuTilda2Eqn.source()[cellI] -= D20 / deltaT * nuTilda[cellI] * meshV;

    // Minus D21 / halfDeltaT[i] * T2 * V() to source term
    nuTilda2Eqn.source()[cellI] -= D21 / deltaT * nuTilda1[cellI] * meshV;
}

nuTilda2Eqn.relax(relaxNuTildaEqn);

nuTilda2Eqn.solve(solverDictNuTilda);

// Explicitly relax nuTilda2
nuTilda2 = nuTilda2.oldTime() + relaxNuTilda * relaxStage2 * (nuTilda2 - nuTilda2.oldTime());

// Correct nut after solving nuTilda2
correctNut(nut2, nuTilda2, nu);

nuTilda2.oldTime() = nuTilda2;
