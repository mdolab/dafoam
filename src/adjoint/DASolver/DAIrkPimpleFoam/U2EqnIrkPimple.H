// Get nuEff2
volScalarField nuEff2("nuEff2", nu + nut2);

// Initialize U2Eqn w/o ddt term
fvVectorMatrix U2Eqn(
    fvm::div(phi2, U2, divUScheme)
    //+ turbulence->divDevReff(U)
    - fvm::laplacian(nuEff2, U2)
    - fvc::div(nuEff2 * dev2(T(fvc::grad(U2))), divGradUScheme));

// Update U2Eqn with pseudo-spectral terms
forAll(U2, cellI)
{
    scalar meshV = U2.mesh().V()[cellI];

    // Add D22 / halfDeltaT[i] * V() to diagonal
    U2Eqn.diag()[cellI] += D22 / deltaT * meshV; // Use one seg for now: halfDeltaTList[segI]

    // Minus D20 / halfDeltaT[i] * T0 * V() to source term
    U2Eqn.source()[cellI] -= D20 / deltaT * U[cellI] * meshV;

    // Minus D21 / halfDeltaT[i] * T2 * V() to source term
    U2Eqn.source()[cellI] -= D21 / deltaT * U1[cellI] * meshV;
}

U2Eqn.relax(relaxUEqn);

solve(U2Eqn == -fvc::grad(p2), solverDictU);
