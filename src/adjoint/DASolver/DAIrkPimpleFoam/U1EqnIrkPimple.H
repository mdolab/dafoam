// Initialize U1Eqn w/o ddt term
fvVectorMatrix U1Eqn(
    fvm::div(phi1, U1, divUScheme)
    //+ turbulence->divDevReff(U)
    - fvm::laplacian(nu, U1));

// Update U1Eqn with pseudo-spectral terms
forAll(U1, cellI)
{
    scalar meshV = U1.mesh().V()[cellI];

    // Add D11 / halfDeltaT[i] * V() to diagonal
    U1Eqn.diag()[cellI] += D11 / deltaT * meshV; // Use one seg for now: halfDeltaTList[segI]

    // Minus D10 / halfDeltaT[i] * T0 * V() to source term
    U1Eqn.source()[cellI] -= D10 / deltaT * U[cellI] * meshV;

    // Minus D12 / halfDeltaT[i] * T2 * V() to source term
    U1Eqn.source()[cellI] -= D12 / deltaT * U2[cellI] * meshV;
}

U1Eqn.relax(relaxUEqn);

solve(U1Eqn == -fvc::grad(p1), solverDictU);
