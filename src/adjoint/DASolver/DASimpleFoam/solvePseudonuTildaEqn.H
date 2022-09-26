// Overwrite the r.h.s.
pseudoNuTildaEqn.source() = nuTildaSource.primitiveField();

// Make sure that boundary contribution to source is zero,
// Alternatively, we can deduct source by boundary contribution, so that it would cancel out during solve.
forAll(pseudoNuTilda.boundaryField(), patchI)
{
    const fvPatch& pp = pseudoNuTilda.boundaryField()[patchI].patch();
    forAll(pp, faceI)
    {
        label cellI = pp.faceCells()[faceI];
        //myDiag[cellI] += TEqn.internalCoeffs()[patchI][faceI];
        pseudoNuTildaEqn.source()[cellI] -= pseudoNuTildaEqn.boundaryCoeffs()[patchI][faceI];
        //pseudoNuTildaEqn.boundaryCoeffs()[patchI][faceI] = 0;
    }
}

// Before solve, force xxEqn.psi to be solved into all zero
// This ensures the zero (internal) initial guess
forAll(pseudoNuTilda.primitiveFieldRef(), cellI)
{
    pseudoNuTilda.primitiveFieldRef()[cellI] = 0;
}
// Solve using the zero (internal) initial guess
pseudoNuTildaEqn.solve();