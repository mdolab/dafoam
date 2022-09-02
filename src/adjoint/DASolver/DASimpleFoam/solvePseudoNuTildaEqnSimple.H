// Overwrite the r.h.s.
pseudo_nuTildaEqn.source() = nuTildaSource;

// Make sure that boundary contribution to source is zero,
// Alternatively, we can deduct source by boundary contribution, so that it would cancel out during solve.
forAll(pseudo_nuTilda.boundaryField(), patchI)
{
    const fvPatch& pp = pseudo_nuTilda.boundaryField()[patchI].patch();
    forAll(pp, faceI)
    {
        label cellI = pp.faceCells()[faceI];
        //myDiag[cellI] += TEqn.internalCoeffs()[patchI][faceI];
        pseudo_nuTildaEqn.source()[cellI] -= pseudo_nuTildaEqn.boundaryCoeffs()[patchI][faceI];
        //pseudo_nuTildaEqn.boundaryCoeffs()[patchI][faceI] = 0;
    }
}

// Before solve, force xxEqn.psi to be solved into all zero
// This ensures the zero (internal) initial guess
forAll(pseudo_nuTilda.primitiveFieldRef(), cellI)
{
    pseudo_nuTilda.primitiveFieldRef()[cellI] = 0;
}
// Solve using the zero (internal) initial guess
pseudo_nuTildaEqn.solve();