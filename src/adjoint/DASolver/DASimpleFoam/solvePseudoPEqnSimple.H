// Overwrite the r.h.s.
pseudo_pEqn.source() = pSource;

// pEqn.setReference(pRefCell, pRefValue);
// Here, pRefCell is a label, and pRefValue is a scalar
// In actual implementation, they need to passed into this function.
pseudo_pEqn.setReference(0, 0.0);

// Make sure that boundary contribution to source is zero,
// Alternatively, we can deduct source by boundary contribution, so that it would cancel out during solve.
forAll(pseudo_p.boundaryField(), patchI)
{
    const fvPatch& pp = pseudo_p.boundaryField()[patchI].patch();
    forAll(pp, faceI)
    {
        label cellI = pp.faceCells()[faceI];
        //myDiag[cellI] += TEqn.internalCoeffs()[patchI][faceI];
        pseudo_pEqn.source()[cellI] -= pseudo_pEqn.boundaryCoeffs()[patchI][faceI];
        //pseudo_pEqn.boundaryCoeffs()[patchI][faceI] = 0;
    }
}

// Before solve, force xEqn.psi() to be solved into all zero
forAll(pseudo_p.primitiveFieldRef(), cellI)
{
    pseudo_p.primitiveFieldRef()[cellI] = 0;
}

pseudo_pEqn.solve();