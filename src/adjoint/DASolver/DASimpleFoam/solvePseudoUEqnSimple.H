// Overwrite the r.h.s.
pseudo_UEqn.source() = USource;

// Make sure that boundary contribution to source is zero,
// Alternatively, we can deduct source by boundary contribution, so that it would cancel out during solve.
forAll(pseudo_U.boundaryField(), patchI)
{
    const fvPatch& pp = pseudo_U.boundaryField()[patchI].patch();
    forAll(pp, faceI)
    {
        label cellI = pp.faceCells()[faceI];
        //myDiag[cellI] += TEqn.internalCoeffs()[patchI][faceI];
        pseudo_UEqn.source()[cellI] -= pseudo_UEqn.boundaryCoeffs()[patchI][faceI];
        //pseudo_UEqn.boundaryCoeffs()[patchI][faceI][0] = 0;
        //pseudo_UEqn.boundaryCoeffs()[patchI][faceI][1] = 0;
        //pseudo_UEqn.boundaryCoeffs()[patchI][faceI][2] = 0;
    }
}

// Before solve, force xEqn.psi() to be solved into all zero
forAll(pseudo_U.primitiveFieldRef(), cellI)
{
    pseudo_U.primitiveFieldRef()[cellI][0] = 0;
    pseudo_U.primitiveFieldRef()[cellI][1] = 0;
    pseudo_U.primitiveFieldRef()[cellI][2] = 0;
}

pseudo_UEqn.solve();