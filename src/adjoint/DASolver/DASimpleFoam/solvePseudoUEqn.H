// Overwrite the r.h.s.
pseudoUEqn.source() = USource.primitiveField();

// Make sure that boundary contribution to source is zero,
// Alternatively, we can deduct source by boundary contribution, so that it would cancel out during solve.
forAll(pseudoU.boundaryField(), patchI)
{
    const fvPatch& pp = pseudoU.boundaryField()[patchI].patch();
    forAll(pp, faceI)
    {
        label cellI = pp.faceCells()[faceI];
        //myDiag[cellI] += TEqn.internalCoeffs()[patchI][faceI];
        pseudoUEqn.source()[cellI] -= pseudoUEqn.boundaryCoeffs()[patchI][faceI];
        //pseudoUEqn.boundaryCoeffs()[patchI][faceI][0] = 0;
        //pseudoUEqn.boundaryCoeffs()[patchI][faceI][1] = 0;
        //pseudoUEqn.boundaryCoeffs()[patchI][faceI][2] = 0;
    }
}

// Before solve, force xEqn.psi() to be solved into all zero
forAll(pseudoU.primitiveFieldRef(), cellI)
{
    pseudoU.primitiveFieldRef()[cellI][0] = 0;
    pseudoU.primitiveFieldRef()[cellI][1] = 0;
    pseudoU.primitiveFieldRef()[cellI][2] = 0;
}

pseudoUEqn.solve(solverDictU_);
