// Overwrite the r.h.s.
pseudoPEqn.source() = pSource.primitiveField();

// pEqn.setReference(pRefCell, pRefValue);
// Here, pRefCell is a label, and pRefValue is a scalar
pseudoPEqn.setReference(0, 0.0);

// Make sure that boundary contribution to source is zero,
// Alternatively, we can deduct source by boundary contribution, so that it would cancel out during solve.
forAll(pseudoP.boundaryField(), patchI)
{
    const fvPatch& pp = pseudoP.boundaryField()[patchI].patch();
    forAll(pp, faceI)
    {
        label cellI = pp.faceCells()[faceI];
        pseudoPEqn.source()[cellI] -= pseudoPEqn.boundaryCoeffs()[patchI][faceI];
        //pseudoPEqn.boundaryCoeffs()[patchI][faceI] = 0;
    }
}

// Before solve, force xEqn.psi() to be solved into all zero
forAll(pseudoP.primitiveFieldRef(), cellI)
{
    pseudoP.primitiveFieldRef()[cellI] = 0;
}

pseudoPEqn.solve(solverDictP_);
