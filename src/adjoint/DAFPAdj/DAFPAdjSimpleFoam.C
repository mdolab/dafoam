/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAFPAdjSimpleFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFPAdjSimpleFoam, 0);
addToRunTimeSelectionTable(DAFPAdj, DAFPAdjSimpleFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFPAdjSimpleFoam::DAFPAdjSimpleFoam(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const DAResidual& daResidual)
    : DAFPAdj(
        modelType,
        mesh,
        daOption,
        daModel,
        daIndex,
        daResidual)
{
}

label DAFPAdjSimpleFoam::run(
    Vec dFdW,
    Vec psi)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Solve the adjoint using the fixed-point iteration method
    
    dFdW:
        The dF/dW vector 

    psi:
        The adjoint solution vector
    */

    Info << "Solving the adjoint using fixed-point iteration method..." << endl;

    const PetscScalar* dFdWArray;
    VecGetArrayRead(dFdW, &dFdWArray);
    // we can read the value from the Petsc Vec dFdW, e.g., tmp = dFdWArray[0], between
    // the VecGetArray and VecRestoreArray calls
    VecRestoreArrayRead(dFdW, &dFdWArray);

    PetscScalar* psiArray;
    VecGetArray(psi, &psiArray);
    // we can assign the value to the Petsc Vec psi, e.g., psiArray[0] = 2.0, between
    // the VecGetArray and VecRestoreArray calls
    VecRestoreArray(psi, &psiArray);

#endif
    return 0;
}

void DAFPAdjSimpleFoam::invTranProd_UEqn(
    const List<vector>& mySource,
    volVectorField& pseudo_U)
{
    /*
    Description:
        Describe what this function does
    */

    const objectRegistry& db = mesh_.thisDb();
    const surfaceScalarField& phi = db.lookupObject<surfaceScalarField>("phi");
    const DATurbulenceModel& daTurb = daModel_.getDATurbulenceModel();
    volScalarField nuEff = daTurb.nuEff();

    // Get the pseudo_UEqn,
    // the most important thing here is to make sure the l.h.s. matches that of UEqn.
    fvVectorMatrix pseudo_UEqn(
        fvm::div(phi, pseudo_U)
        - fvm::laplacian(nuEff, pseudo_U)
        - fvc::div(nuEff * dev2(T(fvc::grad(pseudo_U)))));
    pseudo_UEqn.relax();

    // Swap upper() and lower()
    List<scalar> temp = pseudo_UEqn.upper();
    pseudo_UEqn.upper() = pseudo_UEqn.lower();
    pseudo_UEqn.lower() = temp;

    // Overwrite the r.h.s.
    pseudo_UEqn.source() = mySource;

    // Make sure that boundary contribution to source is zero,
    // Alternatively, we can deduct source by boundary contribution, so that it would cancel out during solve.
    forAll(pseudo_U.boundaryField(), patchI)
    {
        const fvPatch& pp = pseudo_U.boundaryField()[patchI].patch();
        forAll(pp, faceI)
        {
            label cellI = pp.faceCells()[faceI];
            pseudo_UEqn.source()[cellI] -= pseudo_UEqn.boundaryCoeffs()[patchI][faceI];
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
}

void DAFPAdjSimpleFoam::invTranProd_pEqn(
    const List<scalar>& mySource,
    volScalarField& pseudo_p)
{
    /*
    Description:
        Describe what this function does
    */

    const objectRegistry& db = mesh_.thisDb();
    const volVectorField& U = db.lookupObject<volVectorField>("U");
    const surfaceScalarField& phi = db.lookupObject<surfaceScalarField>("phi");
    const DATurbulenceModel& daTurb = daModel_.getDATurbulenceModel();
    volScalarField nuEff = daTurb.nuEff();

    // Construct UEqn first
    fvVectorMatrix UEqn(
        fvm::div(phi, U)
        - fvm::laplacian(nuEff, U)
        - fvc::div(nuEff * dev2(T(fvc::grad(U)))));
    // Without this, pRes would be way off.
    UEqn.relax();

    // create a scalar field with 1/A, reverse of A() of U
    volScalarField rAU(1.0 / UEqn.A());

    // Get the pseudo_pEqn,
    // the most important thing here is to make sure the l.h.s. matches that of pEqn.
    fvScalarMatrix pseudo_pEqn(fvm::laplacian(rAU, pseudo_p));

    // Swap upper() and lower()
    List<scalar> temp = pseudo_pEqn.upper();
    pseudo_pEqn.upper() = pseudo_pEqn.lower();
    pseudo_pEqn.lower() = temp;

    // Overwrite the r.h.s.
    pseudo_pEqn.source() = mySource;

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
            pseudo_pEqn.source()[cellI] -= pseudo_pEqn.boundaryCoeffs()[patchI][faceI];
        }
    }

    // Before solve, force xEqn.psi() to be solved into all zero
    forAll(pseudo_p.primitiveFieldRef(), cellI)
    {
        pseudo_p.primitiveFieldRef()[cellI] = 0;
    }

    pseudo_pEqn.solve();
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
