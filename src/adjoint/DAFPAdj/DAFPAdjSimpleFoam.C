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
    const volVectorField& U = db.lookupObject<volVectorField>("U");
    const volScalarField& p = db.lookupObject<volScalarField>("p");
    const surfaceScalarField& phi = db.lookupObject<surfaceScalarField>("phi");
    const DATurbulenceModel& daTurb = daModel_.getDATurbulenceModel();
    volScalarField nuEff = daTurb.nuEff();

    // Get the pseudo_UEqn,
    // the most important thing here is to make sure the l.h.s. matches that of UEqn.
    //pseudo_U = U;
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

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
