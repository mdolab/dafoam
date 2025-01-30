/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFunctionMassFlowRate.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFunctionMassFlowRate, 0);
addToRunTimeSelectionTable(DAFunction, DAFunctionMassFlowRate, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunctionMassFlowRate::DAFunctionMassFlowRate(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const word functionName)
    : DAFunction(
        mesh,
        daOption,
        daModel,
        daIndex,
        functionName),
      daTurb_(const_cast<DATurbulenceModel&>(daModel.getDATurbulenceModel()))
{
}

/// calculate the value of objective function
scalar DAFunctionMassFlowRate::calcFunction()
{
    /*
    Description:
        Calculate mass flow rate M = integral( rho*U*dS )
    */

    // initialize objFunValue
    scalar functionValue = 0.0;

    const objectRegistry& db = mesh_.thisDb();
    const volVectorField& U = db.lookupObject<volVectorField>("U");

    const volVectorField::Boundary& UBf = U.boundaryField();
    volScalarField rho = daTurb_.rho();
    const volScalarField::Boundary& rhoBf = rho.boundaryField();

    forAll(faceSources_, idxI)
    {
        const label& functionFaceI = faceSources_[idxI];
        label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];

        vector US = UBf[patchI][faceI];
        vector Sf = mesh_.Sf().boundaryField()[patchI][faceI];
        scalar rhoS = rhoBf[patchI][faceI];
        scalar mfr = rhoS * (US & Sf) * scale_;

        functionValue += mfr;
    }

    // need to reduce the sum of force across all processors
    reduce(functionValue, sumOp<scalar>());

    // check if we need to calculate refDiff.
    this->calcRefVar(functionValue);

    return functionValue;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
