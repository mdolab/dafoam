/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFunctionTotalPressure.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFunctionTotalPressure, 0);
addToRunTimeSelectionTable(DAFunction, DAFunctionTotalPressure, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunctionTotalPressure::DAFunctionTotalPressure(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const word functionName,
    const word functionPart,
    const dictionary& functionDict)
    : DAFunction(
        mesh,
        daOption,
        daModel,
        daIndex,
        functionName,
        functionPart,
        functionDict),
      daTurb_(const_cast<DATurbulenceModel&>(daModel.getDATurbulenceModel())),
      rho_(daTurb_.rho())
{
    // Assign type, this is common for all objectives
    functionDict_.readEntry<word>("type", functionType_);

    functionDict_.readEntry<scalar>("scale", scale_);
}

/// calculate the value of objective function
void DAFunctionTotalPressure::calcFunction(scalar& functionValue)
{
    /*
    Description:
        Calculate the total pressure TP=p+0.5*rho*U^2.

    Output:
    
        functionValue: the sum of objective, reduced across all processors and scaled by "scale"
    */

    // always calculate the area of all the patches
    areaSum_ = 0.0;
    forAll(faceSources_, idxI)
    {
        const label& functionFaceI = faceSources_[idxI];
        label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];
        areaSum_ += mesh_.magSf().boundaryField()[patchI][faceI];
    }
    reduce(areaSum_, sumOp<scalar>());

    // initialize objFunValue
    functionValue = 0.0;

    const objectRegistry& db = mesh_.thisDb();
    const volScalarField& p = db.lookupObject<volScalarField>("p");
    const volVectorField& U = db.lookupObject<volVectorField>("U");

    const volScalarField::Boundary& pBf = p.boundaryField();
    const volVectorField::Boundary& UBf = U.boundaryField();
    const volScalarField::Boundary& rhoBf = rho_.boundaryField();

    // calculate area weighted heat flux
    forAll(faceSources_, idxI)
    {
        const label& functionFaceI = faceSources_[idxI];
        label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];

        scalar area = mesh_.magSf().boundaryField()[patchI][faceI];
        scalar val =
            pBf[patchI][faceI] + 0.5 * rhoBf[patchI][faceI] * mag(UBf[patchI][faceI]) * mag(UBf[patchI][faceI]);
        val *= scale_ * area / areaSum_;

        functionValue += val;
    }

    // need to reduce the sum of force across all processors
    reduce(functionValue, sumOp<scalar>());

    // check if we need to calculate refDiff.
    this->calcRefVar(functionValue);

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
