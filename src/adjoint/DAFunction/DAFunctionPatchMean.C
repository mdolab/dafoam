/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFunctionPatchMean.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFunctionPatchMean, 0);
addToRunTimeSelectionTable(DAFunction, DAFunctionPatchMean, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunctionPatchMean::DAFunctionPatchMean(
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
        functionName)
{
    functionDict_.readEntry<word>("varName", varName_);

    functionDict_.readEntry<word>("varType", varType_);

    functionDict_.readEntry<label>("index", index_);
}

/// calculate the value of objective function
scalar DAFunctionPatchMean::calcFunction()
{
    /*
    Description:
        Calculate the patch mean
    */

    // calculate the area of all the patches. We need to recompute because the surface area
    // may change during the optimization
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
    scalar functionValue = 0.0;

    const objectRegistry& db = mesh_.thisDb();

    if (varType_ == "scalar")
    {
        const volScalarField& var = db.lookupObject<volScalarField>(varName_);

        // calculate area weighted heat flux
        forAll(faceSources_, idxI)
        {
            const label& functionFaceI = faceSources_[idxI];
            label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
            const label patchI = daIndex_.bFacePatchI[bFaceI];
            const label faceI = daIndex_.bFaceFaceI[bFaceI];
            scalar area = mesh_.magSf().boundaryField()[patchI][faceI];
            functionValue += scale_ * area * var.boundaryField()[patchI][faceI] / areaSum_;
        }
    }
    else if (varType_ == "vector")
    {
        const volVectorField& var = db.lookupObject<volVectorField>(varName_);

        // calculate area weighted heat flux
        forAll(faceSources_, idxI)
        {
            const label& functionFaceI = faceSources_[idxI];
            label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
            const label patchI = daIndex_.bFacePatchI[bFaceI];
            const label faceI = daIndex_.bFaceFaceI[bFaceI];
            scalar area = mesh_.magSf().boundaryField()[patchI][faceI];
            functionValue += scale_ * area * var.boundaryField()[patchI][faceI][index_] / areaSum_;
        }
    }
    else
    {
        FatalErrorIn("DAFunctionPatchMean::calcFunction")
            << "varType not valid. Options are scalar or vector"
            << abort(FatalError);
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
