/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

\*---------------------------------------------------------------------------*/

#include "DAFunctionFieldMax.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFunctionFieldMax, 0);
addToRunTimeSelectionTable(DAFunction, DAFunctionFieldMax, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunctionFieldMax::DAFunctionFieldMax(
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
    functionDict_.readEntry<word>("fieldName", fieldName_);

    functionDict_.readEntry<word>("fieldType", fieldType_);

    if (fieldType_ == "vector")
    {
        functionDict_.readEntry<labelList>("indices", indices_);
    }
}

/// calculate the value of objective function
scalar DAFunctionFieldMax::calcFunction()
{
    /*
    Description:
        Calculate the field max
    */

    if (cellSources_.size() != 0)
    {
        FatalErrorIn("")
            << "only support patchToFace source!"
            << abort(FatalError);
    }

    scalar maxVal = -1e16;
    forAll(faceSources_, idxI)
    {
        const label& functionFaceI = faceSources_[idxI];
        label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];
        if (fieldType_ == "scalar")
        {
            const volScalarField& field = mesh_.thisDb().lookupObject<volScalarField>(fieldName_);
            const scalar& val = field.boundaryField()[patchI][faceI];
            if (val > maxVal)
            {
                maxVal = val;
            }
        }
        else if (fieldType_ == "vector")
        {
            const volVectorField& field = mesh_.thisDb().lookupObject<volVectorField>(fieldName_);
            forAll(indices_, idxJ)
            {
                label compI = indices_[idxJ];
                const scalar& val = field.boundaryField()[patchI][faceI][compI];
                if (val > maxVal)
                {
                    maxVal = val;
                }
            }
        }
        else
        {
            FatalErrorIn("")
                << "varType not valid. Options are scalar or vector"
                << abort(FatalError);
        }
    }
    reduce(maxVal, maxOp<scalar>());

    maxVal = maxVal * scale_;

    return maxVal;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
