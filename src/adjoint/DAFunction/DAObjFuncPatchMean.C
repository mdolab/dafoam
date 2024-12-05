/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAObjFuncPatchMean.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAObjFuncPatchMean, 0);
addToRunTimeSelectionTable(DAObjFunc, DAObjFuncPatchMean, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAObjFuncPatchMean::DAObjFuncPatchMean(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const DAResidual& daResidual,
    const word objFuncName,
    const word objFuncPart,
    const dictionary& objFuncDict)
    : DAObjFunc(
        mesh,
        daOption,
        daModel,
        daIndex,
        daResidual,
        objFuncName,
        objFuncPart,
        objFuncDict)
{
    // Assign type, this is common for all objectives
    objFuncDict_.readEntry<word>("type", objFuncType_);

    objFuncDict_.readEntry<scalar>("scale", scale_);

    objFuncDict_.readEntry<word>("varName", varName_);

    objFuncDict_.readEntry<word>("varType", varType_);

    objFuncDict_.readEntry<label>("component", component_);
}

/// calculate the value of objective function
void DAObjFuncPatchMean::calcObjFunc(
    const labelList& objFuncFaceSources,
    const labelList& objFuncCellSources,
    scalarList& objFuncFaceValues,
    scalarList& objFuncCellValues,
    scalar& objFuncValue)
{
    /*
    Description:
        Calculate the patch mean

    Input:
        objFuncFaceSources: List of face source (index) for this objective
    
        objFuncCellSources: List of cell source (index) for this objective

    Output:
        objFuncFaceValues: the discrete value of objective for each face source (index). 
        This  will be used for computing df/dw in the adjoint.
    
        objFuncCellValues: the discrete value of objective on each cell source (index). 
        This will be used for computing df/dw in the adjoint.
    
        objFuncValue: the sum of objective, reduced across all processors and scaled by "scale"
    */

    // calculate the area of all the patches. We need to recompute because the surface area 
    // may change during the optimization
    areaSum_ = 0.0;
    forAll(objFuncFaceSources, idxI)
    {
        const label& objFuncFaceI = objFuncFaceSources[idxI];
        label bFaceI = objFuncFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];
        areaSum_ += mesh_.magSf().boundaryField()[patchI][faceI];
    }
    reduce(areaSum_, sumOp<scalar>());

    // initialize objFunValue
    objFuncValue = 0.0;

    const objectRegistry& db = mesh_.thisDb();

    if (varType_ == "scalar")
    {
        const volScalarField& var = db.lookupObject<volScalarField>(varName_);

        // calculate area weighted heat flux
        forAll(objFuncFaceSources, idxI)
        {
            const label& objFuncFaceI = objFuncFaceSources[idxI];
            label bFaceI = objFuncFaceI - daIndex_.nLocalInternalFaces;
            const label patchI = daIndex_.bFacePatchI[bFaceI];
            const label faceI = daIndex_.bFaceFaceI[bFaceI];
            scalar area = mesh_.magSf().boundaryField()[patchI][faceI];
            objFuncValue += scale_ * area * var.boundaryField()[patchI][faceI] / areaSum_;
        }
    }
    else if (varType_ == "vector")
    {
        const volVectorField& var = db.lookupObject<volVectorField>(varName_);

        // calculate area weighted heat flux
        forAll(objFuncFaceSources, idxI)
        {
            const label& objFuncFaceI = objFuncFaceSources[idxI];
            label bFaceI = objFuncFaceI - daIndex_.nLocalInternalFaces;
            const label patchI = daIndex_.bFacePatchI[bFaceI];
            const label faceI = daIndex_.bFaceFaceI[bFaceI];
            scalar area = mesh_.magSf().boundaryField()[patchI][faceI];
            objFuncValue += scale_ * area * var.boundaryField()[patchI][faceI][component_] / areaSum_;
        }
    }
    else
    {
        FatalErrorIn("DAObjFuncPatchMean::calcObjFunc")
            << "varType not valid. Options are scalar or vector"
            << abort(FatalError);
    }

    // need to reduce the sum of force across all processors
    reduce(objFuncValue, sumOp<scalar>());

    // check if we need to calculate refDiff.
    this->calcRefVar(objFuncValue);

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
