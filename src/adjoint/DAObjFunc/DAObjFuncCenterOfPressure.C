/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAObjFuncCenterOfPressure.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAObjFuncCenterOfPressure, 0);
addToRunTimeSelectionTable(DAObjFunc, DAObjFuncCenterOfPressure, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAObjFuncCenterOfPressure::DAObjFuncCenterOfPressure(
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

    // for computing center of pressure, first read in some parameters from objFuncDict_
    // these parameters are only for moment center of pressure objective

    // Assign type, this is common for all objectives
    objFuncDict_.readEntry<word>("type", objFuncType_);

    scalarList dir;
    objFuncDict_.readEntry<scalarList>("axis", dir);
    pressureDir_[0] = dir[0];
    pressureDir_[1] = dir[1];
    pressureDir_[2] = dir[2];

    if (fabs(mag(pressureDir_) - 1.0) > 1.0e-4)
    {
        FatalErrorIn(" ") << "the magnitude of the axis parameter in "
                          << objFuncName << " " << objFuncPart << " is not 1.0!"
                          << abort(FatalError);
    }

    scalarList center;
    objFuncDict_.readEntry<scalarList>("center", center);
    pressureOrigin_[0] = center[0];
    pressureOrigin_[1] = center[1];
    pressureOrigin_[2] = center[2];

    objFuncDict_.readEntry<scalar>("scale", scale_);
}

/// calculate the value of objective function
void DAObjFuncCenterOfPressure::calcObjFunc(
    const labelList& objFuncFaceSources,
    const labelList& objFuncCellSources,
    scalarList& objFuncFaceValues,
    scalarList& objFuncCellValues,
    scalar& objFuncValue)
{
    /*
    Description:
        Calculate the average location of pressure applied to the body.

    Input:
        objFuncFaceSources: List of face source (index) for this objective

    Output:
        objFuncValue: the sum of objective along a chosen "axis", reduced across all processsors and scaled by "scale"
    */

    // initialize objFuncValue
    objFuncValue = 0.0;
    vector weightedPressure(0, 0, 0);
    scalar totalPressure = 0.0;

    const objectRegistry& db = mesh_.thisDb();
    const volScalarField& p = db.lookupObject<volScalarField>("p");

    const surfaceVectorField::Boundary& Sfb = mesh_.Sf().boundaryField();

    // calculate discrete force for each objFuncFace
    forAll(objFuncFaceSources, idxI)
    {
        const label& objFuncFaceI = objFuncFaceSources[idxI];
        label bFaceI = objFuncFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];

        // normal force
        vector fN(Sfb[patchI][faceI] * p.boundaryField()[patchI][faceI]);
        // r vector
        vector rVec = mesh_.Cf().boundaryField()[patchI][faceI];
        // force weighted by distance
        weightedPressure += rVec * mag(fN);
        // total force
        totalPressure += mag(fN);
    }

    // need to reduce the sum of force across all processors
    reduce(weightedPressure, sumOp<vector>());
    reduce(totalPressure, sumOp<scalar>());

    objFuncValue = scale_ * ((weightedPressure / totalPressure) - pressureOrigin_) & pressureDir_;

    return;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
