/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

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

    scalarList axisList;
    objFuncDict_.readEntry<scalarList>("axis", axisList);
    axis_[0] = axisList[0];
    axis_[1] = axisList[1];
    axis_[2] = axisList[2];
    // normalize
    axis_ /= mag(axis_);

    scalarList forceAxisList;
    objFuncDict_.readEntry<scalarList>("forceAxis", forceAxisList);
    forceAxis_[0] = forceAxisList[0];
    forceAxis_[1] = forceAxisList[1];
    forceAxis_[2] = forceAxisList[2];
    // normalize
    forceAxis_ /= mag(forceAxis_);

    if (fabs(axis_ & forceAxis_) > 1e-8)
    {
        FatalErrorIn(" ") << "axis and forceAxis vectors need to be orthogonal! "
                          << abort(FatalError);
    }

    scalarList centerList;
    objFuncDict_.readEntry<scalarList>("center", centerList);
    center_[0] = centerList[0];
    center_[1] = centerList[1];
    center_[2] = centerList[2];

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
        objFuncValue: the sum of objective along a chosen "axis", reduced across all processors and scaled by "scale"
    */

    // initialize objFuncValue
    objFuncValue = 0.0;
    scalar weightedPressure = 0.0;
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
        // r vector projected to the axis vector
        scalar r = (mesh_.Cf().boundaryField()[patchI][faceI] - center_) & axis_;
        // force projected to force axis vector
        scalar f = fN & forceAxis_;
        // force weighted by distance
        weightedPressure += r * f;
        // total force
        totalPressure += f;
    }

    // need to reduce the sum of force across all processors
    reduce(weightedPressure, sumOp<scalar>());
    reduce(totalPressure, sumOp<scalar>());

    objFuncValue = scale_ * (weightedPressure / totalPressure + (center_ & axis_));

    return;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
