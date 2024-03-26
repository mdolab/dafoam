/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAObjFuncLocation.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAObjFuncLocation, 0);
addToRunTimeSelectionTable(DAObjFunc, DAObjFuncLocation, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAObjFuncLocation::DAObjFuncLocation(
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

    // setup the connectivity for Location.
    objFuncConInfo_ = {};

    objFuncDict_.readEntry<scalar>("scale", scale_);

    objFuncDict_.readEntry<word>("mode", mode_);

    if (objFuncDict_.found("coeffKS"))
    {
        objFuncDict_.readEntry<scalar>("coeffKS", coeffKS_);
    }

    if (objFuncDict_.found("axis"))
    {
        scalarList axisRead;
        objFuncDict_.readEntry<scalarList>("axis", axisRead);
        axis_ = {axisRead[0], axisRead[1], axisRead[2]};
        axis_ /= mag(axis_);
    }

    if (objFuncDict_.found("center"))
    {
        scalarList centerRead;
        objFuncDict_.readEntry<scalarList>("center", centerRead);
        center_ = {centerRead[0], centerRead[1], centerRead[2]};
    }

    if (mode_ == "maxRadius")
    {
        // we need to identify the patchI and faceI that has maxR
        // the proc that does not own the maxR face will have negative patchI and faceI
        // we assume the patchI and faceI do not change during the optimization
        // otherwise, we should use maxRadiusKS instead
        scalar maxR = -100000;
        label maxRPatchI, maxRFaceI;
        forAll(objFuncFaceSources_, idxI)
        {
            const label& objFuncFaceI = objFuncFaceSources_[idxI];
            label bFaceI = objFuncFaceI - daIndex_.nLocalInternalFaces;
            const label patchI = daIndex_.bFacePatchI[bFaceI];
            const label faceI = daIndex_.bFaceFaceI[bFaceI];

            vector faceC = mesh_.Cf().boundaryField()[patchI][faceI] - center_;

            tensor faceCTensor(tensor::zero);
            faceCTensor.xx() = faceC.x();
            faceCTensor.yy() = faceC.y();
            faceCTensor.zz() = faceC.z();

            vector faceCAxial = faceCTensor & axis_;
            vector faceCRadial = faceC - faceCAxial;

            scalar radius = mag(faceCRadial);

            if (radius > maxR)
            {
                maxR = radius;
                maxRPatchI = patchI;
                maxRFaceI = faceI;
            }
        }

        scalar maxRGlobal = maxR;
        reduce(maxRGlobal, maxOp<scalar>());

        if (fabs(maxRGlobal - maxR) < 1e-8)
        {
            maxRPatchI_ = maxRPatchI;
            maxRFaceI_ = maxRFaceI;
        }
    }

}

/// calculate the value of objective function
void DAObjFuncLocation::calcObjFunc(
    const labelList& objFuncFaceSources,
    const labelList& objFuncCellSources,
    scalarList& objFuncFaceValues,
    scalarList& objFuncCellValues,
    scalar& objFuncValue)
{
    /*
    Description:
        Calculate the Location of a selected patch. 

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

    // initialize faceValues to zero
    forAll(objFuncFaceValues, idxI)
    {
        objFuncFaceValues[idxI] = 0.0;
    }
    // initialize objFunValue
    objFuncValue = 0.0;

    if (mode_ == "maxRadiusKS")
    {
        // calculate Location
        scalar objValTmp = 0.0;

        forAll(objFuncFaceSources, idxI)
        {
            const label& objFuncFaceI = objFuncFaceSources[idxI];
            label bFaceI = objFuncFaceI - daIndex_.nLocalInternalFaces;
            const label patchI = daIndex_.bFacePatchI[bFaceI];
            const label faceI = daIndex_.bFaceFaceI[bFaceI];

            vector faceC = mesh_.Cf().boundaryField()[patchI][faceI] - center_;

            tensor faceCTensor(tensor::zero);
            faceCTensor.xx() = faceC.x();
            faceCTensor.yy() = faceC.y();
            faceCTensor.zz() = faceC.z();

            vector faceCAxial = faceCTensor & axis_;
            vector faceCRadial = faceC - faceCAxial;

            scalar radius = mag(faceCRadial);

            objFuncFaceValues[idxI] = exp(coeffKS_ * radius);

            objValTmp += objFuncFaceValues[idxI];

            if (objValTmp > 1e200)
            {
                FatalErrorIn(" ") << "KS function summation term too large! "
                                  << "Reduce coeffKS! " << abort(FatalError);
            }
        }

        // need to reduce the sum of force across all processors
        reduce(objValTmp, sumOp<scalar>());

        // expSumKS stores sum[exp(coeffKS*x_i)], it will be used to scale dFdW
        expSumKS = objValTmp;

        objFuncValue = log(objValTmp) / coeffKS_;
    }
    else if (mode_ == "maxRadius")
    {
        scalar radius = 0.0;
        if (maxRPatchI_ >= 0 && maxRFaceI_ >= 0)
        {
            vector faceC = mesh_.Cf().boundaryField()[maxRPatchI_][maxRFaceI_] - center_;

            tensor faceCTensor(tensor::zero);
            faceCTensor.xx() = faceC.x();
            faceCTensor.yy() = faceC.y();
            faceCTensor.zz() = faceC.z();

            vector faceCAxial = faceCTensor & axis_;
            vector faceCRadial = faceC - faceCAxial;

            radius = mag(faceCRadial);
        }

        reduce(radius, sumOp<scalar>());

        objFuncValue = radius;
    }
    else
    {
        FatalErrorIn("DAObjFuncLocation") << "mode: " << mode_ << " not supported!"
                                          << "Options are: maxRadius"
                                          << abort(FatalError);
    }

    // check if we need to calculate refDiff.
    this->calcRefDiff(objFuncValue);

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
