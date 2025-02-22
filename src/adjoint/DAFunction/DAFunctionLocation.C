/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFunctionLocation.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFunctionLocation, 0);
addToRunTimeSelectionTable(DAFunction, DAFunctionLocation, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunctionLocation::DAFunctionLocation(
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

    functionDict_.readEntry<word>("mode", mode_);

    if (functionDict_.found("coeffKS"))
    {
        functionDict_.readEntry<scalar>("coeffKS", coeffKS_);
    }

    if (functionDict_.found("axis"))
    {
        scalarList axisRead;
        functionDict_.readEntry<scalarList>("axis", axisRead);
        axis_ = {axisRead[0], axisRead[1], axisRead[2]};
        axis_ /= mag(axis_);
    }

    if (functionDict_.found("center"))
    {
        scalarList centerRead;
        functionDict_.readEntry<scalarList>("center", centerRead);
        center_ = {centerRead[0], centerRead[1], centerRead[2]};
    }

    snapCenter2Cell_ = functionDict_.lookupOrDefault<label>("snapCenter2Cell", 0);
    if (snapCenter2Cell_)
    {
        point centerPoint = {center_[0], center_[1], center_[2]};

        // NOTE: we need to call a self-defined findCell func to make it work correctly in ADR
        snappedCenterCellI_ = DAUtility::myFindCell(mesh_, centerPoint);

        label foundCellI = 0;
        if (snappedCenterCellI_ >= 0)
        {
            foundCellI = 1;
        }
        reduce(foundCellI, sumOp<label>());
        if (foundCellI != 1)
        {
            FatalErrorIn(" ") << "There should be only one cell found globally while "
                              << foundCellI << " was returned."
                              << " Please adjust center such that it is located completely"
                              << " within a cell in the mesh domain. The center should not "
                              << " be outside of the mesh domain or on a mesh face "
                              << abort(FatalError);
        }
        vector snappedCenter = vector::zero;
        this->findGlobalSnappedCenter(snappedCenterCellI_, snappedCenter);
        Info << "snap to center " << snappedCenter << endl;
    }

    if (mode_ == "maxRadius")
    {
        // we need to identify the patchI and faceI that has maxR
        // the proc that does not own the maxR face will have negative patchI and faceI
        // we assume the patchI and faceI do not change during the optimization
        // otherwise, we should use maxRadiusKS instead
        scalar maxR = -100000;
        label maxRPatchI = -999, maxRFaceI = -999;
        forAll(faceSources_, idxI)
        {
            const label& functionFaceI = faceSources_[idxI];
            label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
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

void DAFunctionLocation::findGlobalSnappedCenter(
    label snappedCenterCellI,
    vector& center)
{
    scalar centerX = 0.0;
    scalar centerY = 0.0;
    scalar centerZ = 0.0;

    if (snappedCenterCellI >= 0)
    {
        centerX = mesh_.C()[snappedCenterCellI][0];
        centerY = mesh_.C()[snappedCenterCellI][1];
        centerZ = mesh_.C()[snappedCenterCellI][2];
    }
    reduce(centerX, sumOp<scalar>());
    reduce(centerY, sumOp<scalar>());
    reduce(centerZ, sumOp<scalar>());

    center[0] = centerX;
    center[1] = centerY;
    center[2] = centerZ;
}

/// calculate the value of objective function
scalar DAFunctionLocation::calcFunction()
{
    /*
    Description:
        Calculate the Location of a selected patch. The actual computation depends on the mode
    */

    // initialize objFunValue
    scalar functionValue = 0.0;

    if (mode_ == "maxRadiusKS")
    {
        // calculate Location
        scalar objValTmp = 0.0;

        vector center = center_;
        if (snapCenter2Cell_)
        {
            this->findGlobalSnappedCenter(snappedCenterCellI_, center);
        }

        forAll(faceSources_, idxI)
        {
            const label& functionFaceI = faceSources_[idxI];
            label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
            const label patchI = daIndex_.bFacePatchI[bFaceI];
            const label faceI = daIndex_.bFaceFaceI[bFaceI];

            vector faceC = mesh_.Cf().boundaryField()[patchI][faceI] - center;

            tensor faceCTensor(tensor::zero);
            faceCTensor.xx() = faceC.x();
            faceCTensor.yy() = faceC.y();
            faceCTensor.zz() = faceC.z();

            vector faceCAxial = faceCTensor & axis_;
            vector faceCRadial = faceC - faceCAxial;

            scalar radius = mag(faceCRadial);

            objValTmp += exp(coeffKS_ * radius);

            if (objValTmp > 1e200)
            {
                FatalErrorIn(" ") << "KS function summation term too large! "
                                  << "Reduce coeffKS! " << abort(FatalError);
            }
        }

        // need to reduce the sum of force across all processors
        reduce(objValTmp, sumOp<scalar>());

        functionValue = log(objValTmp) / coeffKS_;
    }
    else if (mode_ == "maxInverseRadiusKS")
    {
        // this is essentially minimal radius using KS

        // calculate Location
        scalar objValTmp = 0.0;

        vector center = center_;
        if (snapCenter2Cell_)
        {
            this->findGlobalSnappedCenter(snappedCenterCellI_, center);
        }

        forAll(faceSources_, idxI)
        {
            const label& functionFaceI = faceSources_[idxI];
            label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
            const label patchI = daIndex_.bFacePatchI[bFaceI];
            const label faceI = daIndex_.bFaceFaceI[bFaceI];

            vector faceC = mesh_.Cf().boundaryField()[patchI][faceI] - center;

            tensor faceCTensor(tensor::zero);
            faceCTensor.xx() = faceC.x();
            faceCTensor.yy() = faceC.y();
            faceCTensor.zz() = faceC.z();

            vector faceCAxial = faceCTensor & axis_;
            vector faceCRadial = faceC - faceCAxial;

            scalar radius = mag(faceCRadial);
            scalar iRadius = 1.0 / (radius + 1e-12);

            objValTmp += exp(coeffKS_ * iRadius);

            if (objValTmp > 1e200)
            {
                FatalErrorIn(" ") << "KS function summation term too large! "
                                  << "Reduce coeffKS! " << abort(FatalError);
            }
        }

        // need to reduce the sum of force across all processors
        reduce(objValTmp, sumOp<scalar>());

        functionValue = log(objValTmp) / coeffKS_;
    }
    else if (mode_ == "maxRadius")
    {
        scalar radius = 0.0;

        vector center = center_;
        if (snapCenter2Cell_)
        {
            this->findGlobalSnappedCenter(snappedCenterCellI_, center);
        }

        if (maxRPatchI_ >= 0 && maxRFaceI_ >= 0)
        {

            vector faceC = mesh_.Cf().boundaryField()[maxRPatchI_][maxRFaceI_] - center;

            tensor faceCTensor(tensor::zero);
            faceCTensor.xx() = faceC.x();
            faceCTensor.yy() = faceC.y();
            faceCTensor.zz() = faceC.z();

            vector faceCAxial = faceCTensor & axis_;
            vector faceCRadial = faceC - faceCAxial;

            radius = mag(faceCRadial);
        }

        reduce(radius, sumOp<scalar>());

        functionValue = radius;
    }
    else
    {
        FatalErrorIn("DAFunctionLocation") << "mode: " << mode_ << " not supported!"
                                           << "Options are: maxRadius"
                                           << abort(FatalError);
    }

    // check if we need to calculate refDiff.
    this->calcRefVar(functionValue);

    return functionValue;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
