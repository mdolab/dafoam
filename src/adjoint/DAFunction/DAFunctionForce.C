/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFunctionForce.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFunctionForce, 0);
addToRunTimeSelectionTable(DAFunction, DAFunctionForce, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunctionForce::DAFunctionForce(
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
      daTurb_(daModel.getDATurbulenceModel())
{

    // for computing force, first read in some parameters from functionDict_
    // these parameters are only for force objective

    // Assign type, this is common for all objectives
    functionDict_.readEntry<word>("type", functionType_);

    // we support three direction modes
    dirMode_ = functionDict_.getWord("directionMode");
    if (dirMode_ == "fixedDirection")
    {
        scalarList dir;
        functionDict_.readEntry<scalarList>("direction", dir);
        forceDir_[0] = dir[0];
        forceDir_[1] = dir[1];
        forceDir_[2] = dir[2];
    }
    else if (dirMode_ == "parallelToFlow" || dirMode_ == "normalToFlow")
    {
        // initial value for forceDir_. it will be dynamically adjusted later
        forceDir_ = {1.0, 0.0, 0.0};
        word alphaName = functionDict_.getWord("alphaName");
        dictionary alphaSubDict = daOption_.getAllOptions().subDict("designVar").subDict(alphaName);
        wordList patches;
        alphaSubDict.readEntry<wordList>("patches", patches);
        inoutRefPatchName_ = patches[0];
        HashTable<label> axisIndices;
        axisIndices.set("x", 0);
        axisIndices.set("y", 1);
        axisIndices.set("z", 2);
        word flowAxis = alphaSubDict.getWord("flowAxis");
        word normalAxis = alphaSubDict.getWord("normalAxis");
        flowAxisIndex_ = axisIndices[flowAxis];
        normalAxisIndex_ = axisIndices[normalAxis];
    }
    else
    {
        FatalErrorIn(" ") << "directionMode for "
                          << functionName << " " << functionPart << " not valid!"
                          << "Options: fixedDirection, parallelToFlow, normalToFlow."
                          << abort(FatalError);
    }

    if (fabs(mag(forceDir_) - 1.0) > 1.0e-8)
    {
        FatalErrorIn(" ") << "the magnitude of the direction parameter in "
                          << functionName << " " << functionPart << " is not 1.0!"
                          << abort(FatalError);
    }

    functionDict_.readEntry<scalar>("scale", scale_);
}

/// calculate the value of objective function
void DAFunctionForce::calcFunction(scalar& functionValue)
{
    /*
    Description:
        Calculate the force which consist of pressure and viscous components.
        This code is modified based on:
        src/functionObjects/forcces/forces.C

    Output:
        functionValue: the sum of objective, reduced across all processsors and scaled by "scale"
    */

    if (dirMode_ != "fixedDirection")
    {
        this->updateForceDir(forceDir_);
    }

    // reload the scale, which may be needed for multipoint optimization
    functionDict_.readEntry<scalar>("scale", scale_);

    // initialize objFunValue
    functionValue = 0.0;

    const objectRegistry& db = mesh_.thisDb();
    const volScalarField& p = db.lookupObject<volScalarField>("p");

    const surfaceVectorField::Boundary& Sfb = mesh_.Sf().boundaryField();

    tmp<volSymmTensorField> tdevRhoReff = daTurb_.devRhoReff();
    const volSymmTensorField::Boundary& devRhoReffb = tdevRhoReff().boundaryField();

    // calculate discrete force for each functionFace
    forAll(faceSources_, idxI)
    {
        const label& functionFaceI = faceSources_[idxI];
        label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];

        // normal force
        vector fN(Sfb[patchI][faceI] * p.boundaryField()[patchI][faceI]);
        // tangential force
        vector fT(Sfb[patchI][faceI] & devRhoReffb[patchI][faceI]);
        // project the force to forceDir
        scalar faceValue = scale_ * ((fN + fT) & forceDir_);

        functionValue += faceValue;
    }

    // need to reduce the sum of force across all processors
    reduce(functionValue, sumOp<scalar>());

    // check if we need to calculate refDiff.
    this->calcRefVar(functionValue);

    return;
}

void DAFunctionForce::updateForceDir(vector& forceDir)
{
    /*
    Description:
        Dynamically adjust the force direction based on the flow direction from
        far field
        NOTE: we have special implementation for forward mode AD because
        we need to progagate the aoa seed to here. The problem of the regular
        treatment is that we use reduce(flowDir[0], maxOp<scalar>()); to get the
        flowDir for thoese processors that have no inout patches, this reduce
        function will lose track of AD seeds for aoa.

    Output:
        forceDir: the force direction vector
    */

    if (DAUtility::angleOfAttackRadForwardAD > -999.0)
    {
        // DAUtility::angleOfAttackRadForwardAD is set, this is dF/dAOA forwardMode
        // derivative, use special treatment here

        scalar compA = cos(DAUtility::angleOfAttackRadForwardAD);
        scalar compB = sin(DAUtility::angleOfAttackRadForwardAD);

        if (dirMode_ == "parallelToFlow")
        {
            forceDir[flowAxisIndex_] = compA;
            forceDir[normalAxisIndex_] = compB;
        }
        else if (dirMode_ == "normalToFlow")
        {
            forceDir[flowAxisIndex_] = -compB;
            forceDir[normalAxisIndex_] = compA;
        }
        else
        {
            FatalErrorIn(" ") << "directionMode not valid!"
                              << "Options: parallelToFlow, normalToFlow."
                              << abort(FatalError);
        }
    }
    else
    {
        // DAUtility::angleOfAttackRadForwardAD is not set, usual regular forceDir computation

        label patchI = mesh_.boundaryMesh().findPatchID(inoutRefPatchName_);

        volVectorField& U =
            const_cast<volVectorField&>(mesh_.thisDb().lookupObject<volVectorField>("U"));

        vector flowDir = {-1e16, -1e16, -1e16};

        // for decomposed domain, don't set BC if the patch is empty
        if (mesh_.boundaryMesh()[patchI].size() > 0)
        {
            if (U.boundaryField()[patchI].type() == "fixedValue")
            {
                flowDir = U.boundaryField()[patchI][0];
                flowDir = flowDir / mag(flowDir);
            }
            else if (U.boundaryField()[patchI].type() == "inletOutlet")
            {
                // perturb inletValue
                mixedFvPatchField<vector>& inletOutletPatch =
                    refCast<mixedFvPatchField<vector>>(U.boundaryFieldRef()[patchI]);
                flowDir = inletOutletPatch.refValue()[0];
                flowDir = flowDir / mag(flowDir);
            }
            else
            {
                FatalErrorIn("") << "boundaryType: " << U.boundaryField()[patchI].type()
                                 << " not supported!"
                                 << "Available options are: fixedValue, inletOutlet"
                                 << abort(FatalError);
            }
        }

        // need to reduce the sum of force across all processors, this is because some of
        // the processor might not own the inoutRefPatchName_ so their flowDir will be -1e16, but
        // when calling the following reduce function, they will get the correct flowDir
        // computed by other processors
        reduce(flowDir[0], maxOp<scalar>());
        reduce(flowDir[1], maxOp<scalar>());
        reduce(flowDir[2], maxOp<scalar>());

        if (dirMode_ == "parallelToFlow")
        {
            forceDir = flowDir;
        }
        else if (dirMode_ == "normalToFlow")
        {
            forceDir[flowAxisIndex_] = -flowDir[normalAxisIndex_];
            forceDir[normalAxisIndex_] = flowDir[flowAxisIndex_];
        }
        else
        {
            FatalErrorIn(" ") << "directionMode not valid!"
                              << "Options: parallelToFlow, normalToFlow."
                              << abort(FatalError);
        }
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
