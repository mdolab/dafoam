/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAObjFuncForce.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAObjFuncForce, 0);
addToRunTimeSelectionTable(DAObjFunc, DAObjFuncForce, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAObjFuncForce::DAObjFuncForce(
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
        objFuncDict),
      daTurb_(daModel.getDATurbulenceModel())
{

    // for computing force, first read in some parameters from objFuncDict_
    // these parameters are only for force objective

    // Assign type, this is common for all objectives
    objFuncDict_.readEntry<word>("type", objFuncType_);

    // we support three direction modes
    dirMode_ = objFuncDict_.getWord("directionMode");
    if (dirMode_ == "fixedDirection")
    {
        scalarList dir;
        objFuncDict_.readEntry<scalarList>("direction", dir);
        forceDir_[0] = dir[0];
        forceDir_[1] = dir[1];
        forceDir_[2] = dir[2];
    }
    else if (dirMode_ == "parallelToFlow" || dirMode_ == "normalToFlow")
    {
        // initial value for forceDir_. it will be dynamically adjusted later
        forceDir_ = {1.0, 0.0, 0.0};
        word alphaName = objFuncDict_.getWord("alphaName");
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
                          << objFuncName << " " << objFuncPart << " not valid!"
                          << "Options: fixedDirection, parallelToFlow, normalToFlow."
                          << abort(FatalError);
    }

    if (fabs(mag(forceDir_) - 1.0) > 1.0e-8)
    {
        FatalErrorIn(" ") << "the magnitude of the direction parameter in "
                          << objFuncName << " " << objFuncPart << " is not 1.0!"
                          << abort(FatalError);
    }

    objFuncDict_.readEntry<scalar>("scale", scale_);

    // setup the connectivity for force, this is needed in Foam::DAJacCondFdW
    // need to determine the name of pressure because some buoyant OF solver use
    // p_rgh as pressure
    word pName = "p";
    if (mesh_.thisDb().foundObject<volScalarField>("p_rgh"))
    {
        pName = "p_rgh";
    }
#ifdef IncompressibleFlow
    // For incompressible flow, it depends on zero level
    // of U, nut, and p, and one level of U
    objFuncConInfo_ = {
        {"U", "nut", pName}, // level 0
        {"U"}}; // level 1
#endif

#ifdef CompressibleFlow
    // For compressible flow, it depends on zero level
    // of U, nut, T, and p, and one level of U
    objFuncConInfo_ = {
        {"U", "nut", "T", pName}, // level 0
        {"U"}}; // level 1
#endif

    // now replace nut with the corrected name for the selected turbulence model
    daModel.correctModelStates(objFuncConInfo_[0]);
}

/// calculate the value of objective function
void DAObjFuncForce::calcObjFunc(
    const labelList& objFuncFaceSources,
    const labelList& objFuncCellSources,
    scalarList& objFuncFaceValues,
    scalarList& objFuncCellValues,
    scalar& objFuncValue)
{
    /*
    Description:
        Calculate the force which consist of pressure and viscous components.
        This code is modified based on:
        src/functionObjects/forcces/forces.C

    Input:
        objFuncFaceSources: List of face source (index) for this objective
    
        objFuncCellSources: List of cell source (index) for this objective

    Output:
        objFuncFaceValues: the discrete value of objective for each face source (index). 
        This  will be used for computing df/dw in the adjoint.
    
        objFuncCellValues: the discrete value of objective on each cell source (index). 
        This will be used for computing df/dw in the adjoint.
    
        objFuncValue: the sum of objective, reduced across all processsors and scaled by "scale"
    */

    if (dirMode_ != "fixedDirection")
    {
        this->updateForceDir(forceDir_);
    }

    // initialize faceValues to zero
    forAll(objFuncFaceValues, idxI)
    {
        objFuncFaceValues[idxI] = 0.0;
    }
    // initialize objFunValue
    objFuncValue = 0.0;

    const objectRegistry& db = mesh_.thisDb();
    const volScalarField& p = db.lookupObject<volScalarField>("p");

    const surfaceVectorField::Boundary& Sfb = mesh_.Sf().boundaryField();

    tmp<volSymmTensorField> tdevRhoReff = daTurb_.devRhoReff();
    const volSymmTensorField::Boundary& devRhoReffb = tdevRhoReff().boundaryField();

    // calculate discrete force for each objFuncFace
    forAll(objFuncFaceSources, idxI)
    {
        const label& objFuncFaceI = objFuncFaceSources[idxI];
        label bFaceI = objFuncFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];

        // normal force
        vector fN(Sfb[patchI][faceI] * p.boundaryField()[patchI][faceI]);
        // tangential force
        vector fT(Sfb[patchI][faceI] & devRhoReffb[patchI][faceI]);
        // project the force to forceDir
        objFuncFaceValues[idxI] = scale_ * ((fN + fT) & forceDir_);

        objFuncValue += objFuncFaceValues[idxI];
    }

    // need to reduce the sum of force across all processors
    reduce(objFuncValue, sumOp<scalar>());

    return;
}

void DAObjFuncForce::updateForceDir(vector& forceDir)
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
