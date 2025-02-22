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
    const word functionName)
    : DAFunction(
        mesh,
        daOption,
        daModel,
        daIndex,
        functionName),
      daTurb_(daModel.getDATurbulenceModel())
{

    // for computing force, first read in some parameters from functionDict_
    // these parameters are only for force objective

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
        word patchVelocityInputName = functionDict_.getWord("patchVelocityInputName");
        dictionary patchVSubDict = daOption_.getAllOptions().subDict("inputInfo").subDict(patchVelocityInputName);
        HashTable<label> axisIndices;
        axisIndices.set("x", 0);
        axisIndices.set("y", 1);
        axisIndices.set("z", 2);
        word flowAxis = patchVSubDict.getWord("flowAxis");
        word normalAxis = patchVSubDict.getWord("normalAxis");
        flowAxisIndex_ = axisIndices[flowAxis];
        normalAxisIndex_ = axisIndices[normalAxis];
    }
    else
    {
        FatalErrorIn(" ") << "directionMode for "
                          << functionName << " not valid!"
                          << "Options: fixedDirection, parallelToFlow, normalToFlow."
                          << abort(FatalError);
    }

    if (fabs(mag(forceDir_) - 1.0) > 1.0e-8)
    {
        FatalErrorIn(" ") << "the magnitude of the direction parameter in "
                          << functionName << " is not 1.0!"
                          << abort(FatalError);
    }
}

/// calculate the value of objective function
scalar DAFunctionForce::calcFunction()
{
    /*
    Description:
        Calculate the force which consist of pressure and viscous components.
        This code is modified based on:
        src/functionObjects/forcces/forces.C

    Output:
        functionValue: the sum of objective, reduced across all processsors and scaled by "scale"
    */

    // dynamically update the force direction,  if either parallelToFlow or normalToFlow is active.
    if (dirMode_ != "fixedDirection")
    {
        // we need to read the velocity magnitude and aoa
        // from DAGlobalVar::patchVelocity. NOTE: DAGlobalVar::patchVelocity is already set
        // by DAInputPatchVelocity
        DAGlobalVar& globalVar =
            const_cast<DAGlobalVar&>(mesh_.thisDb().lookupObject<DAGlobalVar>("DAGlobalVar"));
        scalar aoaDeg = globalVar.patchVelocity[1];
        scalar aoaRad = aoaDeg * constant::mathematical::pi / 180.0;

        scalar compA = cos(aoaRad);
        scalar compB = sin(aoaRad);
        if (dirMode_ == "parallelToFlow")
        {
            forceDir_[flowAxisIndex_] = compA;
            forceDir_[normalAxisIndex_] = compB;
        }
        else if (dirMode_ == "normalToFlow")
        {
            forceDir_[flowAxisIndex_] = -compB;
            forceDir_[normalAxisIndex_] = compA;
        }
    }

    // initialize objFunValue
    scalar functionValue = 0.0;

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

    return functionValue;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
