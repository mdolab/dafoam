/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFunctionMoment.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFunctionMoment, 0);
addToRunTimeSelectionTable(DAFunction, DAFunctionMoment, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunctionMoment::DAFunctionMoment(
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

    // for computing moment, first read in some parameters from functionDict_
    // these parameters are only for moment objective

    scalarList dir;
    functionDict_.readEntry<scalarList>("axis", dir);
    momentDir_[0] = dir[0];
    momentDir_[1] = dir[1];
    momentDir_[2] = dir[2];

    if (fabs(mag(momentDir_) - 1.0) > 1.0e-4)
    {
        FatalErrorIn(" ") << "the magnitude of the axis parameter in "
                          << functionName << " is not 1.0!"
                          << abort(FatalError);
    }

    scalarList center;
    functionDict_.readEntry<scalarList>("center", center);
    momentCenter_[0] = center[0];
    momentCenter_[1] = center[1];
    momentCenter_[2] = center[2];
}

/// calculate the value of objective function
scalar DAFunctionMoment::calcFunction()
{
    /*
    Description:
        Calculate the moment which consist of pressure and viscous components
        of force cross-producting the r vector wrt to a ref point
        This code for computiong force is modified based on:
        src/functionObjects/forcces/forces.C

    Output:
    
        functionValue: the sum of objective, reduced across all processsors and scaled by "scale"
    */

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
        // r vector
        vector rVec = mesh_.Cf().boundaryField()[patchI][faceI] - momentCenter_;
        // compute moment
        scalar val = scale_ * (rVec ^ (fN + fT)) & momentDir_;

        functionValue += val;
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
