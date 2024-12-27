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
    scalarList dir;
    functionDict_.readEntry<scalarList>("direction", dir);
    forceDir_[0] = dir[0];
    forceDir_[1] = dir[1];
    forceDir_[2] = dir[2];
    
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

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
