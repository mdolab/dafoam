/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAObjFuncPower.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAObjFuncPower, 0);
addToRunTimeSelectionTable(DAObjFunc, DAObjFuncPower, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAObjFuncPower::DAObjFuncPower(
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

    // for computing moment, first read in some parameters from objFuncDict_
    // these parameters are only for moment objective

    // Assign type, this is common for all objectives
    objFuncDict_.readEntry<word>("type", objFuncType_);

    scalarList dir;
    objFuncDict_.readEntry<scalarList>("axis", dir);
    momentDir_[0] = dir[0];
    momentDir_[1] = dir[1];
    momentDir_[2] = dir[2];

    if (fabs(mag(momentDir_) - 1.0) > 1.0e-4)
    {
        FatalErrorIn(" ") << "the magnitude of the axis parameter in "
                          << objFuncName << " " << objFuncPart << " is not 1.0!"
                          << abort(FatalError);
    }

    scalarList center;
    objFuncDict_.readEntry<scalarList>("center", center);
    momentCenter_[0] = center[0];
    momentCenter_[1] = center[1];
    momentCenter_[2] = center[2];

    objFuncDict_.readEntry<scalar>("scale", scale_);

    // setup the connectivity for moment, this is needed in Foam::DAJacCondFdW
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
void DAObjFuncPower::calcObjFunc(
    const labelList& objFuncFaceSources,
    const labelList& objFuncCellSources,
    scalarList& objFuncFaceValues,
    scalarList& objFuncCellValues,
    scalar& objFuncValue)
{
    /*
    Description:
        Calculate the moment which consist of pressure and viscous components
        of force cross-producting the r vector wrt to a ref point
        This code for computiong force is modified based on:
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

    // reload the scale, which may be needed for multipoint optimization
    objFuncDict_.readEntry<scalar>("scale", scale_);

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
        // r vector
        vector rVec = mesh_.Cf().boundaryField()[patchI][faceI] - momentCenter_;
        // compute moment
        objFuncFaceValues[idxI] = scale_ * (rVec ^ (fN + fT)) & momentDir_;

        objFuncValue += objFuncFaceValues[idxI];
    }

    const IOMRFZoneListDF& MRF = mesh_.thisDb().lookupObject<IOMRFZoneListDF>("MRFProperties");
    scalar& omega = const_cast<scalar&>(MRF.getOmegaRef());
    objFuncValue *= omega;

    // need to reduce the sum of force across all processors
    reduce(objFuncValue, sumOp<scalar>());

    return;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
