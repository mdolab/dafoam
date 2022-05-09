/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAObjFuncTotalPressure.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAObjFuncTotalPressure, 0);
addToRunTimeSelectionTable(DAObjFunc, DAObjFuncTotalPressure, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAObjFuncTotalPressure::DAObjFuncTotalPressure(
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
      daTurb_(const_cast<DATurbulenceModel&>(daModel.getDATurbulenceModel())),
      rho_(daTurb_.getRho())
{
    // Assign type, this is common for all objectives
    objFuncDict_.readEntry<word>("type", objFuncType_);

    // setup the connectivity for total pressure, this is needed in Foam::DAJacCondFdW
    // NOTE: for pressureInlet velocity, U depends on phi
#ifdef CompressibleFlow
    // for compressible cases
    objFuncConInfo_ = {
        {"U", "T", "p", "phi"}, // level 0
        {"U", "T", "p", "phi"}}; // level 1
#endif
#ifdef IncompressibleFlow
    // for incompressible cases
    objFuncConInfo_ = {
        {"U", "p", "phi"}, // level 0
        {"U", "p", "phi"}}; // level 1
#endif

    objFuncDict_.readEntry<scalar>("scale", scale_);
}

/// calculate the value of objective function
void DAObjFuncTotalPressure::calcObjFunc(
    const labelList& objFuncFaceSources,
    const labelList& objFuncCellSources,
    scalarList& objFuncFaceValues,
    scalarList& objFuncCellValues,
    scalar& objFuncValue)
{
    /*
    Description:
        Calculate the total pressure TP=p+0.5*rho*U^2.

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

    // calculate the area of all the heat flux patches
    if (areaSum_ < 0.0)
    {
        areaSum_ = 0.0;
        forAll(objFuncFaceSources, idxI)
        {
            const label& objFuncFaceI = objFuncFaceSources[idxI];
            label bFaceI = objFuncFaceI - daIndex_.nLocalInternalFaces;
            const label patchI = daIndex_.bFacePatchI[bFaceI];
            const label faceI = daIndex_.bFaceFaceI[bFaceI];
            areaSum_ += mesh_.magSf().boundaryField()[patchI][faceI];
        }
        reduce(areaSum_, sumOp<scalar>());
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
    const volVectorField& U = db.lookupObject<volVectorField>("U");

    const volScalarField::Boundary& pBf = p.boundaryField();
    const volVectorField::Boundary& UBf = U.boundaryField();
    const volScalarField::Boundary& rhoBf = rho_.boundaryField();

    // calculate area weighted heat flux
    forAll(objFuncFaceSources, idxI)
    {
        const label& objFuncFaceI = objFuncFaceSources[idxI];
        label bFaceI = objFuncFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];

        scalar area = mesh_.magSf().boundaryField()[patchI][faceI];
        objFuncFaceValues[idxI] =
            pBf[patchI][faceI] + 0.5 * rhoBf[patchI][faceI] * mag(UBf[patchI][faceI]) * mag(UBf[patchI][faceI]);
        objFuncFaceValues[idxI] *= scale_ * area / areaSum_;

        objFuncValue += objFuncFaceValues[idxI];
    }

    // need to reduce the sum of force across all processors
    reduce(objFuncValue, sumOp<scalar>());

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
