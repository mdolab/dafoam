/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAObjFuncMassFlowRate.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAObjFuncMassFlowRate, 0);
addToRunTimeSelectionTable(DAObjFunc, DAObjFuncMassFlowRate, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAObjFuncMassFlowRate::DAObjFuncMassFlowRate(
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
    word pName = "p";
    if (mesh_.thisDb().foundObject<volScalarField>("p_rgh"))
    {
        pName = "p_rgh";
    }
    // for pressureInlet velocity, U depends on phi
    objFuncConInfo_ = {{"U", "T", pName, "phi"}};

    objFuncDict_.readEntry<scalar>("scale", scale_);
}

/// calculate the value of objective function
void DAObjFuncMassFlowRate::calcObjFunc(
    const labelList& objFuncFaceSources,
    const labelList& objFuncCellSources,
    scalarList& objFuncFaceValues,
    scalarList& objFuncCellValues,
    scalar& objFuncValue)
{
    /*
    Description:
        Calculate mass flow rate M = integral( rho*U*dS )

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

    const objectRegistry& db = mesh_.thisDb();
    const volVectorField& U = db.lookupObject<volVectorField>("U");

    const volVectorField::Boundary& UBf = U.boundaryField();
    const volScalarField::Boundary& rhoBf = rho_.boundaryField();

    forAll(objFuncFaceSources, idxI)
    {
        const label& objFuncFaceI = objFuncFaceSources[idxI];
        label bFaceI = objFuncFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];

        vector US = UBf[patchI][faceI];
        vector Sf = mesh_.Sf().boundaryField()[patchI][faceI];
        scalar rhoS = rhoBf[patchI][faceI];
        scalar mfr = rhoS * (US & Sf);
        objFuncFaceValues[idxI] = mfr * scale_;

        objFuncValue += objFuncFaceValues[idxI];
    }

    // need to reduce the sum of force across all processors
    reduce(objFuncValue, sumOp<scalar>());

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
