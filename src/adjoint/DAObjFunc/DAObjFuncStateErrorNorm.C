/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAObjFuncStateErrorNorm.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAObjFuncStateErrorNorm, 0);
addToRunTimeSelectionTable(DAObjFunc, DAObjFuncStateErrorNorm, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAObjFuncStateErrorNorm::DAObjFuncStateErrorNorm(
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

    stateName_ = objFuncDict_.getWord("stateName");
    stateRefName_ = objFuncDict_.getWord("stateRefName");
    stateType_ = objFuncDict_.getWord("stateType");
    scale_ = objFuncDict_.getScalar("scale");

    // setup the connectivity, this is needed in Foam::DAJacCondFdW
    // this objFunc only depends on the state variable at the zero level cell
    if (DAUtility::isInList<word>(stateName_, daIndex.adjStateNames))
    {
        objFuncConInfo_ = {{stateName_}}; // level 0
    }
    else
    {
        objFuncConInfo_ = {{}}; // level 0
    }
}

/// calculate the value of objective function
void DAObjFuncStateErrorNorm::calcObjFunc(
    const labelList& objFuncFaceSources,
    const labelList& objFuncCellSources,
    scalarList& objFuncFaceValues,
    scalarList& objFuncCellValues,
    scalar& objFuncValue)
{
    /*
    Description:
        Calculate the stateErrorNorm
        f = scale * L2Norm( state-stateRef )

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

    // initialize to zero
    forAll(objFuncCellValues, idxI)
    {
        objFuncCellValues[idxI] = 0.0;
    }
    // initialize objFunValue
    objFuncValue = 0.0;

    const objectRegistry& db = mesh_.thisDb();
    if (stateType_ == "scalar")
    {
        const volScalarField& state = db.lookupObject<volScalarField>(stateName_);
        const volScalarField& stateRef = db.lookupObject<volScalarField>(stateRefName_);
        forAll(objFuncCellSources, idxI)
        {
            const label& cellI = objFuncCellSources[idxI];
            objFuncCellValues[idxI] = scale_ * (sqr(state[cellI] - stateRef[cellI]));
            objFuncValue += objFuncCellValues[idxI];
        }
    }
    else if (stateType_ == "vector")
    {
        const volVectorField& state = db.lookupObject<volVectorField>(stateName_);
        const volVectorField& stateRef = db.lookupObject<volVectorField>(stateRefName_);
        forAll(objFuncCellSources, idxI)
        {
            const label& cellI = objFuncCellSources[idxI];
            objFuncCellValues[idxI] = scale_ * (sqr(mag(state[cellI] - stateRef[cellI])));
            objFuncValue += objFuncCellValues[idxI];
        }
    }

    // need to reduce the sum of force across all processors
    reduce(objFuncValue, sumOp<scalar>());

    return;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
