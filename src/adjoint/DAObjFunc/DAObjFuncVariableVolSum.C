/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAObjFuncVariableVolSum.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAObjFuncVariableVolSum, 0);
addToRunTimeSelectionTable(DAObjFunc, DAObjFuncVariableVolSum, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAObjFuncVariableVolSum::DAObjFuncVariableVolSum(
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

    if (daOption.getAllOptions().getWord("adjJacobianOption") != "JacobianFree")
    {
        FatalErrorIn("") << "Only support adjJacobianOption = JacobianFree"
                         << abort(FatalError);
    }

    // Assign type, this is common for all objectives
    objFuncDict_.readEntry<word>("type", objFuncType_);

    objFuncDict_.readEntry<scalar>("scale", scale_);

    objFuncDict_.readEntry<word>("varName", varName_);

    objFuncDict_.readEntry<word>("varType", varType_);

    objFuncDict_.readEntry<label>("component", component_);

    objFuncDict_.readEntry<label>("isSquare", isSquare_);
}

/// calculate the value of objective function
void DAObjFuncVariableVolSum::calcObjFunc(
    const labelList& objFuncFaceSources,
    const labelList& objFuncCellSources,
    scalarList& objFuncFaceValues,
    scalarList& objFuncCellValues,
    scalar& objFuncValue)
{
    /*
    Description:
        Calculate the obj = mesh volume * variable (whether to take a square of the variable
        depends on isSquare)

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

    // initialize objFunValue
    objFuncValue = 0.0;

    const objectRegistry& db = mesh_.thisDb();

    if (varType_ == "scalar")
    {
        const volScalarField& var = db.lookupObject<volScalarField>(varName_);
        // calculate mass
        forAll(objFuncCellSources, idxI)
        {
            const label& cellI = objFuncCellSources[idxI];
            scalar volume = mesh_.V()[cellI];
            if (isSquare_)
            {
                objFuncValue += scale_ * volume * var[cellI] * var[cellI];
            }
            else
            {
                objFuncValue += scale_ * volume * var[cellI];
            }
        }
    }
    else if (varType_ == "vector")
    {
        const volVectorField& var = db.lookupObject<volVectorField>(varName_);
        // calculate mass
        forAll(objFuncCellSources, idxI)
        {
            const label& cellI = objFuncCellSources[idxI];
            scalar volume = mesh_.V()[cellI];
            if (isSquare_)
            {
                objFuncValue += scale_ * volume * var[cellI][component_] * var[cellI][component_];
            }
            else
            {
                objFuncValue += scale_ * volume * var[cellI][component_];
            }
        }
    }
    else
    {
        FatalErrorIn("") << "varType " << varType_ << " not supported!"
                         << "Options are: scalar or vector"
                         << abort(FatalError);
    }

    // need to reduce the sum of force across all processors
    reduce(objFuncValue, sumOp<scalar>());

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
