/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

\*---------------------------------------------------------------------------*/

#include "DAFunctionVariableVolSum.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFunctionVariableVolSum, 0);
addToRunTimeSelectionTable(DAFunction, DAFunctionVariableVolSum, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunctionVariableVolSum::DAFunctionVariableVolSum(
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
        functionName)
{

    functionDict_.readEntry<word>("varName", varName_);

    functionDict_.readEntry<word>("varType", varType_);

    functionDict_.readEntry<label>("index", index_);

    isSquare_ = functionDict_.lookupOrDefault<label>("isSquare", 0);

    multiplyVol_ = functionDict_.lookupOrDefault<label>("multiplyVol", 1);

    divByTotalVol_ = functionDict_.lookupOrDefault<label>("divByTotalVol", 0);

    invertField_ = functionDict_.lookupOrDefault<label>("invertField", 0);

    invertVal_ = functionDict_.lookupOrDefault<label>("invertVal", 1); // Binary inversion from 0 to 1, vice versa
}

/// calculate the value of objective function
scalar DAFunctionVariableVolSum::calcFunction()
{
    /*
    Description:
        Calculate the obj = mesh-volume * variable (whether to take a square of the variable
        depends on isSquare)
    */

    // initialize objFunValue
    scalar functionValue = 0.0;

    const objectRegistry& db = mesh_.thisDb();

    scalar totalVol = 1.0;

    if (divByTotalVol_)
    {
        forAll(mesh_.cells(), cellI)
        {
            totalVol += mesh_.V()[cellI];
        }
        reduce(totalVol, sumOp<scalar>());
    }

    if (varType_ == "scalar")
    {
        const volScalarField& var = db.lookupObject<volScalarField>(varName_);
        // calculate mass
        forAll(cellSources_, idxI)
        {
            const label& cellI = cellSources_[idxI];
            scalar volume = 1.0;
            if (multiplyVol_)
            {
                volume = mesh_.V()[cellI];
            }
            if (invertField_)
            {
                var[cellI] = invertVal_ - var[cellI];
            }
            if (isSquare_)
            {
                functionValue += scale_ * volume * var[cellI] * var[cellI];
            }
            else
            {
                functionValue += scale_ * volume * var[cellI];
            }
        }
    }
    else if (varType_ == "vector")
    {
       
        const volVectorField& var = db.lookupObject<volVectorField>(varName_);
        // calculate mass
        forAll(cellSources_, idxI)
        {
            const label& cellI = cellSources_[idxI];
            scalar volume = 1.0;
            if (multiplyVol_)
            {
                volume = mesh_.V()[cellI];
            }
            if (invertField_)
            {
                val[cellI][index_] = invertVal_ - var[cellI][index_]
            }
            if (isSquare_)
            {
                functionValue += scale_ * volume * var[cellI][index_] * var[cellI][index_];
            }
            else
            {
                functionValue += scale_ * volume * var[cellI][index_];
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
    reduce(functionValue, sumOp<scalar>());

    functionValue /= totalVol;

    // check if we need to calculate refDiff.
    this->calcRefVar(functionValue);

    return functionValue;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
