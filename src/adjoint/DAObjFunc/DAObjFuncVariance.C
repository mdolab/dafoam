/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAObjFuncVariance.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAObjFuncVariance, 0);
addToRunTimeSelectionTable(DAObjFunc, DAObjFuncVariance, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAObjFuncVariance::DAObjFuncVariance(
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

    objFuncDict_.readEntry<scalar>("scale", scale_);

    objFuncDict_.readEntry<word>("varName", varName_);

    objFuncDict_.readEntry<word>("varType", varType_);

    timeOperator_ = objFuncDict.lookupOrDefault<word>("timeOperator", "None");

    if (daIndex.adjStateNames.found(varName_))
    {
        objFuncConInfo_ = {{varName_}};
    }
}

/// calculate the value of objective function
void DAObjFuncVariance::calcObjFunc(
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
    label nRefPoints = 0;

    const objectRegistry& db = mesh_.thisDb();

    if (varType_ == "scalar")
    {
        const volScalarField& var = db.lookupObject<volScalarField>(varName_);

        volScalarField varData(
            IOobject(
                varName_ + "Data",
                mesh_.time().timeName(),
                mesh_,
                IOobject::MUST_READ,
                IOobject::NO_WRITE),
            mesh_);

        forAll(var, cellI)
        {
            if (varData[cellI] < 1e16)
            {
                scalar varDif = (var[cellI] - varData[cellI]);
                objFuncValue += scale_ * varDif * varDif;
                nRefPoints++;
            }
        }
    }
    else if (varType_ == "vector")
    {
        const volVectorField& var = db.lookupObject<volVectorField>(varName_);

        volVectorField varData(
            IOobject(
                varName_ + "Data",
                mesh_.time().timeName(),
                mesh_,
                IOobject::MUST_READ,
                IOobject::NO_WRITE),
            mesh_);

        forAll(var, cellI)
        {
            for (label comp = 0; comp < 3; comp++)
            {
                if (varData[cellI][comp] < 1e16)
                {
                    scalar varDif = (var[cellI][comp] - varData[cellI][comp]);
                    objFuncValue += scale_ * varDif * varDif;
                    nRefPoints++;
                }
            }
        }
    }
    else
    {
        FatalErrorIn("") << "varType " << varType_ << " not supported!"
                         << "Options are: scalar or vector"
                         << abort(FatalError);
    }

    // reduce the sum of all the ref points for averaging
    reduce(nRefPoints, sumOp<label>());

    // need to reduce the sum of force across all processors
    reduce(objFuncValue, sumOp<scalar>());

    objFuncValue /= nRefPoints;

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
