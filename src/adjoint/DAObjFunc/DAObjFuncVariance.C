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

    varUpperBound_ = objFuncDict_.lookupOrDefault<scalar>("varUpperBound", 1e15);

    timeOperator_ = objFuncDict.lookupOrDefault<word>("timeOperator", "None");

    if (daIndex.adjStateNames.found(varName_))
    {
        objFuncConInfo_ = {{varName_}};
    }

    // first loop, we find the number of ref points. We expect the ref data
    // are the same across the runtime, so we read in the endTime and compute
    // nRefPoints
    nRefPoints_ = 0;
    scalar endTime = mesh_.time().endTime().value();
    scalar deltaT = mesh_.time().deltaT().value();
    label nTimeSteps = round(endTime / deltaT);

    if (varType_ == "scalar")
    {
        volScalarField varData(
            IOobject(
                varName_ + "Data",
                Foam::name(endTime),
                mesh_,
                IOobject::MUST_READ,
                IOobject::NO_WRITE),
            mesh_);

        forAll(varData, cellI)
        {
            if (fabs(varData[cellI]) < varUpperBound_)
            {
                nRefPoints_++;
            }
        }
    }
    else if (varType_ == "vector")
    {
        volVectorField varData(
            IOobject(
                varName_ + "Data",
                Foam::name(endTime),
                mesh_,
                IOobject::MUST_READ,
                IOobject::NO_WRITE),
            mesh_);

        forAll(varData, cellI)
        {
            for (label comp = 0; comp < 3; comp++)
            {
                if (fabs(varData[cellI][comp]) < varUpperBound_)
                {
                    nRefPoints_++;
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
    nRefPointsGlobal_ = nRefPoints_;
    reduce(nRefPointsGlobal_, sumOp<label>());

    refCellIndex_.setSize(nRefPoints_, -1);
    refCellComp_.setSize(nRefPoints_, -1);
    refValue_.setSize(nTimeSteps);

    // second loop, set refValue
    if (varType_ == "scalar")
    {
        for (label n = 0; n < nTimeSteps; n++)
        {
            scalar t = (n + 1) * deltaT;
            word timeName = Foam::name(t);

            refValue_[n].setSize(nRefPoints_);

            volScalarField varData(
                IOobject(
                    varName_ + "Data",
                    timeName,
                    mesh_,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE),
                mesh_);

            label pointI = 0;
            forAll(varData, cellI)
            {
                if (fabs(varData[cellI]) < varUpperBound_)
                {
                    refValue_[n][pointI] = varData[cellI];
                    refCellIndex_[pointI] = cellI;
                    pointI++;
                }
            }
        }
    }
    else if (varType_ == "vector")
    {
        for (label n = 0; n < nTimeSteps; n++)
        {
            scalar t = (n + 1) * deltaT;
            word timeName = Foam::name(t);

            refValue_[n].setSize(nRefPoints_);

            volVectorField varData(
                IOobject(
                    varName_ + "Data",
                    timeName,
                    mesh_,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE),
                mesh_);

            label pointI = 0;
            forAll(varData, cellI)
            {
                for (label comp = 0; comp < 3; comp++)
                {
                    if (fabs(varData[cellI][comp]) < varUpperBound_)
                    {
                        refValue_[n][pointI] = varData[cellI][comp];
                        refCellIndex_[pointI] = cellI;
                        refCellComp_[pointI] = comp;
                        pointI++;
                    }
                }
            }
        }
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

    const objectRegistry& db = mesh_.thisDb();

    label timeIndex = mesh_.time().timeIndex();

    if (varType_ == "scalar")
    {
        const volScalarField& var = db.lookupObject<volScalarField>(varName_);

        forAll(refCellIndex_, idxI)
        {
            label cellI = refCellIndex_[idxI];
            scalar varDif = (var[cellI] - refValue_[timeIndex - 1][idxI]);
            objFuncValue += scale_ * varDif * varDif;
        }
    }
    else if (varType_ == "vector")
    {
        const volVectorField& var = db.lookupObject<volVectorField>(varName_);

        forAll(refCellIndex_, idxI)
        {
            label cellI = refCellIndex_[idxI];
            label comp = refCellComp_[idxI];
            scalar varDif = (var[cellI][comp] - refValue_[timeIndex - 1][idxI]);
            objFuncValue += scale_ * varDif * varDif;
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

    objFuncValue /= nRefPointsGlobal_;

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
