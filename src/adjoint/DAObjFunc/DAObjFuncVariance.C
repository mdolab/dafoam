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
        objFuncDict),
      daTurb_(daModel.getDATurbulenceModel())
{

    // Assign type, this is common for all objectives
    objFuncDict_.readEntry<word>("type", objFuncType_);

    objFuncDict_.readEntry<scalar>("scale", scale_);

    // read the parameters

    objFuncDict_.readEntry<word>("mode", mode_);

    if (mode_ == "surface")
    {
        objFuncDict_.readEntry<wordList>("surfaceNames", surfaceNames_);
    }

    if (mode_ == "probePoint")
    {
        objFuncDict_.readEntry<List<List<scalar>>>("probePointCoords", probePointCoords_);
    }

    objFuncDict_.readEntry<word>("varName", varName_);

    objFuncDict_.readEntry<word>("varType", varType_);

    if (varType_ == "vector")
    {
        objFuncDict_.readEntry<labelList>("components", components_);
    }

    timeDependentRefData_ = objFuncDict_.getLabel("timeDependentRefData");

    if (daIndex.adjStateNames.found(varName_))
    {
        objFuncConInfo_ = {{varName_}};
    }

    // get the time information
    scalar endTime = mesh_.time().endTime().value();
    scalar deltaT = mesh_.time().deltaT().value();
    label nTimeSteps = round(endTime / deltaT);

    word checkRefDataFolder;
    label nRefValueInstances;
    if (timeDependentRefData_)
    {
        // check if we can find the ref data in the endTime folder
        checkRefDataFolder = Foam::name(endTime);
        nRefValueInstances = nTimeSteps;
    }
    else
    {
        // check if we can find the ref data in the 0 folder
        checkRefDataFolder = Foam::name(0);
        nRefValueInstances = 1;
    }

    // check if the reference data files exist
    isRefData_ = 1;
    if (varType_ == "scalar")
    {
        volScalarField varData(
            IOobject(
                varName_ + "Data",
                checkRefDataFolder,
                mesh_,
                IOobject::READ_IF_PRESENT,
                IOobject::NO_WRITE),
            mesh_,
            dimensionedScalar("dummy", dimensionSet(0, 0, 0, 0, 0, 0, 0), 1e16),
            "zeroGradient");

        if (gSumMag(varData) > 1e16)
        {
            isRefData_ = 0;
        }
    }
    else if (varType_ == "vector")
    {
        volVectorField varData(
            IOobject(
                varName_ + "Data",
                checkRefDataFolder,
                mesh_,
                IOobject::READ_IF_PRESENT,
                IOobject::NO_WRITE),
            mesh_,
            dimensionedVector("dummy", dimensionSet(0, 0, 0, 0, 0, 0, 0), {1e16, 1e16, 1e16}),
            "zeroGradient");

        if (mag(gSumCmptMag(varData)) > 1e16)
        {
            isRefData_ = 0;
        }
    }
    else
    {
        FatalErrorIn("") << "varType " << varType_ << " not supported!"
                         << "Options are: scalar or vector"
                         << abort(FatalError);
    }

    // if no varData file found, we print out a warning
    if (isRefData_ == 0)
    {
        Info << endl;
        Info << "**************************************************************************** " << endl;
        Info << "*         WARNING! Can't find data files or can't find valid               * " << endl;
        Info << "*         values in data files for the variance objFunc " << varName_ << "       * " << endl;
        Info << "**************************************************************************** " << endl;
        Info << endl;
    }
    else
    {
        // varData file found, we need to read in the ref values for all time instances

        refValue_.setSize(nRefValueInstances);

        // set refValue
        if (varType_ == "scalar")
        {
            for (label n = 0; n < nRefValueInstances; n++)
            {
                word timeName;
                if (timeDependentRefData_)
                {
                    scalar t = (n + 1) * deltaT;
                    timeName = Foam::name(t);
                }
                else
                {
                    timeName = Foam::name(0);
                }

                volScalarField varData(
                    IOobject(
                        varName_ + "Data",
                        timeName,
                        mesh_,
                        IOobject::MUST_READ,
                        IOobject::NO_WRITE),
                    mesh_);

                nRefPoints_ = 0;

                if (mode_ == "probePoint")
                {
                    probeCellIndex_.setSize(0);

                    forAll(probePointCoords_, idxI)
                    {
                        point pointCoord = {probePointCoords_[idxI][0], probePointCoords_[idxI][1], probePointCoords_[idxI][2]};
                        label cellI = mesh_.findCell(pointCoord);
                        if (cellI >= 0)
                        {
                            probeCellIndex_.append(cellI);
                            refValue_[n].append(varData[cellI]);
                            nRefPoints_++;
                        }
                    }
                }
                else if (mode_ == "surface")
                {
                    forAll(surfaceNames_, idxI)
                    {
                        word surfaceName = surfaceNames_[idxI];
                        label patchI = mesh_.boundaryMesh().findPatchID(surfaceName);
                        if (patchI >= 0)
                        {
                            forAll(varData.boundaryField()[patchI], faceI)
                            {
                                refValue_[n].append(varData.boundaryField()[patchI][faceI]);
                                nRefPoints_++;
                            }
                        }
                        else
                        {
                            FatalErrorIn("") << "surfaceName " << surfaceName << " not found!"
                                             << abort(FatalError);
                        }
                    }
                }
                else if (mode_ == "field")
                {
                    forAll(varData, cellI)
                    {
                        refValue_[n].append(varData[cellI]);
                        nRefPoints_++;
                    }
                }
                else
                {
                    FatalErrorIn("") << "mode " << mode_ << " not supported!"
                                     << "Options are: probePoint, field, or surface"
                                     << abort(FatalError);
                }
            }
        }
        else if (varType_ == "vector")
        {
            for (label n = 0; n < nRefValueInstances; n++)
            {
                word timeName;
                if (timeDependentRefData_)
                {
                    scalar t = (n + 1) * deltaT;
                    timeName = Foam::name(t);
                }
                else
                {
                    timeName = Foam::name(0);
                }

                volVectorField varData(
                    IOobject(
                        varName_ + "Data",
                        timeName,
                        mesh_,
                        IOobject::MUST_READ,
                        IOobject::NO_WRITE),
                    mesh_);

                nRefPoints_ = 0;

                if (mode_ == "probePoint")
                {
                    probeCellIndex_.setSize(0);

                    forAll(probePointCoords_, idxI)
                    {
                        point pointCoord = {probePointCoords_[idxI][0], probePointCoords_[idxI][1], probePointCoords_[idxI][2]};
                        label cellI = mesh_.findCell(pointCoord);
                        if (cellI >= 0)
                        {
                            probeCellIndex_.append(cellI);
                            forAll(components_, idxJ)
                            {
                                label compI = components_[idxJ];
                                refValue_[n].append(varData[cellI][compI]);
                                nRefPoints_++;
                            }
                        }
                    }
                }
                else if (mode_ == "surface")
                {
                    forAll(surfaceNames_, idxI)
                    {
                        word surfaceName = surfaceNames_[idxI];
                        label patchI = mesh_.boundaryMesh().findPatchID(surfaceName);
                        if (patchI >= 0)
                        {
                            forAll(varData.boundaryField()[patchI], faceI)
                            {
                                forAll(components_, idxJ)
                                {
                                    label compI = components_[idxJ];
                                    refValue_[n].append(varData.boundaryField()[patchI][faceI][compI]);
                                    nRefPoints_++;
                                }
                            }
                        }
                        else
                        {
                            FatalErrorIn("") << "surfaceName " << surfaceName << " not found!"
                                             << abort(FatalError);
                        }
                    }
                }
                else if (mode_ == "field")
                {
                    forAll(varData, cellI)
                    {
                        forAll(components_, idxJ)
                        {
                            label compI = components_[idxJ];
                            refValue_[n].append(varData[cellI][compI]);
                            nRefPoints_++;
                        }
                    }
                }
                else
                {
                    FatalErrorIn("") << "mode " << mode_ << " not supported!"
                                     << "Options are: probePoint, field, or surface"
                                     << abort(FatalError);
                }
            }
        }

        reduce(nRefPoints_, sumOp<label>());

        Info << "Find " << nRefPoints_ << " reference points for variance of " << varName_ << endl;
        if (nRefPoints_ == 0)
        {
            FatalErrorIn("") << "varData field exists but one can not find any valid data!"
                             << abort(FatalError);
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

    if (isRefData_)
    {

        const objectRegistry& db = mesh_.thisDb();

        label timeIndex;
        if (timeDependentRefData_)
        {
            timeIndex = mesh_.time().timeIndex();
        }
        else
        {
            timeIndex = 1;
        }

        if (varName_ == "wallShearStress")
        {
            volSymmTensorField devRhoReff = daTurb_.devRhoReff();

            label pointI = 0;
            forAll(surfaceNames_, idxI)
            {
                word surfaceName = surfaceNames_[idxI];
                label patchI = mesh_.boundaryMesh().findPatchID(surfaceName);

                if (mesh_.boundaryMesh().size() > 0)
                {
                    const vectorField& SfB = mesh_.Sf().boundaryField()[patchI];
                    const scalarField& magSfB = mesh_.magSf().boundaryField()[patchI];
                    const symmTensorField& ReffB = devRhoReff.boundaryField()[patchI];

                    vectorField shearB = (-SfB / magSfB) & ReffB;

                    forAll(shearB, faceI)
                    {
                        forAll(components_, idxJ)
                        {
                            label compI = components_[idxJ];
                            scalar varDif = (shearB[faceI][compI] - refValue_[timeIndex - 1][pointI]);
                            objFuncValue += scale_ * varDif * varDif;
                            pointI++;
                        }
                    }
                }
            }
        }
        else
        {
            if (varType_ == "scalar")
            {
                const volScalarField& var = db.lookupObject<volScalarField>(varName_);

                if (mode_ == "probePoint")
                {
                    forAll(probeCellIndex_, idxI)
                    {
                        label cellI = probeCellIndex_[idxI];
                        scalar varDif = (var[cellI] - refValue_[timeIndex - 1][idxI]);
                        objFuncValue += scale_ * varDif * varDif;
                    }
                }
                else if (mode_ == "surface")
                {
                    label pointI = 0;
                    forAll(surfaceNames_, idxI)
                    {
                        word surfaceName = surfaceNames_[idxI];
                        label patchI = mesh_.boundaryMesh().findPatchID(surfaceName);
                        forAll(var.boundaryField()[patchI], faceI)
                        {
                            scalar varDif = (var.boundaryField()[patchI][faceI] - refValue_[timeIndex - 1][pointI]);
                            objFuncValue += scale_ * varDif * varDif;
                            pointI++;
                        }
                    }
                }
                else if (mode_ == "field")
                {
                    forAll(var, cellI)
                    {
                        scalar varDif = (var[cellI] - refValue_[timeIndex - 1][cellI]);
                        objFuncValue += scale_ * varDif * varDif;
                    }
                }
            }
            else if (varType_ == "vector")
            {
                const volVectorField& var = db.lookupObject<volVectorField>(varName_);

                if (mode_ == "probePoint")
                {
                    label pointI = 0;
                    forAll(probeCellIndex_, idxI)
                    {
                        label cellI = probeCellIndex_[idxI];
                        forAll(components_, idxJ)
                        {
                            label compI = components_[idxJ];
                            scalar varDif = (var[cellI][compI] - refValue_[timeIndex - 1][pointI]);
                            objFuncValue += scale_ * varDif * varDif;
                            pointI++;
                        }
                    }
                }
                else if (mode_ == "surface")
                {
                    label pointI = 0;
                    forAll(surfaceNames_, idxI)
                    {
                        word surfaceName = surfaceNames_[idxI];
                        label patchI = mesh_.boundaryMesh().findPatchID(surfaceName);
                        forAll(var.boundaryField()[patchI], faceI)
                        {
                            forAll(components_, idxJ)
                            {
                                label compI = components_[idxJ];
                                scalar varDif = (var.boundaryField()[patchI][faceI][compI] - refValue_[timeIndex - 1][pointI]);
                                objFuncValue += scale_ * varDif * varDif;
                                pointI++;
                            }
                        }
                    }
                }
                else if (mode_ == "field")
                {
                    label pointI = 0;
                    forAll(var, cellI)
                    {
                        forAll(components_, idxJ)
                        {
                            label compI = components_[idxJ];
                            scalar varDif = (var[cellI][compI] - refValue_[timeIndex - 1][pointI]);
                            objFuncValue += scale_ * varDif * varDif;
                            pointI++;
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
        }
        // need to reduce the sum of force across all processors
        reduce(objFuncValue, sumOp<scalar>());

        if (nRefPoints_ != 0)
        {
            objFuncValue /= nRefPoints_;
        }
    }

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
