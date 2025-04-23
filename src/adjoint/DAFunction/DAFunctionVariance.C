/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFunctionVariance.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFunctionVariance, 0);
addToRunTimeSelectionTable(DAFunction, DAFunctionVariance, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunctionVariance::DAFunctionVariance(
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
        functionName),
      daTurb_(const_cast<DATurbulenceModel&>(daModel.getDATurbulenceModel()))
{

    // read the parameters

    functionDict_.readEntry<word>("mode", mode_);

    if (mode_ == "surface")
    {
        // check if the faceSources is set
        label size = faceSources_.size();
        reduce(size, sumOp<label>());
        if (size == 0)
        {
            FatalErrorIn("") << "surface mode is used but patchToFace is not set!"
                             << abort(FatalError);
        }
    }

    if (mode_ == "probePoint")
    {
        functionDict_.readEntry<List<List<scalar>>>("probePointCoords", probePointCoords_);
    }

    functionDict_.readEntry<word>("varName", varName_);

    if (varName_ == "wallHeatFlux")
    {
        if (daTurb_.getTurbModelType() == "incompressible")
        {
            // initialize the Prandtl number from transportProperties
            IOdictionary transportProperties(
                IOobject(
                    "transportProperties",
                    mesh.time().constant(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE,
                    false));
            // for incompressible flow, we need to read Cp from transportProperties
            Cp_ = readScalar(transportProperties.lookup("Cp"));
        }
    }

    functionDict_.readEntry<word>("varType", varType_);

    if (varType_ == "vector")
    {
        functionDict_.readEntry<labelList>("indices", indices_);
    }

    timeDependentRefData_ = functionDict_.getLabel("timeDependentRefData");

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
        Info << "*         values in data files for the variance function " << varName_ << "       * " << endl;
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
                        label cellI = DAUtility::myFindCell(mesh_, pointCoord);
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
                    forAll(faceSources_, idxI)
                    {
                        const label& functionFaceI = faceSources_[idxI];
                        label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
                        const label patchI = daIndex_.bFacePatchI[bFaceI];
                        const label faceI = daIndex_.bFaceFaceI[bFaceI];
                        refValue_[n].append(varData.boundaryField()[patchI][faceI]);
                        nRefPoints_++;
                    }
                }
                else if (mode_ == "field")
                {
                    forAll(cellSources_, idxI)
                    {
                        label cellI = cellSources_[idxI];
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
                        label cellI = DAUtility::myFindCell(mesh_, pointCoord);
                        if (cellI >= 0)
                        {
                            probeCellIndex_.append(cellI);
                            forAll(indices_, idxJ)
                            {
                                label compI = indices_[idxJ];
                                refValue_[n].append(varData[cellI][compI]);
                                nRefPoints_++;
                            }
                        }
                    }
                }
                else if (mode_ == "surface")
                {
                    forAll(faceSources_, idxI)
                    {
                        const label& functionFaceI = faceSources_[idxI];
                        label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
                        const label patchI = daIndex_.bFacePatchI[bFaceI];
                        const label faceI = daIndex_.bFaceFaceI[bFaceI];

                        forAll(indices_, idxJ)
                        {
                            label compI = indices_[idxJ];
                            refValue_[n].append(varData.boundaryField()[patchI][faceI][compI]);
                            nRefPoints_++;
                        }
                    }
                }
                else if (mode_ == "field")
                {
                    forAll(cellSources_, idxI)
                    {
                        label cellI = cellSources_[idxI];
                        forAll(indices_, idxJ)
                        {
                            label compI = indices_[idxJ];
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
scalar DAFunctionVariance::calcFunction()
{
    /*
    Description:
        Calculate the variance
    */

    // initialize objFunValue
    scalar functionValue = 0.0;

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
            forAll(faceSources_, idxI)
            {
                const label& functionFaceI = faceSources_[idxI];
                label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
                const label patchI = daIndex_.bFacePatchI[bFaceI];
                const label faceI = daIndex_.bFaceFaceI[bFaceI];

                const vectorField& SfB = mesh_.Sf().boundaryField()[patchI];
                const scalarField& magSfB = mesh_.magSf().boundaryField()[patchI];
                const symmTensorField& ReffB = devRhoReff.boundaryField()[patchI];

                vectorField shearB = (-SfB / magSfB) & ReffB;

                forAll(indices_, idxJ)
                {
                    label compI = indices_[idxJ];
                    scalar varDif = (shearB[faceI][compI] - refValue_[timeIndex - 1][pointI]);
                    functionValue += scale_ * varDif * varDif;
                    pointI++;
                }
            }
        }
        else if (varName_ == "wallHeatFlux")
        {
            if (daTurb_.getTurbModelType() == "incompressible")
            {
                // incompressible flow does not have he, so we do H = Cp * alphaEff * dT/dz
                const volScalarField& T = mesh_.thisDb().lookupObject<volScalarField>("T");
                volScalarField alphaEff = daTurb_.alphaEff();
                const volScalarField::Boundary& TBf = T.boundaryField();
                const volScalarField::Boundary& alphaEffBf = alphaEff.boundaryField();

                label pointI = 0;
                forAll(faceSources_, idxI)
                {
                    const label& functionFaceI = faceSources_[idxI];
                    label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
                    const label patchI = daIndex_.bFacePatchI[bFaceI];
                    const label faceI = daIndex_.bFaceFaceI[bFaceI];

                    scalarField hfx = Cp_ * alphaEffBf[patchI] * TBf[patchI].snGrad();

                    scalar varDif = (hfx[faceI] - refValue_[timeIndex - 1][pointI]);
                    functionValue += scale_ * varDif * varDif;
                    pointI++;
                }
            }

            if (daTurb_.getTurbModelType() == "compressible")
            {
                // compressible flow, H = alphaEff * dHE/dz
                fluidThermo& thermo_(const_cast<fluidThermo&>(
                    mesh_.thisDb().lookupObject<fluidThermo>("thermophysicalProperties")));
                volScalarField& he = thermo_.he();
                const volScalarField::Boundary& heBf = he.boundaryField();
                volScalarField alphaEff = daTurb_.alphaEff();
                const volScalarField::Boundary& alphaEffBf = alphaEff.boundaryField();

                label pointI = 0;
                forAll(faceSources_, idxI)
                {
                    const label& functionFaceI = faceSources_[idxI];
                    label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
                    const label patchI = daIndex_.bFacePatchI[bFaceI];
                    const label faceI = daIndex_.bFaceFaceI[bFaceI];

                    scalarField hfx = alphaEffBf[patchI] * heBf[patchI].snGrad();

                    scalar varDif = (hfx[faceI] - refValue_[timeIndex - 1][pointI]);
                    functionValue += scale_ * varDif * varDif;
                    pointI++;
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
                        functionValue += scale_ * varDif * varDif;
                    }
                }
                else if (mode_ == "surface")
                {
                    label pointI = 0;
                    forAll(faceSources_, idxI)
                    {
                        const label& functionFaceI = faceSources_[idxI];
                        label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
                        const label patchI = daIndex_.bFacePatchI[bFaceI];
                        const label faceI = daIndex_.bFaceFaceI[bFaceI];
                        scalar varDif = (var.boundaryField()[patchI][faceI] - refValue_[timeIndex - 1][pointI]);
                        functionValue += scale_ * varDif * varDif;
                        pointI++;
                    }
                }
                else if (mode_ == "field")
                {
                    forAll(cellSources_, idxI)
                    {
                        label cellI = cellSources_[idxI];
                        scalar varDif = (var[cellI] - refValue_[timeIndex - 1][cellI]);
                        functionValue += scale_ * varDif * varDif;
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
                        forAll(indices_, idxJ)
                        {
                            label compI = indices_[idxJ];
                            scalar varDif = (var[cellI][compI] - refValue_[timeIndex - 1][pointI]);
                            functionValue += scale_ * varDif * varDif;
                            pointI++;
                        }
                    }
                }
                else if (mode_ == "surface")
                {
                    label pointI = 0;
                    forAll(faceSources_, idxI)
                    {
                        const label& functionFaceI = faceSources_[idxI];
                        label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
                        const label patchI = daIndex_.bFacePatchI[bFaceI];
                        const label faceI = daIndex_.bFaceFaceI[bFaceI];
                        forAll(indices_, idxJ)
                        {
                            label compI = indices_[idxJ];
                            scalar varDif = (var.boundaryField()[patchI][faceI][compI] - refValue_[timeIndex - 1][pointI]);
                            functionValue += scale_ * varDif * varDif;
                            pointI++;
                        }
                    }
                }
                else if (mode_ == "field")
                {
                    label pointI = 0;
                    forAll(cellSources_, idxI)
                    {
                        label cellI = cellSources_[idxI];
                        forAll(indices_, idxJ)
                        {
                            label compI = indices_[idxJ];
                            scalar varDif = (var[cellI][compI] - refValue_[timeIndex - 1][pointI]);
                            functionValue += scale_ * varDif * varDif;
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
        reduce(functionValue, sumOp<scalar>());

        if (nRefPoints_ != 0)
        {
            functionValue /= nRefPoints_;
        }
    }

    return functionValue;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
