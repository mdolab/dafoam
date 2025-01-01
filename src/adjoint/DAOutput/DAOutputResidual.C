/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAOutputResidual.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAOutputResidual, 0);
addToRunTimeSelectionTable(DAOutput, DAOutputResidual, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAOutputResidual::DAOutputResidual(
    const word outputName,
    const word outputType,
    fvMesh& mesh,
    const DAOption& daOption,
    DAModel& daModel,
    const DAIndex& daIndex,
    DAResidual& daResidual,
    UPtrList<DAFunction>& daFunctionList)
    : DAOutput(
        outputName,
        outputType,
        mesh,
        daOption,
        daModel,
        daIndex,
        daResidual,
        daFunctionList)
{
}

void DAOutputResidual::run(scalarList& output)
{
    /*
    Description:
        Compute OF's residual variables and then assign them to the output array
    */

    label isPC = 0;

    // calculate the residual
    dictionary resOptions;
    resOptions.set("isPC", isPC);
    daResidual_.calcResiduals(resOptions);
    daModel_.calcResiduals(resOptions);

    // assign the calculated residuals to output
    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        const word stateResName = stateName + "Res";
        volVectorField& stateRes = const_cast<volVectorField&>(
            mesh_.thisDb().lookupObject<volVectorField>(stateResName));

        forAll(mesh_.cells(), cellI)
        {
            for (label i = 0; i < 3; i++)
            {
                label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI, i);
                output[localIdx] = stateRes[cellI][i];
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        const word stateResName = stateName + "Res";
        volScalarField& stateRes = const_cast<volScalarField&>(
            mesh_.thisDb().lookupObject<volScalarField>(stateResName));

        forAll(mesh_.cells(), cellI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI);
            output[localIdx] = stateRes[cellI];
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        const word stateResName = stateName + "Res";
        volScalarField& stateRes = const_cast<volScalarField&>(
            mesh_.thisDb().lookupObject<volScalarField>(stateResName));

        forAll(mesh_.cells(), cellI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI);
            output[localIdx] = stateRes[cellI];
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        const word stateResName = stateName + "Res";
        surfaceScalarField& stateRes = const_cast<surfaceScalarField&>(
            mesh_.thisDb().lookupObject<surfaceScalarField>(stateResName));

        forAll(mesh_.faces(), faceI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, faceI);

            if (faceI < daIndex_.nLocalInternalFaces)
            {
                output[localIdx] = stateRes[faceI];
            }
            else
            {
                label relIdx = faceI - daIndex_.nLocalInternalFaces;
                label patchIdx = daIndex_.bFacePatchI[relIdx];
                label faceIdx = daIndex_.bFaceFaceI[relIdx];
                output[localIdx] = stateRes.boundaryFieldRef()[patchIdx][faceIdx];
            }
        }
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
