/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAInputStateVar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAInputStateVar, 0);
addToRunTimeSelectionTable(DAInput, DAInputStateVar, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAInputStateVar::DAInputStateVar(
    const word inputName,
    const word inputType,
    fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAInput(
        inputName,
        inputType,
        mesh,
        daOption,
        daModel,
        daIndex)
{
}

void DAInputStateVar::run(const scalarList& input)
{
    /*
    Description:
        Assign the input array to OF's state variables
    */

#ifndef CODI_ADR
    Info << "DAInputStateVar. " << endl;
    Info << "Setting state variables. " << endl;
#endif

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        volVectorField& state = const_cast<volVectorField&>(
            mesh_.thisDb().lookupObject<volVectorField>(stateName));

        forAll(mesh_.cells(), cellI)
        {
            for (label i = 0; i < 3; i++)
            {
                label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI, i);
                state[cellI][i] = input[localIdx];
            }
        }
        state.correctBoundaryConditions();
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        volScalarField& state = const_cast<volScalarField&>(
            mesh_.thisDb().lookupObject<volScalarField>(stateName));

        forAll(mesh_.cells(), cellI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI);
            state[cellI] = input[localIdx];
        }
        state.correctBoundaryConditions();
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        volScalarField& state = const_cast<volScalarField&>(
            mesh_.thisDb().lookupObject<volScalarField>(stateName));

        forAll(mesh_.cells(), cellI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, cellI);
            state[cellI] = input[localIdx];
        }
        state.correctBoundaryConditions();
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        surfaceScalarField& state = const_cast<surfaceScalarField&>(
            mesh_.thisDb().lookupObject<surfaceScalarField>(stateName));

        forAll(mesh_.faces(), faceI)
        {
            label localIdx = daIndex_.getLocalAdjointStateIndex(stateName, faceI);

            if (faceI < daIndex_.nLocalInternalFaces)
            {
                state[faceI] = input[localIdx];
            }
            else
            {
                label relIdx = faceI - daIndex_.nLocalInternalFaces;
                label patchIdx = daIndex_.bFacePatchI[relIdx];
                label faceIdx = daIndex_.bFaceFaceI[relIdx];
                state.boundaryFieldRef()[patchIdx][faceIdx] = input[localIdx];
            }
        }
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
