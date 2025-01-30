/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Macro functions for DAFoam

\*---------------------------------------------------------------------------*/

#ifndef DAMacroFunctions_H
#define DAMacroFunctions_H

// make state
#define makeState(regState, fieldType, db) \
    const word stateName = regState;       \
    fieldType& state(                      \
        const_cast<fieldType&>(            \
            db.lookupObject<fieldType>(stateName)));

#define makeStateRes(regState, fieldType, db)    \
    const word stateName = regState;             \
    const word stateResName = stateName + "Res"; \
    fieldType& stateRes(                         \
        const_cast<fieldType&>(                  \
            db.lookupObject<fieldType>(stateResName)));

#define normalizeResiduals(resName)                                           \
    if (!daOption_.getOption<wordList>("normalizeResiduals").found(#resName)) \
    {                                                                         \
        forAll(resName##_, cellI)                                             \
        {                                                                     \
            resName##_[cellI] *= mesh_.V()[cellI];                            \
        }                                                                     \
    }

#define normalizePhiResiduals(resName)                                                                        \
    if (daOption_.getOption<wordList>("normalizeResiduals").found(#resName))                                  \
    {                                                                                                         \
        forAll(resName##_, faceI)                                                                             \
        {                                                                                                     \
            resName##_[faceI] /= mesh_.magSf()[faceI];                                                        \
        }                                                                                                     \
        forAll(resName##_.boundaryField(), patchI)                                                            \
        {                                                                                                     \
            forAll(resName##_.boundaryField()[patchI], faceI)                                                 \
            {                                                                                                 \
                resName##_.boundaryFieldRef()[patchI][faceI] /= mesh_.magSf().boundaryField()[patchI][faceI]; \
            }                                                                                                 \
        }                                                                                                     \
    }

#define setResidualClassMemberScalar(stateName, stateUnit)            \
    stateName##_(                                                     \
        const_cast<volScalarField&>(                                  \
            mesh.thisDb().lookupObject<volScalarField>(#stateName))), \
        stateName##Res_(                                              \
            IOobject(                                                 \
                #stateName "Res",                                     \
                mesh.time().timeName(),                               \
                mesh,                                                 \
                IOobject::NO_READ,                                    \
                IOobject::NO_WRITE),                                  \
            mesh,                                                     \
            dimensionedScalar(#stateName "Res", stateUnit, 0.0),      \
            zeroGradientFvPatchField<scalar>::typeName)

#define setResidualClassMemberVector(stateName, stateUnit)                \
    stateName##_(                                                         \
        const_cast<volVectorField&>(                                      \
            mesh.thisDb().lookupObject<volVectorField>(#stateName))),     \
        stateName##Res_(                                                  \
            IOobject(                                                     \
                #stateName "Res",                                         \
                mesh.time().timeName(),                                   \
                mesh,                                                     \
                IOobject::NO_READ,                                        \
                IOobject::NO_WRITE),                                      \
            mesh,                                                         \
            dimensionedVector(#stateName "Res", stateUnit, vector::zero), \
            zeroGradientFvPatchField<vector>::typeName)

#define setResidualClassMemberPhi(stateName)                              \
    stateName##_(                                                         \
        const_cast<surfaceScalarField&>(                                  \
            mesh.thisDb().lookupObject<surfaceScalarField>(#stateName))), \
        stateName##Res_(                                                  \
            IOobject(                                                     \
                #stateName "Res",                                         \
                mesh.time().timeName(),                                   \
                mesh,                                                     \
                IOobject::NO_READ,                                        \
                IOobject::NO_WRITE),                                      \
            mesh.thisDb().lookupObject<surfaceScalarField>(#stateName))

#endif

// If AD is used, do valA=valB.getValue(), otherwise, do valA=valB
#if defined(CODI_ADF) || defined(CODI_ADR)
#define assignValueCheckAD(valA, valB) \
    valA = valB.getValue();
#else
#define assignValueCheckAD(valA, valB) \
    valA = valB;
#endif
