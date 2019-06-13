/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Macro functions used in the AdjointDerivative class

\*---------------------------------------------------------------------------*/
  
// make state 
#define makeState(stateType,fieldType,classType)                       \
const word stateName = classType.stateType[idxI];                      \
fieldType& state                                                       \
(                                                                      \
    const_cast<fieldType&>                                             \
    (                                                                  \
        db_.lookupObject<fieldType>(stateName)                         \
    )                                                                  \
);

// make state Ref
#define makeStateRef(stateType,fieldType,classType)                    \
const word stateRefName =  classType.stateType[idxI]+"Ref";            \
fieldType& stateRef                                                    \
(                                                                      \
    const_cast<fieldType&>                                             \
    (                                                                  \
        db_.lookupObject<fieldType>(stateRefName)                      \
    )                                                                  \
);

// make res, refRef, and resPartDeriv
#define makeResAll(stateType,fieldType,classType)                          \
const word resName = classType.stateType[idxI]+"Res";                      \
const word resRefName = classType.stateType[idxI]+"ResRef";                \
const word resPartDerivName = classType.stateType[idxI]+"ResPartDeriv";    \
const fieldType& stateRes                                                  \
(                                                                          \
    db_.lookupObject<fieldType>(resName)                                   \
);                                                                         \
const fieldType& stateResRef                                               \
(                                                                          \
    db_.lookupObject<fieldType>(resRefName)                                \
);                                                                         \
fieldType& stateResPartDeriv                                               \
(                                                                          \
    const_cast<fieldType&>                                                 \
    (                                                                      \
        db_.lookupObject<fieldType>(resPartDerivName)                      \
    )                                                                      \
);


// mak resPartDeriv
#define makeResPartDeriv(stateType,fieldType,classType)                    \
const word resPartDerivName = classType.stateType[idxI]+"ResPartDeriv";    \
fieldType& stateResPartDeriv                                               \
(                                                                          \
    const_cast<fieldType&>                                                 \
    (                                                                      \
        db_.lookupObject<fieldType>(resPartDerivName)                      \
    )                                                                      \
);

#define makeRes(stateType,fieldType,classType)                             \
const word resName = classType.stateType[idxI]+"Res";                      \
fieldType& stateRes                                                        \
(                                                                          \
    const_cast<fieldType&>                                                 \
    (                                                                      \
        db_.lookupObject<fieldType>(resName)                               \
    )                                                                      \
);

#define makeResRef(stateType,fieldType,classType)                          \
const word resRefName = classType.stateType[idxI]+"ResRef";                \
fieldType& stateResRef                                                     \
(                                                                          \
    const_cast<fieldType&>                                                 \
    (                                                                      \
        db_.lookupObject<fieldType>(resRefName)                            \
    )                                                                      \
);

#define normalizeResiduals(resName)                                        \
if(!adjIO_.isInList<word>(#resName,adjIO_.normalizeResiduals))             \
{                                                                          \
    forAll(resName##Ref_,cellI)                                            \
    {                                                                      \
        if(isRef)                                                          \
        {                                                                  \
            resName##Ref_[cellI] *= mesh_.V()[cellI];                      \
        }                                                                  \
        else                                                               \
        {                                                                  \
            resName##_[cellI] *= mesh_.V()[cellI];                         \
        }                                                                  \
    }                                                                      \
}

#define scaleResiduals(resName)                                                 \
if(adjIO_.isInList<word>(#resName"Scaling",adjIO_.residualScaling.toc()))       \
{                                                                               \
    forAll(resName##Ref_,cellI)                                                 \
    {                                                                           \
        if(isRef)                                                               \
        {                                                                       \
            resName##Ref_[cellI] /= adjIO_.residualScaling[#resName"Scaling"];  \
        }                                                                       \
        else                                                                    \
        {                                                                       \
            resName##_[cellI] /= adjIO_.residualScaling[#resName"Scaling"];     \
        }                                                                       \
    }                                                                           \
}

#define scalePhiResiduals(resName)                                         \
if(adjIO_.isInList<word>(#resName"Scaling",adjIO_.residualScaling.toc()))  \
{                                                                          \
    forAll(resName##Ref_,faceI)                                            \
    {                                                                      \
        if(isRef)                                                          \
        {                                                                  \
            resName##Ref_[faceI] /= adjIO_.residualScaling[#resName"Scaling"];  \
        }                                                                  \
        else                                                               \
        {                                                                  \
            resName##_[faceI] /= adjIO_.residualScaling[#resName"Scaling"];  \
        }                                                                  \
    }                                                                      \
    forAll(resName##Ref_.boundaryField(),patchI)                           \
    {                                                                      \
        forAll(resName##Ref_.boundaryField()[patchI],faceI)                \
        {                                                                  \
            if(isRef)                                                      \
            {                                                              \
                resName##Ref_.boundaryFieldRef()[patchI][faceI] /= adjIO_.residualScaling[#resName"Scaling"];  \
            }                                                              \
            else                                                           \
            {                                                              \
                resName##_.boundaryFieldRef()[patchI][faceI] /= adjIO_.residualScaling[#resName"Scaling"];  \
            }                                                              \
        }                                                                  \
    }                                                                      \
}

#define normalizePhiResiduals(resName)                                     \
if(adjIO_.isInList<word>(#resName,adjIO_.normalizeResiduals))              \
{                                                                          \
    forAll(resName##Ref_,faceI)                                            \
    {                                                                      \
        if(isRef)                                                          \
        {                                                                  \
            resName##Ref_[faceI] /= mesh_.magSf()[faceI];                  \
        }                                                                  \
        else                                                               \
        {                                                                  \
            resName##_[faceI] /= mesh_.magSf()[faceI];                     \
        }                                                                  \
    }                                                                      \
    forAll(resName##Ref_.boundaryField(),patchI)                           \
    {                                                                      \
        forAll(resName##Ref_.boundaryField()[patchI],faceI)                \
        {                                                                  \
            if(isRef)                                                      \
            {                                                              \
                resName##Ref_.boundaryFieldRef()[patchI][faceI] /= mesh_.magSf().boundaryField()[patchI][faceI]; \
            }                                                              \
            else                                                           \
            {                                                              \
                resName##_.boundaryFieldRef()[patchI][faceI] /= mesh_.magSf().boundaryField()[patchI][faceI];   \
            }                                                              \
        }                                                                  \
    }                                                                      \
}


#define setResidualClassMemberScalar(stateName,stateUnit)                         \
stateName##_                                                                      \
(                                                                                 \
    const_cast<volScalarField&>                                                   \
    (                                                                             \
        db_.lookupObject<volScalarField>(#stateName)                              \
    )                                                                             \
),                                                                                \
stateName##Ref_                                                                   \
(                                                                                 \
    IOobject                                                                      \
    (                                                                             \
        #stateName"Ref",                                                          \
        mesh.time().timeName(),                                                   \
        mesh,                                                                     \
        IOobject::NO_READ,                                                        \
        IOobject::NO_WRITE                                                        \
    ),                                                                            \
    stateName##_                                                                  \
),                                                                                \
stateName##Res_                                                                   \
(                                                                                 \
    IOobject                                                                      \
    (                                                                             \
        #stateName"Res",                                                          \
        mesh.time().timeName(),                                                   \
        mesh,                                                                     \
        IOobject::NO_READ,                                                        \
        IOobject::NO_WRITE                                                        \
    ),                                                                            \
    mesh,                                                                         \
    dimensionedScalar(#stateName"Res",stateUnit,0.0),                             \
    zeroGradientFvPatchScalarField::typeName                                      \
),                                                                                \
stateName##ResRef_                                                                \
(                                                                                 \
    IOobject                                                                      \
    (                                                                             \
        #stateName"ResRef",                                                       \
        mesh.time().timeName(),                                                   \
        mesh,                                                                     \
        IOobject::NO_READ,                                                        \
        IOobject::NO_WRITE                                                        \
    ),                                                                            \
    stateName##Res_                                                               \
),                                                                                \
stateName##ResPartDeriv_                                                          \
(                                                                                 \
    IOobject                                                                      \
    (                                                                             \
        #stateName"ResPartDeriv",                                                 \
        mesh.time().timeName(),                                                   \
        mesh,                                                                     \
        IOobject::NO_READ,                                                        \
        IOobject::NO_WRITE                                                        \
    ),                                                                            \
    stateName##Res_                                                               \
)

#define setResidualClassMemberVector(stateName,stateUnit)                         \
stateName##_                                                                      \
(                                                                                 \
    const_cast<volVectorField&>                                                   \
    (                                                                             \
        db_.lookupObject<volVectorField>(#stateName)                              \
    )                                                                             \
),                                                                                \
stateName##Ref_                                                                   \
(                                                                                 \
    IOobject                                                                      \
    (                                                                             \
        #stateName"Ref",                                                          \
        mesh.time().timeName(),                                                   \
        mesh,                                                                     \
        IOobject::NO_READ,                                                        \
        IOobject::NO_WRITE                                                        \
    ),                                                                            \
    stateName##_                                                                  \
),                                                                                \
stateName##Res_                                                                   \
(                                                                                 \
    IOobject                                                                      \
    (                                                                             \
        #stateName"Res",                                                          \
        mesh.time().timeName(),                                                   \
        mesh,                                                                     \
        IOobject::NO_READ,                                                        \
        IOobject::NO_WRITE                                                        \
    ),                                                                            \
    mesh,                                                                         \
    dimensionedVector(#stateName"Res",stateUnit,vector::zero),                    \
    zeroGradientFvPatchVectorField::typeName                                      \
),                                                                                \
stateName##ResRef_                                                                \
(                                                                                 \
    IOobject                                                                      \
    (                                                                             \
        #stateName"ResRef",                                                       \
        mesh.time().timeName(),                                                   \
        mesh,                                                                     \
        IOobject::NO_READ,                                                        \
        IOobject::NO_WRITE                                                        \
    ),                                                                            \
    stateName##Res_                                                               \
),                                                                                \
stateName##ResPartDeriv_                                                          \
(                                                                                 \
    IOobject                                                                      \
    (                                                                             \
        #stateName"ResPartDeriv",                                                 \
        mesh.time().timeName(),                                                   \
        mesh,                                                                     \
        IOobject::NO_READ,                                                        \
        IOobject::NO_WRITE                                                        \
    ),                                                                            \
    stateName##Res_                                                               \
)

#define setResidualClassMemberPhi(stateName)                                      \
stateName##_                                                                      \
(                                                                                 \
    const_cast<surfaceScalarField&>                                               \
    (                                                                             \
        db_.lookupObject<surfaceScalarField>(#stateName)                          \
    )                                                                             \
),                                                                                \
stateName##Ref_                                                                   \
(                                                                                 \
    IOobject                                                                      \
    (                                                                             \
        #stateName"Ref",                                                          \
        mesh.time().timeName(),                                                   \
        mesh,                                                                     \
        IOobject::NO_READ,                                                        \
        IOobject::NO_WRITE                                                        \
    ),                                                                            \
    stateName##_                                                                  \
),                                                                                \
stateName##Res_                                                                   \
(                                                                                 \
    IOobject                                                                      \
    (                                                                             \
        #stateName"Res",                                                          \
        mesh.time().timeName(),                                                   \
        mesh,                                                                     \
        IOobject::NO_READ,                                                        \
        IOobject::NO_WRITE                                                        \
    ),                                                                            \
    db_.lookupObject<surfaceScalarField>(#stateName)                              \
),                                                                                \
stateName##ResRef_                                                                \
(                                                                                 \
    IOobject                                                                      \
    (                                                                             \
        #stateName"ResRef",                                                       \
        mesh.time().timeName(),                                                   \
        mesh,                                                                     \
        IOobject::NO_READ,                                                        \
        IOobject::NO_WRITE                                                        \
    ),                                                                            \
    db_.lookupObject<surfaceScalarField>(#stateName)                              \
),                                                                                \
stateName##ResPartDeriv_                                                          \
(                                                                                 \
    IOobject                                                                      \
    (                                                                             \
        #stateName"ResPartDeriv",                                                 \
        mesh.time().timeName(),                                                   \
        mesh,                                                                     \
        IOobject::NO_READ,                                                        \
        IOobject::NO_WRITE                                                        \
    ),                                                                            \
    db_.lookupObject<surfaceScalarField>(#stateName)                              \
)

#define setFvMatrix(stateName,fvMatrix)    \
if( fvMatrixName==stateName"Eqn" )         \
{                                          \
    fvMatrixDiag.clear();                  \
    fvMatrixLower.clear();                 \
    fvMatrixUpper.clear();                 \
    fvMatrixDiag  = fvMatrix.diag();       \
    fvMatrixLower = fvMatrix.lower();      \
    fvMatrixUpper = fvMatrix.upper();      \
}                                          \

