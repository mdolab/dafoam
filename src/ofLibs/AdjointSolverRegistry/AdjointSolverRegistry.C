/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

\*---------------------------------------------------------------------------*/

#include "AdjointSolverRegistry.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(AdjointSolverRegistry, 0);
defineRunTimeSelectionTable(AdjointSolverRegistry, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointSolverRegistry::AdjointSolverRegistry
(
    const fvMesh& mesh
)
    :
    mesh_(mesh)
{  
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<AdjointSolverRegistry> AdjointSolverRegistry::New
(
    const fvMesh& mesh
)
{
    // get model name, but do not register the dictionary
    // otherwise it is registered in the database twice
    const word modelType
    (
        IOdictionary
        (
            IOobject
            (
                "controlDict",
                mesh.time().system(),
                mesh,
                IOobject::MUST_READ_IF_MODIFIED,
                IOobject::NO_WRITE,
                false
            )
        ).lookup("application")
    );

    Info<< "Selecting " << modelType << " for AdjointSolverRegistry" <<endl;

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn
        (
            "AdjointSolverRegistry::New"
            "("
            "    const fvMesh&"
            ")"
        )   << "Unknown AdjointSolverRegistry type "
            << modelType << nl << nl
            << "Valid AdjointSolverRegistry types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    return autoPtr<AdjointSolverRegistry>
           (
               cstrIter()(mesh)
           );
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void AdjointSolverRegistry::validate()
{
    /// check if the registered states are actually created in the objectRegistry db
    /// Remember to call this after you have registered all the state variables
    
    const objectRegistry& db = mesh_.thisDb();
    
    #define runValidate(stateType,fieldType)                               \
    forAll(stateType,idxI)                                                 \
    {                                                                      \
        bool foundObj = db.foundObject<fieldType>(stateType[idxI]);        \
        if (!foundObj)                                                     \
        {                                                                  \
            FatalErrorIn("")<<stateType[idxI]<<" not found in db!"         \
                            << exit(FatalError);                           \
        }                                                                  \
    }
    
    runValidate(volVectorStates,volVectorField);
    runValidate(volScalarStates,volScalarField);
    runValidate(surfaceScalarStates,surfaceScalarField);
    
    return;
    
}

void AdjointSolverRegistry::setDerivedInfo()
{
    
    // setup the derived info for adjoint states
    
    #define runDerivedInfo(stateType)                                      \
    forAll(stateType,idxI)                                                 \
    {                                                                      \
        stateType##Ref.append(stateType[idxI] + "Ref");                    \
        stateType##Res.append(stateType[idxI] + "Res");                    \
        stateType##ResRef.append(stateType[idxI] + "ResRef");              \
        stateType##ResPartDeriv.append(stateType[idxI] + "ResPartDeriv");  \
    }
    
    runDerivedInfo(volVectorStates);
    runDerivedInfo(volScalarStates);
    runDerivedInfo(surfaceScalarStates);
    
    return;
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
