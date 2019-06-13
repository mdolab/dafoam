/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Register states and residuals for the adjoint

\*---------------------------------------------------------------------------*/

#ifndef AdjointSolverRegistry_H
#define AdjointSolverRegistry_H

#include "runTimeSelectionTables.H"
#include "fvOptions.H"
#include "surfaceFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class AdjointSolverRegistry Declaration
\*---------------------------------------------------------------------------*/

class AdjointSolverRegistry
{

private:

    // Private Member Functions
    const fvMesh& mesh_;

    //- Disallow default bitwise copy construct
    AdjointSolverRegistry(const AdjointSolverRegistry&);

    //- Disallow default bitwise assignment
    void operator=(const AdjointSolverRegistry&);

public:

    //- Runtime type information
    TypeName("adjointSolver");

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable
    (
        autoPtr,
        AdjointSolverRegistry,
        dictionary,
        (
            const fvMesh& mesh
        ),
        (mesh)
    );


    // Constructors

    //- Construct from components
    AdjointSolverRegistry
    (
        const fvMesh& mesh
    );


    // Selectors

    //- Return a reference to the selected model
    static autoPtr<AdjointSolverRegistry> New
    (
        const fvMesh& mesh
    );


    //- Destructor
    virtual ~AdjointSolverRegistry()
    {}

    // These are public variables, so they dont have underscore suffices
    // Adjoint state information to register
    /// state variable names for volVectorState 
    DynamicList<Foam::word> volVectorStates;
    /// state variable names for volScalarState 
    DynamicList<Foam::word> volScalarStates;
    /// state variable names for surfaceScalarState 
    DynamicList<Foam::word> surfaceScalarStates;
    
    // these are derived information
    /// reference state variables names for volVectorState 
    DynamicList<Foam::word> volVectorStatesRef;
    /// reference state variable names for volScalarState 
    DynamicList<Foam::word> volScalarStatesRef;
    /// reference state variable names for surfaceScalarState 
    DynamicList<Foam::word> surfaceScalarStatesRef;
    
    /// residual names for volVectorState 
    DynamicList<Foam::word> volVectorStatesRes;
    /// residual names for volScalarState 
    DynamicList<Foam::word> volScalarStatesRes;
    /// residual names for surfaceScalarState 
    DynamicList<Foam::word> surfaceScalarStatesRes;
    
    /// reference residual names for volVectorState 
    DynamicList<Foam::word> volVectorStatesResRef;
    /// reference residual names for volScalarState 
    DynamicList<Foam::word> volScalarStatesResRef;
    /// reference residual names for surfaceScalarState 
    DynamicList<Foam::word> surfaceScalarStatesResRef;
    
    /// residual partial derivative names
    DynamicList<Foam::word> volVectorStatesResPartDeriv;
    /// residual partial derivative names
    DynamicList<Foam::word> volScalarStatesResPartDeriv;
    /// residual partial derivative names
    DynamicList<Foam::word> surfaceScalarStatesResPartDeriv;
    
    /// setup the derived information based on the registered states
    void setDerivedInfo();
    
    /// Check if the registered information is valid
    void validate();
    

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
