/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for rhoSimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointSolverRegistryRhoSimpleFoam_H
#define AdjointSolverRegistryRhoSimpleFoam_H

#include "AdjointSolverRegistry.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class AdjointSolverRegistryRhoSimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointSolverRegistryRhoSimpleFoam
:
public AdjointSolverRegistry
{

private:


public:

    TypeName ("rhoSimpleDAFoam");
    // Constructors

    //- Construct from components
    AdjointSolverRegistryRhoSimpleFoam
    (
        const fvMesh& mesh
    );

    //- Destructor
    virtual ~AdjointSolverRegistryRhoSimpleFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
