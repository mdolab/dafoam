/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for rhoSimpleCFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointSolverRegistryRhoSimpleCFoam_H
#define AdjointSolverRegistryRhoSimpleCFoam_H

#include "AdjointSolverRegistry.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class AdjointSolverRegistryRhoSimpleCFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointSolverRegistryRhoSimpleCFoam
:
public AdjointSolverRegistry
{

private:


public:

    TypeName ("rhoSimpleCDAFoam");
    // Constructors

    //- Construct from components
    AdjointSolverRegistryRhoSimpleCFoam
    (
        const fvMesh& mesh
    );

    //- Destructor
    virtual ~AdjointSolverRegistryRhoSimpleCFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
