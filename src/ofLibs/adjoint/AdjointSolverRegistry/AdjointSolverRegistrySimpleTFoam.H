/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for simpleTFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointSolverRegistrySimpleTFoam_H
#define AdjointSolverRegistrySimpleTFoam_H

#include "AdjointSolverRegistry.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class AdjointSolverRegistrySimpleTFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointSolverRegistrySimpleTFoam
:
public AdjointSolverRegistry
{

private:


public:

    TypeName ("simpleTDAFoam");
    // Constructors

    //- Construct from components
    AdjointSolverRegistrySimpleTFoam
    (
        const fvMesh& mesh
    );

    //- Destructor
    virtual ~AdjointSolverRegistrySimpleTFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
