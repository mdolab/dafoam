/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

    Description:
    Child class for laplacianFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointSolverRegistryLaplacianFoam_H
#define AdjointSolverRegistryLaplacianFoam_H

#include "AdjointSolverRegistry.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class AdjointSolverRegistryLaplacianFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointSolverRegistryLaplacianFoam
:
public AdjointSolverRegistry
{

private:


public:

    TypeName ("laplacianDAFoam");
    // Constructors

    //- Construct from components
    AdjointSolverRegistryLaplacianFoam
    (
        const fvMesh& mesh
    );

    //- Destructor
    virtual ~AdjointSolverRegistryLaplacianFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
