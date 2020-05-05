/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for simpleFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointSolverRegistrySimpleFoam_H
#define AdjointSolverRegistrySimpleFoam_H

#include "AdjointSolverRegistry.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class AdjointSolverRegistrySimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointSolverRegistrySimpleFoam
:
public AdjointSolverRegistry
{

private:


public:

    TypeName ("simpleDAFoam");
    // Constructors

    //- Construct from components
    AdjointSolverRegistrySimpleFoam
    (
        const fvMesh& mesh
    );

    //- Destructor
    virtual ~AdjointSolverRegistrySimpleFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
