/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

    Description:
    Child class for turboFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointSolverRegistryTurboFoam_H
#define AdjointSolverRegistryTurboFoam_H

#include "AdjointSolverRegistry.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class AdjointSolverRegistryTurboFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointSolverRegistryTurboFoam
:
public AdjointSolverRegistry
{

private:


public:

    TypeName ("turboDAFoam");
    // Constructors

    //- Construct from components
    AdjointSolverRegistryTurboFoam
    (
        const fvMesh& mesh
    );

    //- Destructor
    virtual ~AdjointSolverRegistryTurboFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
