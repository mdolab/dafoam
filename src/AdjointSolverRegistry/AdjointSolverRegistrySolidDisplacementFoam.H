/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for solidDisplacementFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointSolverRegistrySolidDisplacementFoam_H
#define AdjointSolverRegistrySolidDisplacementFoam_H

#include "AdjointSolverRegistry.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class AdjointSolverRegistrySolidDisplacementFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointSolverRegistrySolidDisplacementFoam
:
public AdjointSolverRegistry
{

private:


public:

    TypeName ("solidDisplacementDAFoam");
    // Constructors

    //- Construct from components
    AdjointSolverRegistrySolidDisplacementFoam
    (
        const fvMesh& mesh
    );

    //- Destructor
    virtual ~AdjointSolverRegistrySolidDisplacementFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
