/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for buoyantSimpleFoam

\*---------------------------------------------------------------------------*/
#ifndef AdjointSolverRegistryBuoyantSimpleFoam_H
#define AdjointSolverRegistryBuoyantSimpleFoam_H

#include "AdjointSolverRegistry.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class AdjointSolverRegistryBuoyantSimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointSolverRegistryBuoyantSimpleFoam
:
public AdjointSolverRegistry
{

private:


public:

    TypeName ("buoyantSimpleDAFoam");
    // Constructors

    //- Construct from components
    AdjointSolverRegistryBuoyantSimpleFoam
    (
        const fvMesh& mesh
    );

    //- Destructor
    virtual ~AdjointSolverRegistryBuoyantSimpleFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
