/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

    Description:
    Child class for scalarTransportFoam

\*---------------------------------------------------------------------------*/
#ifndef AdjointSolverRegistryScalarTransportFoam_H
#define AdjointSolverRegistryScalarTransportFoam_H

#include "AdjointSolverRegistry.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class AdjointSolverRegistryScalarTransportFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointSolverRegistryScalarTransportFoam
:
public AdjointSolverRegistry
{

private:


public:

    TypeName ("scalarTransportDAFoam");
    // Constructors

    //- Construct from components
    AdjointSolverRegistryScalarTransportFoam
    (
        const fvMesh& mesh
    );

    //- Destructor
    virtual ~AdjointSolverRegistryScalarTransportFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
