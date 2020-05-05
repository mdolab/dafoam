/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for buoyantBoussinesqSimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointSolverRegistryBuoyantBoussinesqSimpleFoam_H
#define AdjointSolverRegistryBuoyantBoussinesqSimpleFoam_H

#include "AdjointSolverRegistry.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class AdjointSolverRegistryBuoyantBoussinesqSimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointSolverRegistryBuoyantBoussinesqSimpleFoam
:
public AdjointSolverRegistry
{

private:


public:

    TypeName ("buoyantBoussinesqSimpleDAFoam");
    // Constructors

    //- Construct from components
    AdjointSolverRegistryBuoyantBoussinesqSimpleFoam
    (
        const fvMesh& mesh
    );

    //- Destructor
    virtual ~AdjointSolverRegistryBuoyantBoussinesqSimpleFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
