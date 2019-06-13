/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for solidDisplacementFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointJacobianConnectivitySolidDisplacementFoam_H
#define AdjointJacobianConnectivitySolidDisplacementFoam_H

#include "AdjointJacobianConnectivity.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class AdjointJacobianConnectivitySolidDisplacementFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointJacobianConnectivitySolidDisplacementFoam
:
public AdjointJacobianConnectivity
{

private:


public:

    TypeName ("solidDisplacementDAFoam");
    // Constructors

    //- Construct from components
    AdjointJacobianConnectivitySolidDisplacementFoam
    (
        const fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx
    );

    //- Destructor
    virtual ~AdjointJacobianConnectivitySolidDisplacementFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
