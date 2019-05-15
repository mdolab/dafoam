/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

    Description:
    Child class for laplacianFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointJacobianConnectivityLaplacianFoam_H
#define AdjointJacobianConnectivityLaplacianFoam_H

#include "AdjointJacobianConnectivity.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class AdjointJacobianConnectivityLaplacianFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointJacobianConnectivityLaplacianFoam
:
public AdjointJacobianConnectivity
{

private:


public:

    TypeName ("laplacianDAFoam");
    // Constructors

    //- Construct from components
    AdjointJacobianConnectivityLaplacianFoam
    (
        const fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx
    );

    //- Destructor
    virtual ~AdjointJacobianConnectivityLaplacianFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
