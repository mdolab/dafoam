/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for simpleTFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointJacobianConnectivitySimpleTFoam_H
#define AdjointJacobianConnectivitySimpleTFoam_H

#include "AdjointJacobianConnectivity.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class AdjointJacobianConnectivitySimpleTFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointJacobianConnectivitySimpleTFoam
:
public AdjointJacobianConnectivity
{

private:


public:

    TypeName ("simpleTDAFoam");
    // Constructors

    //- Construct from components
    AdjointJacobianConnectivitySimpleTFoam
    (
        const fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx
    );

    //- Destructor
    virtual ~AdjointJacobianConnectivitySimpleTFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
