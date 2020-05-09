/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for rhoSimpleCFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointJacobianConnectivityRhoSimpleCFoam_H
#define AdjointJacobianConnectivityRhoSimpleCFoam_H

#include "AdjointJacobianConnectivity.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class AdjointJacobianConnectivityRhoSimpleCFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointJacobianConnectivityRhoSimpleCFoam
:
public AdjointJacobianConnectivity
{

private:


public:

    TypeName ("rhoSimpleCDAFoam");
    // Constructors

    //- Construct from components
    AdjointJacobianConnectivityRhoSimpleCFoam
    (
        const fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx
    );

    //- Destructor
    virtual ~AdjointJacobianConnectivityRhoSimpleCFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
