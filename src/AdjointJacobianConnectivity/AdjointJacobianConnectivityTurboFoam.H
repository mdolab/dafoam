/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

    Description:
    Child class for turboFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointJacobianConnectivityTurboFoam_H
#define AdjointJacobianConnectivityTurboFoam_H

#include "AdjointJacobianConnectivity.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class AdjointJacobianConnectivityTurboFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointJacobianConnectivityTurboFoam
:
public AdjointJacobianConnectivity
{

private:


public:

    TypeName ("turboDAFoam");
    // Constructors

    //- Construct from components
    AdjointJacobianConnectivityTurboFoam
    (
        const fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx
    );

    //- Destructor
    virtual ~AdjointJacobianConnectivityTurboFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
